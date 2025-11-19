import logging
import os
import sys
import random
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


from matplotlib import pyplot as plt
from einops import rearrange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
from torchvision.transforms import transforms
from tqdm import tqdm, trange
from omegaconf import OmegaConf
import numpy as np
from sklearn.metrics import roc_curve, auc
from torch.utils.tensorboard import SummaryWriter

import utils
from match import find_matching_pairs
from model import EncoderCosine, EncoderDistance


def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

def dev(device=None):
    """
    Get the device to use for torch.distributed.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device(int(os.environ['LOCAL_RANK']))
        return torch.device("cpu")
    return torch.device(device)



def parse_args():
    parser = argparse.ArgumentParser(description="Training job configuration")

    # Add arguments
    parser.add_argument('conf_path', type=str, help='Path to the configuration file')
    parser.add_argument('job_id', type=str, help='Job ID')
    parser.add_argument('--clear_cache', default=True, action='store_true', help='Clear cache directory')
    # Parse the arguments
    args = parser.parse_args()
    print(args)

    return args


class LeCunContrastiveLoss(nn.Module):
    """
    LeCun's contrastive loss function for training models in a contrastive learning setup.
    """
    def __init__(self, margin=1.0, METHOD='COS'):
        super(LeCunContrastiveLoss, self).__init__()
        self.margin = margin
        self.METHOD = METHOD

    def forward(self, similar, dissimilar):
        if self.METHOD == 'L2' or self.METHOD == 'L1':
            loss = torch.clamp(similar, min=0).mean() + torch.clamp(self.margin - dissimilar, min=0).mean()
        elif self.METHOD == 'COS':
            loss = torch.clamp(self.margin - similar.mean() + dissimilar.mean(), min=0)
        return loss

def compute_embeddings(model, dataloader, device, requires_grad=False, mapping=None,
                       embedding_dim=4096):
    """
    Computes embeddings for the given dataloader using the provided model.
    Args:
        model (torch.nn.Module): The model to use for computing embeddings.
        dataloader (DataLoader): The DataLoader containing the data.
        device (torch.device): The device to run the model on.
        requires_grad (bool): Whether the embeddings should require gradients.
        mapping (dict): A mapping from (group, id, file_name) to index in the embedding bank.
        embedding_dim (int): The dimension of the embeddings.
    Returns:
        torch.nn.Embedding: An embedding bank containing the computed embeddings.
    """
    model.eval()
    num_embeddings = len(mapping.keys())
    embeding_dim = embedding_dim
    embedding_bank = torch.nn.Embedding(num_embeddings, embeding_dim)
    embedding_bank = embedding_bank.to(device)
    pbar = tqdm(dataloader, desc="Computing embeddings", unit="batch", dynamic_ncols=True)
    if requires_grad:
        for images, groups, ids, file_names, idx in pbar:
            output = model(images.to(device))
            return output
    else:
        with torch.no_grad():
            for images, groups, ids, file_names, _ in pbar:
                output = model(images.to(device))
                indicies = torch.tensor([mapping[(group.item(), id.item(), file_name)] for group, id, file_name in zip(groups, ids, file_names)],
                                        device=device, dtype=torch.long)
                embedding_bank.weight.data[indicies] = output

        return embedding_bank

def select_pairs_from_embedding(precomputed_embedding_bank,
                                metadata_to_id,
                                current_embeddings,
                                current_labels,
                                current_ids,
                                percentage=0.2,
                                METHOD='COS'):
    """
    Rychlá verze:
      • Jednou předpočítáme arrays `labels` a `ids` z metadata_to_id
      • Similarity/Distance matice jednou (B×N)
      • Pro každý vzorek použijeme torch.where + torch.topk (jen K nejlepších)
      • Loop jen přes batch dimenzi (B), ne přes všechny N embeddingů
    """
    # 1) Normalizace / výpočet similarity nebo distance
    current_embeddings = F.normalize(current_embeddings, dim=1)
    bank_weight = precomputed_embedding_bank.weight
    if METHOD == 'COS':
        bank_norm = F.normalize(bank_weight, dim=1)
        sim = current_embeddings @ bank_norm.T  # [B, N]
    else:
        # Rozbalení do (B,1,D) vs (1,N,D) a cdist
        p = 2 if METHOD == 'L2' else 1
        sim = torch.cdist(current_embeddings, bank_weight, p=p)  # distances [B, N]

    B, N = sim.shape

    # 2) Předpočítat dvě polia, labels[idx] a ids[idx]
    device = sim.device
    # Assuming metadata_to_id: {(label, id, fname): idx, ...}
    labels = torch.empty(N, dtype=current_labels.dtype, device=device)
    ids    = torch.empty(N, dtype=current_ids.dtype,    device=device)
    # A seznam pro rychlý lookup metadata
    id_to_metadata = [None] * N
    for (lbl, id_, fname), idx in metadata_to_id.items():
        labels[idx] = lbl
        ids[idx]    = id_
        id_to_metadata[idx] = (lbl, id_, fname)

    # 3) Pro každý vzorek B najdeme K pozitiv + K negativ
    all_indices = torch.arange(N, device=device)
    pairs = []

    for i in range(B):
        row = sim[i]  # [N]
        lbl = current_labels[i]
        id_ = current_ids[i]

        mask_same = (labels == lbl) & (ids == id_)
        mask_diff = ~mask_same

        # kolik vzít z každé kategorie
        cnt_same = mask_same.sum().item()
        cnt_diff = mask_diff.sum().item()
        if percentage != -1:
            k = min(int(cnt_same * percentage), int(cnt_diff * percentage))
        else:
            k = 1
        if k <= 0:
            continue  # nic k párování

        if METHOD == 'COS':
            # COS: vyšší = blíž → pozitivní furthest = nejnižší similarity
            same_idx = all_indices[mask_same]
            same_vals = row[mask_same]
            # topk nejmenších
            furthest_k = torch.topk(same_vals, k, largest=False).indices
            furthest_same = same_idx[furthest_k]

            diff_idx = all_indices[mask_diff]
            diff_vals = row[mask_diff]
            # topk největších
            closest_k = torch.topk(diff_vals, k, largest=True).indices
            closest_diff = diff_idx[closest_k]

        else:
            # L2/L1: menší = blíž → pozitivní furthest = největší distance
            same_idx = all_indices[mask_same]
            same_vals = row[mask_same]
            furthest_k = torch.topk(same_vals, k, largest=True).indices
            furthest_same = same_idx[furthest_k]

            diff_idx = all_indices[mask_diff]
            diff_vals = row[mask_diff]
            closest_k = torch.topk(diff_vals, k, largest=False).indices
            closest_diff = diff_idx[closest_k]

        # 4) Přidat výsledné páry (i, metadata)
        for j in furthest_same:
            pairs.append((i, id_to_metadata[j.item()]))
        for j in closest_diff:
            pairs.append((i, id_to_metadata[j.item()]))

    return pairs


def compute_contrastive_loss(embeddings,
                             pairs,
                             labels,
                             ids,
                             criterion,
                             device,
                             precomputed_embeddings,
                             mapping=None,
                             METHOD='COS'):
    """
    Vectorizovaná a zrychlená verze výpočtu kontrastivní ztráty.
    Oprava: labels a ids se hned přesunou na device, aby pasovaly s batch_idx.
    """
    # --- Přesun labelů/ID na stejný device jako batch_idx ---
    labels = labels.to(device,    non_blocking=True)
    ids    = ids.to(device,       non_blocking=True)

    P = len(pairs)
    if P == 0:
        return torch.tensor(0., device=device, requires_grad=True)

    # 1) Rozbalíme seznam párů do GPU‐tenzorů
    batch_idx = torch.tensor([i for i, _ in pairs],
                             device=device, dtype=torch.long)
    meta = [m for _, m in pairs]
    lab_j = torch.tensor([m[0] for m in meta],
                         device=device, dtype=labels.dtype)
    id_j  = torch.tensor([m[1] for m in meta],
                         device=device, dtype=ids.dtype)
    pre_idx = torch.tensor([mapping[m] for m in meta],
                           device=device, dtype=torch.long)

    # 2) Natáhneme embeddings jedním indexováním
    c_embs = embeddings[batch_idx]                       # [P, D]
    p_embs = precomputed_embeddings.weight[pre_idx]      # [P, D]

    # 3) Spočteme všechny vzdálenosti najednou
    if METHOD == 'COS':
        dists = F.cosine_similarity(p_embs, c_embs, dim=1)  # [P]
    else:
        p = 2 if METHOD == 'L2' else 1
        dists = F.pairwise_distance(p_embs, c_embs, p=p)   # [P]

    # 4) Rozdělíme na pozitivní vs negativní
    sim_mask  = (labels[batch_idx] == lab_j) & (ids[batch_idx] == id_j)
    sim_dists = dists[sim_mask]
    diff_dists= dists[~sim_mask]

    # 5) Contrastive loss
    loss = criterion(sim_dists, diff_dists)
    return loss


def train_model(model, criterion, optimizer, bank, dataloader, mapping=None, pair_percentage=0.2, similarity_measure ='COS'):
    """
    Train the model for one epoch using the provided dataloader and embeddings.
    Args:
        model (torch.nn.Module): The model to train.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        embedings (torch.nn.Embedding): The embedding bank containing precomputed embeddings.
        dataloader (DataLoader): The DataLoader for the training dataset.
        mapping (dict): A mapping from (group, id, file_name) to index in the embedding bank.
        pair_percentage (float): The percentage of pairs to select for contrastive loss computation.
    METHOD (str): The method to use for computing distances ('COS', 'L2', 'L1').
    Returns:
        float: The total loss for the training dataset.
        Embedding bank: The embedding bank containing the computed embeddings.
    """
    device = dev()
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training", unit="batch", dynamic_ncols=True)
    for images, groups, ids, file_names, _ in pbar:
        optimizer.zero_grad()
        output = model(images.to(device))
        pairs = select_pairs_from_embedding(bank, mapping, output, groups, ids, pair_percentage, METHOD=similarity_measure)
        loss = compute_contrastive_loss(output, pairs, groups, ids, criterion, device, bank, mapping, METHOD=similarity_measure)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_description(f"TRN: {loss.item():.4f}")

        with torch.no_grad():
            idx = torch.tensor(
                [mapping[(g.item(), i.item(), f)] for g, i, f in zip(groups, ids, file_names)],
                device=device,
                dtype=torch.long,
            )
            bank.weight.data[idx] = output.detach()   # overwrite with fresh output

    return total_loss, bank


def test_model(model, criterion, embedings, dataloader, mapping=None,
               pair_percentage=0.2, similarity_measure='COS'):
    """
    Test the model using the provided dataloader and embeddings.
    Args:
        model (torch.nn.Module): The model to test.
        criterion (nn.Module): The loss function.
        embedings (torch.nn.Embedding): The embedding bank containing precomputed embeddings.
        dataloader (DataLoader): The DataLoader for the validation dataset.
        mapping (dict): A mapping from (group, id, file_name) to index in the embedding bank.
        pair_percentage (float): The percentage of pairs to select for contrastive loss computation.
        similarity_measure (str): The method to use for computing distances ('COS', 'L2', 'L1').
    Returns:
        float: The total loss for the validation dataset.
        Embedding bank: The embedding bank containing the computed embeddings.
    """
    device = dev()
    model.eval()
    total_loss = 0
    for images, groups, ids, file_names, idx in dataloader:
        output = model(images.to(device))
        # pairs = select_pairs_from_dict(embedings, output, groups, ids)
        pairs = select_pairs_from_embedding(embedings, mapping, output, groups, ids, pair_percentage, METHOD=similarity_measure)
        loss = compute_contrastive_loss(output, pairs, groups, ids, criterion, device, embedings, mapping, METHOD=similarity_measure)
        total_loss += loss.item()
    return total_loss, embedings



def get_tpr_at_fpr(fpr, tpr, target_fpr=0.05):
    """
    Get the True Positive Rate (TPR) at a specified False Positive Rate (FPR) (closest lower or equal to target_fpr).
    Args:
        fpr (np.ndarray): Array of false positive rates.
        tpr (np.ndarray): Array of true positive rates.
        target_fpr (float): The target false positive rate to find the corresponding TPR for.
    returns:
        float: The True Positive Rate at the specified False Positive Rate.
    """
    indices = np.where(fpr <= target_fpr)[0]
    if len(indices) == 0:
        return 0.0
    return tpr[indices[-1]]  # highest threshold satisfying FPR <= c

def plot_roc(epoch, auc_trn, auc_val, fpr_trn, tpr_trn, fpr_val, tpr_val, save_dir):
    """
    Plot the ROC curve for training and validation datasets and save the plot.
    Args:
        epoch (int): The current epoch number.
        auc_trn (float): The AUC for the training dataset.
        auc_val (float): The AUC for the validation dataset.
        fpr_trn (np.ndarray): False positive rates for the training dataset.
        tpr_trn (np.ndarray): True positive rates for the training dataset.
        fpr_val (np.ndarray): False positive rates for the validation dataset.
        tpr_val (np.ndarray): True positive rates for the validation dataset.
        save_dir (str): The directory to save the ROC curve plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_trn, tpr_trn, color='blue', label=f'Training AUC = {auc_trn:.2f}')
    plt.plot(fpr_val, tpr_val, color='red', label=f'Validation AUC = {auc_val:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Epoch {epoch}')
    plt.legend(loc='lower right')
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    roc_pdf_path = os.path.join(save_dir, 'roc_pdf')
    os.makedirs(roc_pdf_path, exist_ok=True)

    plt.savefig(os.path.join(roc_pdf_path, f'roc_epoch_{epoch}.pdf'))
    plt.close()


@torch.inference_mode()                     # cheaper than no_grad()
def compute_roc(save_dir, epoch,
                tr_bank, val_bank,
                map_trn, map_val,
                similarity_measure: str = "COS"):
    """
    Vectorised ROC computation.  ≈ 100× fewer Python ops, memory‑controllable.
    """

    def _pairwise_scores(emb_bank, mapping):
        # ----- 1 . gather all embeddings for this split -----
        keys          = list(mapping.keys())
        idx           = torch.as_tensor([mapping[k] for k in keys], device=emb_bank.weight.device)
        E             = emb_bank(idx)                        # (N, D)

        # ----- 2 . pre‑normalise once (if required) -----
        if similarity_measure == "COS" or similarity_measure.startswith("L"):
            E = torch.nn.functional.normalize(E, dim=1)      # in‑place is fine

        # ----- 3 . similarity / distance matrix -----
        if similarity_measure == "COS":
            S = E @ E.T                                      # (N, N) cosine sim
        else:  # L2 or L1
            p = 2 if similarity_measure == "L2" else 1
            S = -torch.cdist(E, E, p=p)                      # negative distance = similarity

        # ----- 4 . label matrix & masking -----
        g, i, f = zip(*keys)
        g = torch.tensor(g, device=S.device)
        i = torch.tensor(i, device=S.device)
        f = torch.tensor([hash(x) for x in f], device=S.device)

        same_id   = (g[:, None] == g) & (i[:, None] == i)     # positive pairs
        diff_file = f[:, None] != f                           # ignore identical files
        keep      = diff_file.triu(1)                         # upper tri ⇢ no self/dup

        y   = same_id[keep].cpu().numpy().astype(int)
        sco = S[keep].cpu().numpy()          # scores; already negated for L‑metrics

        return y, sco

    y_trn, s_trn = _pairwise_scores(tr_bank,  map_trn)
    y_val, s_val = _pairwise_scores(val_bank, map_val)

    fpr_trn, tpr_trn, _ = roc_curve(y_trn,  s_trn)
    fpr_val, tpr_val, _ = roc_curve(y_val,  s_val)
    auc_trn = auc(fpr_trn, tpr_trn)
    auc_val = auc(fpr_val, tpr_val)

    tpr5_trn = get_tpr_at_fpr(fpr_trn, tpr_trn, 0.05)
    tpr5_val = get_tpr_at_fpr(fpr_val, tpr_val, 0.05)

    #plot_roc(epoch, auc_trn, auc_val,
    #         fpr_trn, tpr_trn, fpr_val, tpr_val, save_dir)
    return auc_trn, auc_val, tpr5_trn, tpr5_val


def create_kfold_splits(root_dir, extensions=('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'pt', 'h5'), n_splits=10,
                        subsample=None, matches = None, logger = None):
    data = {'F': {}, 'M': {}}
    for filename in tqdm(os.listdir(root_dir)):
        if matches is not None and filename not in matches:
            continue
        if filename.endswith(extensions):
            group = filename[0]
            if group in ['F', 'M']:
                identifier = ''
                for char in filename[1:]:
                    if char.isdigit():
                        identifier += char
                    else:
                        break

                identifier = int(identifier) if identifier else None

                if identifier is not None:
                    if identifier not in data[group]:
                        data[group][identifier] = []
                    data[group][identifier].append(filename)
    # Prepare list of identities for both groups
    f_ids = list(data['F'].keys())
    m_ids = list(data['M'].keys())

    # Shuffle identities for random splitting
    random.shuffle(f_ids)
    random.shuffle(m_ids)

    # Split identities into n_splits folds
    f_fold_size = len(f_ids) // n_splits
    m_fold_size = len(m_ids) // n_splits

    f_folds = [f_ids[i * f_fold_size:(i + 1) * f_fold_size] for i in range(n_splits)]
    m_folds = [m_ids[i * m_fold_size:(i + 1) * m_fold_size] for i in range(n_splits)]

    # Adjust last fold to include any remaining identities
    f_folds[-1].extend(f_ids[n_splits * f_fold_size:])
    m_folds[-1].extend(m_ids[n_splits * m_fold_size:])

    # Generate the splits
    splits = []
    for i in trange(n_splits):
        train_files, val_files = {'F': {}, 'M': {}}, {'F': {}, 'M': {}}

        for j in range(n_splits):
            if j == i:
                for fid in f_folds[j]:
                    if fid not in val_files['F']:
                        val_files['F'][fid] = []
                    if subsample is None:
                        val_files['F'][fid].extend(data['F'][fid])
                    else:
                        val_files['F'][fid].extend(data['F'][fid][:subsample])
                for mid in m_folds[j]:
                    if mid not in val_files['M']:
                        val_files['M'][mid] = []
                    if subsample is None:
                        val_files['M'][mid].extend(data['M'][mid])
                    else:
                        val_files['M'][mid].extend(data['M'][mid][:subsample])
            else:
                for fid in f_folds[j]:
                    if fid not in train_files['F']:
                        train_files['F'][fid] = []
                    if subsample is None:
                        train_files['F'][fid].extend(data['F'][fid])
                    else:
                        train_files['F'][fid].extend(data['F'][fid][:subsample])
                for mid in m_folds[j]:
                    if mid not in train_files['M']:
                        train_files['M'][mid] = []
                    if subsample is None:
                        train_files['M'][mid].extend(data['M'][mid])
                    else:
                        train_files['M'][mid].extend(data['M'][mid][:subsample])

        splits.append((train_files, val_files))
    # for each split print number of train and val files
    if logger is not None:
        for split in splits:
            logger.debug(f"Number of train files: {sum(len(split[0][group][identifier]) for group in split[0] for identifier in split[0][group])}")
            logger.debug(f"Number of val files: {sum(len(split[1][group][identifier]) for group in split[1] for identifier in split[1][group])}")
    return splits


def set_all_seeds(seed=42):
    """
    Set the random seed for Python, NumPy, and PyTorch (CPU and CUDA) to ensure reproducibility.

    Args:
        seed (int): The seed value to use for all random number generators. Default is 42.

    This function helps make experiments deterministic by controlling sources of randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clear_cache(cache_dir):
    """
    Clear the cache directory by removing all files in it. If the directory does not exist, it will be created.
    Args:
        cache_dir (str): The path to the cache directory to clear.
    """
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            os.remove(os.path.join(root, file))

    # now dirs
    for root, dirs, _ in os.walk(cache_dir):
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))


def prepare_train_val_files(splits, fold_num):
    """
    Extract training and validation files from the k-fold splits based on the specified fold number.
    Args:
        splits (list): A list of tuples containing training and validation files for each fold.
        fold_num (int): The index of the fold to use for training and validation.
    Returns:
        tuple: A tuple containing two lists: training files and validation files.
    """
    if 'trn' in splits[fold_num]:
        return splits[fold_num]['trn'], splits[fold_num]['val']

    tr, ve = splits[int(fold_num)]
    train_files = []
    val_files = []
    for group in tr:
        for identifier in tr[group]:
            train_files.extend(tr[group][identifier])
    for group in ve:
        for identifier in ve[group]:
            val_files.extend(ve[group][identifier])
    return train_files, val_files

def create_mapping(trn_dataset, val_dataset):
    """
    Create mappings from (group, id, file_name) to index in the all_dataloader and all_val_dataloader.
    Args:
        all_dataloader (DataLoader): DataLoader for the training dataset.
        all_val_dataloader (DataLoader): DataLoader for the validation dataset.

    Returns:
        tuple: Two dictionaries, mapping for training and validation datasets.
    """
    mapping_trn = {}
    mapping_val = {}
    # Create mappings for training and validation datasets
    for idx in range(len(trn_dataset)):
        group, id, file_name = trn_dataset.all_images[idx]
        group = 1.0 if group == 'M' else 0.0

        mapping_trn[(group, id, file_name)] = idx

    for idx in range(len(val_dataset)):
        group, id, file_name = val_dataset.all_images[idx]
        group = 1.0 if group == 'M' else 0.0

        mapping_val[(group, id, file_name)] = idx

    return mapping_trn, mapping_val

def plot_loss(epoch, training_losses, validation_losses, plot_dir):
    plt.figure(figsize=(13, 8))
    plt.plot(training_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'losses_epoch_' + str(epoch + 1) + '.pdf'))
    plt.close()


def main():
    #Set random seed for reproducibility
    set_all_seeds(42)
    args = parse_args()

    logger = logging.getLogger()

    config = OmegaConf.load(args.conf_path)

    dataset_dir = os.environ["DATASET_DIR"]
    h5_dataset_dir = "PATH_TO_YOUR_CHANNEL_REPRESENTATION_FILES_STORED_IN_HDF5"
    experiment_dir = os.path.join(os.environ['EXPERIMENTS_DIR'], args.job_id)
    hsd_dataset_dir = dataset_dir + "/HSD/"
    splits_filepath = dataset_dir + '/splits.txt'
    master_mask_filepath = dataset_dir + "/master_mask.png"

    os.makedirs(experiment_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=experiment_dir)

    #save config
    config_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    model_save_dir = os.path.join(experiment_dir, 'models')
    os.makedirs(model_save_dir, exist_ok=True)
    plot_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    device = dev()

    log_path = os.path.join(experiment_dir, 'log')
    setup_logging(log_path)
    cache_dir = os.path.join(os.environ["CACHE_DIR"], args.job_id)
    if args.clear_cache and os.path.exists(cache_dir):
        clear_cache(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # Prepare training and validation files
    logger.info(config)
    logger.info("Fold number: " + str(config.fold_num))

    if config.use_504:
        files = find_matching_pairs()
        splits = create_kfold_splits(hsd_dataset_dir, matches=files, n_splits=config.num_splits, logger=logger)
    else:
        splits = utils.load_splits(splits_filepath)

    if config.detector_type == "old":
        detector_files = utils.load_old_detector_file(dataset_dir + '/old_detector.txt')
    elif config.detector_type == "new":
        detector_files = utils.load_new_detector_file(dataset_dir + '/new_detector.txt')
    else:
        detector_files = None

    train_files, val_files = prepare_train_val_files(splits, config.fold_num)

    data, _= utils.load_data(device, h5_dataset_dir, max_id=None, min_id=None, measurement_number_to_exclude=[],
                             files_to_include=train_files, detector_files_to_include=detector_files)
    data_valid, _ = utils.load_data(device, h5_dataset_dir, max_id=None, min_id=None, measurement_number_to_exclude=[],
                                    files_to_include=val_files, detector_files_to_include=detector_files)

    # Prepare dataset and dataloaders
    trn_dataset = utils.GenderSplitDataset(device, h5_dataset_dir, data, master_mask_filepath,
                                           transform=transforms.Compose([transforms.ToTensor()]), cache_dir=cache_dir,
                                           normalize=False, normalize_to_one=False, log=False, exp=False, poly=False,
                                           gauss_noise=config.gauss_noise, max_percentage=config.gauss_max_percentage,
                                           contrast=config.contrast, contrast_frac=config.contrast_frac,
                                           brightness=config.brightness, brightness_frac=config.brightness_frac)


    val_dataset = utils.GenderSplitDataset(device, h5_dataset_dir, data_valid, master_mask_filepath,
                                           transform=transforms.Compose([transforms.ToTensor()]), cache_dir=cache_dir,
                                           normalize=False, normalize_to_one=False, log=False, exp=False, poly=False,
                                           gauss_noise=False, contrast=False, brightness=False)

    logger.info(f"Number of training samples: {len(trn_dataset)}")
    logger.info(f"Number of validation samples: {len(val_dataset)}")

    if len(trn_dataset) == 0 or len(val_dataset) == 0:
        logger.error("No training or validation samples found. Exiting.")
        sys.exit(1)

    # Create samplers and dataloaders
    # Dataloader returning all and dataloader returning one measurement per ID as anchor
    sampler = utils.OneMeassurementPerIDSampler(trn_dataset)
    dataloader = DataLoader(trn_dataset, batch_size=config.batch_size, sampler=sampler)
    all_dataloader = DataLoader(trn_dataset, batch_size=config.batch_size, shuffle=False)

    sampler = utils.OneMeassurementPerIDSampler(val_dataset)
    all_val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=sampler)

    logger.info(f"Dataloaders created")

    if config.similarity_measure == 'COS':
        model = EncoderCosine(config.model.classname,
                              output_dim=config.model.params.out_dim,
                              input_channels=config.model.params.in_channels,
                              freeze_backbone=config.model.params.freeze_backbone,
                              dropouts=config.model.params.dropouts)
    if config.similarity_measure == 'L2' or config.similarity_measure == 'L1':
        model = EncoderDistance(config.model.classname,
                                output_dim=config.model.params.out_dim,
                                input_channels=config.model.params.in_channels,
                                freeze_backbone=config.model.params.freeze_backbone)

    logger.info(f"Model created")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = LeCunContrastiveLoss(margin=config.margin, METHOD=config.similarity_measure)

    training_losses = []
    validation_losses = []
    mapping_trn, mapping_val = create_mapping(trn_dataset, val_dataset)
    logger.info("Starting training")
    #compute embeddings for the training and validation datasets in the beginning of each epoch
    trn_bank  = compute_embeddings(model, all_dataloader, device,
                                   requires_grad=False,
                                   mapping=mapping_trn,
                                   embedding_dim=config.model.params.out_dim)

    logger.info("Computed embeddings for training dataset")
    pbar = trange(config.num_epochs)
    best_auc = 0
    for epoch in pbar:
        trn_loss, trn_bank = train_model(model, criterion, optimizer, trn_bank, dataloader,
                                          mapping=mapping_trn,
                                          pair_percentage=config.pair_percentage,
                                          similarity_measure=config.similarity_measure)
        training_losses.append(trn_loss)

        if epoch % 10 == 0 or epoch == config.num_epochs - 1:
            val_embedings = compute_embeddings(model, all_val_dataloader, device,
                                               requires_grad=False,
                                               mapping=mapping_val,
                                               embedding_dim=config.model.params.out_dim)
            val_loss, val_embedings = test_model(model, criterion, val_embedings, val_dataloader,
                                                  mapping=mapping_val,
                                                  pair_percentage=config.pair_percentage,
                                                  similarity_measure=config.similarity_measure)
            validation_losses.append(val_loss)

            auc_trn, auc_val, tpr_at_5_trn, tpr_at_5_val = compute_roc(plot_dir, epoch + 1,
                                                                       trn_bank, val_embedings,
                                                                       mapping_trn, mapping_val,
                                                                       similarity_measure=config.similarity_measure)

            logger.info(f"Epoch {epoch}, TRN: {trn_loss:.4f}, VAL: {val_loss:.4f}, AUC VAL: {auc_val:.1%}, TPR@5% VAL: {tpr_at_5_val:.1%}")
            pbar.set_description(f"Epoch {epoch}, TRN: {trn_loss:.4f}, VAL: {val_loss:.4f}, AUC VAL: {auc_val:.1%}, TPR@5% VAL: {tpr_at_5_val:.1%}")

            writer.add_scalar('loss/val', val_loss, epoch)
            writer.add_scalar('auc/val', auc_val, epoch)
            writer.add_scalar('tpr_at_5/val', tpr_at_5_val, epoch)
        else:
            logger.info(f"Epoch {epoch}, TRN: {trn_loss:.4f}")
            pbar.set_description(f"Epoch {epoch}, TRN: {trn_loss:.4f}")

        writer.add_scalar('loss/trn', trn_loss, epoch)
        writer.add_scalar('auc/trn', auc_trn, epoch)
        writer.add_scalar('tpr_at_5/trn', tpr_at_5_trn, epoch)

        if (epoch + 1) % config.save_after_epochs == 0 or epoch == 0 or epoch == config.num_epochs - 1:
            if auc_val > best_auc:
                best_auc = auc_val
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        model_save_dir,
                        f"fold_{config.fold_num}_best.pt"
                    )
                )
            # plot_loss(epoch, training_losses, validation_losses, plot_dir)

    #clear cache
    if args.clear_cache:
        clear_cache(cache_dir)


if __name__ == '__main__':
    main()
