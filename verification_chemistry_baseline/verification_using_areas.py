import matplotlib.pyplot as plt
import os
import numpy as np
import random
import logging
import argparse
import yaml

from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from scipy.stats import spearmanr, pearsonr, ranksums
import torch
import warnings
from match import find_matching_pairs
from tqdm import tqdm, trange


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(dir_path, system=None, files=None):
    separator = "\t"
    files_dict_m = {}
    files_dict_w = {}
    compound_list = []
    for file in files:
        compounds_dict = {}
        file_name = os.path.splitext(file)[0]
        system_number = int(file_name.split("_")[1])
        if system != None and system_number != system:
            continue
        import csv

        with open(dir_path + "/" + file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=separator)

            m_or_f = file_name.split("_")[3][0]
            id = file_name.split("_")[3][1:]

            for row in reader:
                compound = row['Name']
                if compound == 'Benzenoic acid, tetradecyl ester':
                    compound = 'Benzoic acid, tetradecyl ester'
                if compound == '1-Octanol, 2-hexyl-':
                    compound = '2-Hexyl-1-octanol'

                if compound not in compound_list:
                    compound_list.append(compound)

                t1 = float(row['1st Dimension Time (s)'])
                t2 = float(row['2nd Dimension Time (s)'])
                area = float(row['Area'])

                compounds_dict[compound] = (t1, t2, area)

        if m_or_f == 'M':
            files_dict_m[file_name] = (compounds_dict, m_or_f, id, system_number)
        else:
            files_dict_w[file_name] = (compounds_dict, m_or_f, id, system_number)
    # sort list by alphabet
    compound_list.sort()
    return files_dict_m, files_dict_w, compound_list


def make_vector_from_compounds(loaded_files, compound_list):
    files_compounds_vectors = {}
    for file in loaded_files:
        vector = np.zeros(len(compound_list))
        for compound in loaded_files[file][0]:
            index = compound_list.index(compound)
            vector[index] = loaded_files[file][0][compound][2]
        files_compounds_vectors[file] = (vector, loaded_files[file][1], loaded_files[file][2], loaded_files[file][3])
    return files_compounds_vectors


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("identity")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(log_dir, "log"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger


def normalize_vector_sum_one(loaded_files, compound_list):
    for file in loaded_files:
        vector = loaded_files[file][0]
        sum_ = np.sum(vector)
        new_vector = np.zeros(len(compound_list))
        for i in range(len(vector)):
            new_vector[i] = vector[i] / sum_
        loaded_files[file] = (new_vector, *loaded_files[file][1:])
    return loaded_files


def make_training_data_labels(loaded_files):
    data, labels = [], []
    for key, value in loaded_files.items():
        if value[1] == 'M':
            labels.append(value[2])
            data.append(value[0])
        else:
            labels.append(int(value[2]) + 20)
            data.append(value[0])
    return data, labels


def make_id_based_data(data, labels):
    data_result = {}
    for i in range(len(data)):
        if int(labels[i]) not in data_result:
            data_result[int(labels[i])] = []
        data_result[int(labels[i])].append(data[i])
    return data_result


def generate_pairs(data_dict):
    same_pairs, diff_pairs = [], []
    ids = list(data_dict.keys())
    for i, id1 in enumerate(ids):
        vectors = data_dict[id1]
        for j in range(len(vectors)):
            for k in range(j + 1, len(vectors)):
                same_pairs.append((vectors[j], vectors[k]))
        for id2 in ids[i + 1:]:
            for v1 in data_dict[id1]:
                for v2 in data_dict[id2]:
                    diff_pairs.append((v1, v2))
    return same_pairs, diff_pairs


def compute_distance(v1, v2, metric="euclidean"):
    if metric == "euclidean":
        return euclidean(v1, v2)
    elif metric == "pearson":
        return 1 - pearsonr(v1, v2)[0]
    elif metric == "spearman":
        return 1 - spearmanr(v1, v2)[0]
    elif metric == "cosine":
        return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    else:
        raise ValueError("Invalid distance metric")


def get_tpr_at_fpr(fpr, tpr, target_fpr=0.05):
    indices = np.where(fpr <= target_fpr)[0]
    if len(indices) == 0:
        return 0.0
    return tpr[indices[-1]]


def select_compounds(same_pairs, diff_pairs, threshold_pi=1.3):
    same_diffs = np.abs(same_pairs[:, 0] - same_pairs[:, 1])
    diff_diffs = np.abs(diff_pairs[:, 0] - diff_pairs[:, 1])
    p_values = [ranksums(same_diffs[:, c], diff_diffs[:, c], alternative='greater')[1] for c in
                range(same_diffs.shape[1])]
    selected = np.where(-np.log10(p_values) > threshold_pi)[0]
    return selected


def create_kfold_splits(root_dir, extensions=('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'pt', 'h5'), n_splits=10,
                        subsample=None):
    data = {'F': {}, 'M': {}}

    # Load all images and split into groups and subsets
    count_ = 0
    for filename in tqdm(os.listdir(root_dir)):
        if filename.endswith(extensions):
            count_ += 1
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
                    data[group][identifier].append(matches[filename])
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
    for split in splits:
        print(
            f"Number of train files: {sum(len(split[0][group][identifier]) for group in split[0] for identifier in split[0][group])}")
        print(
            f"Number of val files: {sum(len(split[1][group][identifier]) for group in split[1] for identifier in split[1][group])}")
    return splits


def save_splits_to_file(splits):
    with open('/mnt/personal/hlavsja3/experiments2025/splits/splits.txt', 'w') as f:
        for i, (tr, va) in enumerate(splits):
            print(f"Train: {len(tr['F'])} F, {len(tr['M'])} M; Val: {len(va['F'])} F, {len(va['M'])} M")
            f.write("Split " + str(i) + ", Train (Total IDs): " + str(len(tr['F'])) + " F, " + str(
                len(tr['M'])) + " M; Val (Total IDs): " + str(len(va['F'])) + " F, " + str(len(va['M'])) + " M\n")
            f.write("Train files:\n")
            for id in tr['F']:
                for file in tr['F'][id]:
                    file = file.replace(".h5", "")
                    f.write(file + "\n")
            for id in tr['M']:
                for file in tr['M'][id]:
                    file = file.replace(".h5", "")
                    f.write(file + "\n")
            f.write("Val files:\n")
            for id in va['F']:
                for file in va['F'][id]:
                    file = file.replace(".h5", "")
                    f.write(file + "\n")
            for id in va['M']:
                for file in va['M'][id]:
                    file = file.replace(".h5", "")
                    f.write(file + "\n")
            f.write("\n")


def from_split_get_files(f_list):
    result = []
    for gender in f_list:
        for id in f_list[gender]:
            result.extend(f_list[gender][id])
    return result


def main():
    set_all_seeds(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=True)
    parser.add_argument('--job_id', type=str, required=True)
    args = parser.parse_args()


    dataset_dir = os.environ["DATASET_DIR"]
    experiment_dir = os.path.join(os.environ['EXPERIMENTS_DIR'], args.job_id)
    vis_dir = os.path.join(os.environ['EXPERIMENTS_DIR'], args.job_id, 'visualizations')
    hsd_dataset_dir = dataset_dir + "/HSD/"

    splits = create_kfold_splits(hsd_dataset_dir)

    config = yaml.load(open(args.conf_path, "r"), Loader=yaml.FullLoader)
    out_dir = os.path.join(experiment_dir, args.job_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'confs.yaml'), 'w') as f:
        yaml.dump(config, f)

    logger = setup_logger(out_dir)

    use_compound_selection = config['use_compound_selection']
    threshold_pi = config['threshold_pi']
    metric = config['metric']

    plot_dir = out_dir + '/plots'
    os.makedirs(plot_dir, exist_ok=True)
    AUCS_trn = []
    TPRS_5_fpr_trn = []
    AUCS_val = []
    TPRS_5_fpr_val = []
    logger.info(f"Using metric: {metric}, compound selection: {use_compound_selection}, threshold pi: {threshold_pi}")
    for i, (train_files, val_files) in enumerate(splits):
        train_files = from_split_get_files(train_files)
        val_files = from_split_get_files(val_files)
        loaded_files_m, loaded_files_w, compound_list = load_data(vis_dir, files=train_files)
        loaded_files_m = normalize_vector_sum_one(make_vector_from_compounds(loaded_files_m, compound_list),
                                                  compound_list)
        loaded_files_w = normalize_vector_sum_one(make_vector_from_compounds(loaded_files_w, compound_list),
                                                  compound_list)
        data, labels = make_training_data_labels({**loaded_files_m, **loaded_files_w})
        train_data = make_id_based_data(data, labels)
        # Validation data
        loaded_files_m_val, loaded_files_w_val, _ = load_data(vis_dir, files=val_files)
        loaded_files_m_val = normalize_vector_sum_one(make_vector_from_compounds(loaded_files_m_val, compound_list),
                                                      compound_list)
        loaded_files_w_val = normalize_vector_sum_one(make_vector_from_compounds(loaded_files_w_val, compound_list),
                                                      compound_list)
        val_data, val_labels = make_training_data_labels({**loaded_files_m_val, **loaded_files_w_val})
        val_data = make_id_based_data(val_data, val_labels)

        same_pairs_train, diff_pairs_train = generate_pairs(train_data)
        same_pairs_val, diff_pairs_val = generate_pairs(val_data)

        if use_compound_selection:
            selected = select_compounds(np.array(same_pairs_train), np.array(diff_pairs_train), threshold_pi)
            same_pairs_train = np.array(same_pairs_train)[:, :, selected]
            diff_pairs_train = np.array(diff_pairs_train)[:, :, selected]
            same_pairs_val = np.array(same_pairs_val)[:, :, selected]
            diff_pairs_val = np.array(diff_pairs_val)[:, :, selected]

        distances_H0 = [compute_distance(p1, p2, metric) for p1, p2 in same_pairs_train]
        distances_H1 = [compute_distance(p1, p2, metric) for p1, p2 in diff_pairs_train]
        # Remove NaNs because eg all 0 distances will result in NaN
        distances_H0 = np.array(distances_H0)
        distances_H0 = distances_H0[~np.isnan(distances_H0)]

        distances_H1 = np.array(distances_H1)
        distances_H1 = distances_H1[~np.isnan(distances_H1)]

        print(f"Split {i}: Train same pairs: {len(distances_H0)}, Train diff pairs: {len(distances_H1)}")

        gmm_H0 = GaussianMixture(n_components=2, random_state=42).fit(np.array(distances_H0).reshape(-1, 1))
        gmm_H1 = GaussianMixture(n_components=2, random_state=42).fit(np.array(distances_H1).reshape(-1, 1))

        x = np.linspace(0, 1, 1000)
        plt.plot(x, np.exp(gmm_H0.score_samples(x.reshape(-1, 1))), label='H0')
        plt.plot(x, np.exp(gmm_H1.score_samples(x.reshape(-1, 1))), label='H1')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'gmm_{metric}_pi{threshold_pi}_split{i}.pdf'))
        plt.close()

        val_dist = [(compute_distance(p1, p2, metric), 1) for p1, p2 in same_pairs_val] + [
            (compute_distance(p1, p2, metric), 0) for p1, p2 in diff_pairs_val]
        train_dist = [(compute_distance(p1, p2, metric), 1) for p1, p2 in same_pairs_train] + [
            (compute_distance(p1, p2, metric), 0) for p1, p2 in diff_pairs_train]

        val_scores = [np.exp(gmm_H0.score_samples(np.array(d).reshape(1, -1))) / np.exp(
            gmm_H1.score_samples(np.array(d).reshape(1, -1))) for d, _ in val_dist]
        train_scores = [np.exp(gmm_H0.score_samples(np.array(d).reshape(1, -1))) / np.exp(
            gmm_H1.score_samples(np.array(d).reshape(1, -1))) for d, _ in train_dist]

        val_labels = [l for _, l in val_dist]
        train_labels = [l for _, l in train_dist]
        fpr_train, tpr_train, _ = roc_curve(train_labels, train_scores)
        fpr_val, tpr_val, _ = roc_curve(val_labels, val_scores)
        auc_train = auc(fpr_train, tpr_train)
        auc_val = auc(fpr_val, tpr_val)
        tpr_at_5_fpr_train = get_tpr_at_fpr(fpr_train, tpr_train, target_fpr=0.05)
        tpr_at_5_fpr_val = get_tpr_at_fpr(fpr_val, tpr_val, target_fpr=0.05)
        logger.info(f"Split {i}: Train AUC={auc_train:.3f}, TPR@5%FPR={tpr_at_5_fpr_train:.3f}, "
                    f"Val AUC={auc_val:.3f}, TPR@5%FPR={tpr_at_5_fpr_val:.3f}")
        AUCS_trn.append(auc_train)
        TPRS_5_fpr_trn.append(tpr_at_5_fpr_train)
        AUCS_val.append(auc_val)
        TPRS_5_fpr_val.append(tpr_at_5_fpr_val)

        plt.figure()
        plt.plot(fpr_train, tpr_train, label=f"Train AUC={auc_train:.3f}")
        plt.plot(fpr_val, tpr_val, label=f"Val AUC={auc_val:.3f}", linestyle="--")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'roc_curve_{metric}_pi{threshold_pi}_split{i}.pdf'))
        plt.close()
    logger.info("Training and validation completed.")
    avg_trn_auc = np.mean(AUCS_trn)
    avg_trn_tpr_5_fpr = np.mean(TPRS_5_fpr_trn)
    avg_val_auc = np.mean(AUCS_val)
    avg_val_tpr_5_fpr = np.mean(TPRS_5_fpr_val)
    logger.info(f"Average Train AUC: {avg_trn_auc:.3f}, TPR@5%FPR: {avg_trn_tpr_5_fpr:.3f}")
    logger.info(f"Average Val AUC: {avg_val_auc:.3f}, TPR@5%FPR: {avg_val_tpr_5_fpr:.3f}")


if __name__ == '__main__':
    main()
