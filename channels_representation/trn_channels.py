from __future__ import annotations

import importlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import List, Tuple
import os
import time
import argparse
import logging

import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from tqdm import trange
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from multiprocessing import set_start_method
import netCDF4 as nc
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence

from data import prepare_one_data
from utils import load_splits, custom_centers, custom_centered_centers, get_obj_from_str
from loss import TargetWeights
from models import *


logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Training job configuration")
    parser.add_argument('conf_path', type=str, help='Path to the configuration file')
    parser.add_argument('job_id', type=str, help='Job ID')
    args = parser.parse_args()

    return args


# ----------------------------------------------------------------------------- #
#                              0.  GLOBAL CONFIG                                #
# ----------------------------------------------------------------------------- #

torch.set_default_dtype(torch.float32)

# ----------------------------------------------------------------------------- #
#                           1.  DATA PREPARATION                                #
# ----------------------------------------------------------------------------- #


@torch.no_grad()                       # no autograd work here
def pad_batch(
    samples: List[Tuple[np.ndarray, np.ndarray]],
    device: torch.device | None = None,
    pin_memory: bool = False,
    non_blocking: bool = True,
    dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorised padding of ragged spectra.

    Returns
    -------
    pos   : float32 [B, N_max]
    inten : float32 [B, N_max]
    mask  : bool    [B, N_max]   (True for valid entries)
    """
    device = dev() if device is None else device
    B = len(samples)
    lengths = [p.shape[0] for p, _ in samples]
    Nmax = max(lengths)

    start = time.time()
    if dist.get_rank() == 0:
        logger.info(f"Padding {B} samples with max length {Nmax} on device {device}")

    # Allocate the **only** host copy, already pinned.
    pos_gpu   = torch.zeros((B, Nmax), device=dev(), dtype=dtype)
    inten_gpu = torch.zeros_like(pos_gpu)

    # Fill row‑wise without creating per‑sample tensors
    for i, (p, v) in enumerate(samples):
        if i % (len(samples) // 100) == 0 and dist.get_rank() == 0:
            logger.info(f"Filling sample {i + 1}/{B} with length {len(p)}")
        L = lengths[i]
        pos_gpu[i, :L].copy_(torch.as_tensor(p, dtype=dtype, device=dev()))
        inten_gpu[i, :L].copy_(torch.as_tensor(v, dtype=dtype, device=dev()))

    if dist.get_rank() == 0:
        logger.info(f"Padding took {time.time() - start:.2f} seconds")

    # Mask can be produced directly on the GPU
    mask_gpu = (
        torch.arange(Nmax, device=device)
        .expand(B, -1)
        .lt(torch.tensor(lengths, device=device).unsqueeze(1))
    )
    return pos_gpu, inten_gpu, mask_gpu


# ----------------------------------------------------------------------------- #
#                          3.  TRAINING / OPTIMISATION                         #
# ----------------------------------------------------------------------------- #


def optimise(
    cfg,
) -> Tuple[GaussianBank, List[float]]:
    """
    Fits GaussianBank so that the *sum* of kernels reconstructs the input spectra.

    Loss = mean‑squared‑error over valid (mask==True) entries.
    """

    # get global min and max for all processes
    local_min = pos.min()
    local_max = pos.max()
    local_max_inten = inten.max()

    # allocate glob min and max tensors
    global_min = local_min.clone()
    global_max = local_max.clone()
    global_max_inten = local_max_inten.clone()

    # reduce to find global min and max
    dist.all_reduce(global_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(global_max_inten, op=dist.ReduceOp.MAX)

    if dist.get_rank() == 0:
        logger.info(f"Global min: {global_min.item()}, Global max: {global_max.item()}, Global max inten: {global_max_inten.item()}")

    if cfg.centers_mode == 'limits':
        if cfg.model.params.K <= 3:
            centers = torch.linspace(global_min, global_max, cfg.model.params.K, device=dev())
        else:
            centers = custom_centers(K=cfg.model.params.K, max_val=global_max).to(dev())
    elif cfg.centers_mode == 'centers':
        centers = custom_centered_centers(K=cfg.model.params.K, max_val=global_max).to(dev())
    else:
        raise ValueError(f"Unknown centers_mode: {cfg.centers_mode}")

    model = get_obj_from_str(config.model.classname)(
        centers, **config.model.params
    )

    model = model.to(dev())

    if dist.get_rank() == 0:
        model.plot(os.path.join(split_experiment_dir, "initial_kernels.png"))

    model = DDP(model,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                output_device=int(os.environ['LOCAL_RANK']),
                broadcast_buffers=True,
                bucket_cap_mb=128,
                find_unused_parameters=False,
                )

    opt   = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    reweighting = TargetWeights(global_max_value=global_max_inten.item(), percentage=0.01)

    # ------------------------------------------------------------------------ #
    def step(batch_idx: torch.Tensor,print_=False) -> Tuple[torch.tensor, torch.tensor]:
        """One gradient step on a mini‑batch (in dim 0)."""
        opt.zero_grad()
        p   = pos[batch_idx]                 # [b,N]
        tgt = inten[batch_idx]
        m   = mask[batch_idx]
        is_trn = trn_mask_vec[batch_idx]
        is_val = 1.0 - is_trn

        with torch.autocast("cuda", dtype=torch.float16, enabled=False):
            weights = model(p)                   # [K,b,N]
            recon   = (weights * tgt.unsqueeze(0)).sum(dim=0)   # [b,N]
            sqerr = (recon - tgt).pow(2)            # [b,N]
            sqerr_valid = sqerr * m
            sqerr_nonweighted_valid = sqerr_valid.clone()
            if 'loss' in cfg and cfg.loss == "weighted_mse":
                # apply reweighting
                tgt_t = reweighting(tgt)
                sqerr_valid = sqerr_valid * tgt_t  # [b,N]
            step_num_trn = (is_trn * m).sum()
            step_num_val = (is_val * m).sum()
            zero_loss = 0.0 * sqerr_valid.sum()
            step_loss_trn = (is_trn * sqerr_valid).sum() / step_num_trn if step_num_trn.item() > 0 else zero_loss  # [b,N] -> scalar
            step_loss_val = (is_val * sqerr_valid).sum() / step_num_val if step_num_val.item() > 0 else zero_loss  # [b,N] -> scalar
            step_loss_val_nonweighted = (is_val * sqerr_nonweighted_valid).sum() / step_num_val if step_num_val.item() > 0 else zero_loss  # [b,N] -> scalar
            if print_:
                logger.info(f"recon: {recon[100:150]}, tgt: {tgt[100:150]}")

        step_loss_trn.backward()
        opt.step()

        return step_loss_trn, step_loss_val, step_loss_val_nonweighted

    # --- training loop ------------------------------------------------------- #
    pbar = trange(1, cfg.iterations + 1)
    for iteration in pbar:
        batch = balanced_batch(batch_size)  # random mini‑batch
        trn_loss, val_loss, val_loss_unweighted = step(batch)

        dist.all_reduce(trn_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_unweighted, op=dist.ReduceOp.SUM)

        trn_loss /= trn_samples_num
        val_loss /= val_samples_num
        val_loss_unweighted /= val_samples_num

        if dist.get_rank() == 0:
            pbar.set_description(f"[{split_idx + 1:02d}][{iteration:04d}] TRN: {trn_loss.item():.4e}, VAL: {val_loss_unweighted.item():.4e}, VAL_W: {val_loss.item():.4e}")
            writer.add_scalar('loss/trn', trn_loss.item(), iteration)
            writer.add_scalar('loss/val', val_loss_unweighted.item(), iteration)
            writer.add_scalar('loss/val_weighted', val_loss.item(), iteration)
            if iteration % cfg.log_frequency == 0 or iteration == cfg.iterations:
                logger.info(f"[{split_idx + 1:02d}][{iteration:4d}/{cfg.iterations}]  TRN: {trn_loss.item():.4e}, VAL: {val_loss_unweighted.item():.4e}, VAL_W: {val_loss.item():.4e}")
            if iteration % cfg.save_frequency == 0 or iteration == cfg.iterations:
                # save model
                model_dir = os.path.join(split_experiment_dir, "models")
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"model_{iteration:06d}.pt")
                torch.save(model.state_dict(), model_path)
            if iteration % cfg.vis_frequency == 0 or iteration == cfg.iterations:
                # plot
                plot_dir = os.path.join(split_experiment_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                plot_filepath = os.path.join(plot_dir, f"kernels_{iteration:06d}.png")
                model.module.plot(plot_filepath, points=1000)


    return model


# ----------------------------------------------------------------------------- #
#                                 5.                                           #
# ----------------------------------------------------------------------------- #


def dev(device=None):
    """
    Get the device to use for torch.distributed.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device(int(os.environ['LOCAL_RANK']))
        return torch.device("cpu")
    return torch.device(device)


def setup_dist(seed=0):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    set_start_method('forkserver', force=True)

    global_rank = 0
    if torch.cuda.is_available():
        if 'DISABLE_DDP' not in os.environ:
            dist.init_process_group(backend='nccl',
                                    init_method='env://',
                                    timeout=timedelta(seconds=1800))

            global_rank = dist.get_rank()
        torch.cuda.set_device(dev())

    torch.manual_seed(seed + global_rank)
    random.seed(seed + global_rank)
    np.random.seed(seed + global_rank)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + global_rank)
        torch.cuda.manual_seed_all(seed + global_rank)

    return global_rank


def balanced_batch(batch_size: int) -> torch.Tensor:
    F = len(file_bins)
    assert batch_size % F == 0, "batch_size must be divisible by #files on this rank"
    k = batch_size // F           # samples per file

    # Draw k indices *with replacement* from every file bucket
    parts = [
        file_bins[f][torch.randint(0, file_bins[f].numel(), (k,), device=dev())]
        for f in range(F)
    ]
    batch = torch.cat(parts)                     # [batch_size]
    return batch[torch.randperm(batch_size)]     # shuffle the inter‑file order


if __name__ == "__main__":
    # total number of processes
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = 1

    args = parse_args()

    config = OmegaConf.load(args.conf_path)
    num_files_per_rank = config.num_files_per_rank
    batch_size = max(config.batch_size // world_size, num_files_per_rank)
    if batch_size % num_files_per_rank != 0:
        raise ValueError(f"Batch size {batch_size} must be divisible by num_files_per_rank {num_files_per_rank}")

    dataset_dir = os.environ["DATASET_DIR"]
    channel_experiments_dir = dataset_dir + "/channels/"
    hsd_dataset_dir = dataset_dir + "/HSD/"
    reference_positions_filepath = channel_experiments_dir + '/system_1_target_F4_05.txt'
    reference_tensor_filepath = channel_experiments_dir + '/F4_05_system1.pt'
    reference_cdf_filepath = hsd_dataset_dir + '/F4_05_system1.cdf'
    splits_filepath = dataset_dir + '/splits.txt'

    experiment_dir = os.path.join(os.environ["EXPERIMENTS_DIR"], args.job_id)

    # setup distributed environment
    setup_dist(42)

    trn_job_conf_path = os.path.join(experiment_dir, "config.yaml")

    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('[rank %(rank)d]%(asctime)s,%(levelname)s:%(message)s')

    class DDPRankFilter(logging.Filter):
        def filter(self, record):
            record.rank = dist.get_rank() if dist.is_initialized() else 0
            return True

    # Add a stream handler (console output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.addFilter(DDPRankFilter())
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # add file handler
    if dist.get_rank() == 0:
        os.makedirs(experiment_dir, exist_ok=True)

        log_file = os.path.join(experiment_dir, 'log')
        fh = logging.FileHandler(log_file)  # file handler
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        with open(trn_job_conf_path, 'w') as file:
            OmegaConf.save(config, file)

    dist.barrier()  # wait for all processes to finish setup

    splits = load_splits(splits_filepath)
    s = 0
    all_files_sorted_as_in_splits = [f for f in splits[s]['trn'] + splits[s]['val']]
    if config.is_debug:
        all_files_sorted_as_in_splits = [f for f in splits[s]['trn'][:1] + splits[s]['val'][:4] + splits[s]['trn'][1:]]

    filestems_this_rank = all_files_sorted_as_in_splits[num_files_per_rank * dist.get_rank():num_files_per_rank * (dist.get_rank() + 1)]

    logger.info(f"Device: {dev()}, Local Rank: {os.environ.get('LOCAL_RANK', 'N/A')}, Global Rank: {dist.get_rank()}, processing {len(filestems_this_rank)} files: {filestems_this_rank}")

    positions_all, values_all, is_trn_all, file_id_all = [], [], [], []
    for file_idx, filestem in enumerate(filestems_this_rank):
        filename = filestem + '.cdf'
        pos_list, val_list = prepare_one_data(os.path.join(hsd_dataset_dir, filename), logger=logger)

        positions_all.extend(pos_list)
        values_all.extend(val_list)

        file_id_all.extend([file_idx] * len(pos_list))
        dist.barrier()  # ensure all processes finish loading the file before proceeding

    dataset = list(zip(positions_all, values_all))
    file_id_vec = torch.tensor(file_id_all, device=dev())

    file_bins = [
        (file_id_vec == f).nonzero(as_tuple=False).squeeze(1)
        for f in range(len(filestems_this_rank))
    ]

    #all cdf files
    logger.info(f"Dataset size: {len(dataset)}")

    logger.info(f"Padding dataset of size {len(dataset)}...")
    pos, inten, mask = pad_batch(dataset)  # [B,N]
    del dataset, positions_all, values_all

    # --- train -------------------------------------------------------------- #
    for split_idx, split in enumerate(splits):
        for file_idx, filestem in enumerate(filestems_this_rank):
            split_flag = 1.0 if filestem in splits[0]['trn'] else 0.0  # 1 → trn, 0 → val
            is_trn_all.extend([split_flag] * (file_id_vec == file_idx).sum().item())
        trn_mask_vec = torch.tensor(is_trn_all, dtype=torch.float32, device=dev()).unsqueeze(1)  # [B, 1]

        trn_samples_num = trn_mask_vec.sum()
        val_samples_num = (1 - trn_mask_vec).sum()

        dist.all_reduce(trn_samples_num, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_samples_num, op=dist.ReduceOp.SUM)

        if dist.get_rank() == 0:
            logger.info(f"Training on split {split_idx + 1}/{len(splits)}: {filestems_this_rank}, "
                        f"trn_mask_vec ones: {trn_mask_vec.sum().item()}, "
                        f"trn_mask_vec zeros: {(1 - trn_mask_vec).sum().item()}, "
                        f"trn_samples_num: {trn_samples_num.item()}, val_samples_num: {val_samples_num.item()}")
            split_experiment_dir = os.path.join(experiment_dir, f'split_{split_idx + 1:02d}')

            os.makedirs(split_experiment_dir, exist_ok=True)

            writer = SummaryWriter(log_dir=split_experiment_dir)

        optimise(config)

    dist.destroy_process_group()