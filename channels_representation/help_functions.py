import os
from time import sleep

from omegaconf import OmegaConf
import torch
import numpy as np
import matplotlib
import cv2
from einops import rearrange
import h5py

def from_times_pixels(compounds: dict, system_number, x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds,
                      x_ten_seconds, y_ten_seconds, t1_shift=0, t1_offset=0, t2_offset=0):
    compounds_pixels = {}
    if system_number == 1:
        min_ten_seconds = x_six_seconds * 6 + x_eight_seconds * 6
        for compound in compounds:
            t1 = compounds[compound]['t1'] + t1_offset + t1_shift
            t2 = compounds[compound]['t2'] + t2_offset
            t2 *= 1000
            t2_pixels = t2 // 5
            if t1 < min_ten_seconds:
                t1_pixels = t1 // 6
            else:
                t1 = t1 - min_ten_seconds
                t1_pixels = (x_six_seconds + x_eight_seconds) + t1 // 10
            t2_pixels = int(t2_pixels)
            t1_pixels = int(t1_pixels)
            compounds_pixels[compound] = (t1_pixels, t2_pixels)
    elif system_number == -1:
        min_ten_seconds = x_eight_seconds * 8
        for compound in compounds:
            t1 = compounds[compound]['t1'] + t1_offset + t1_shift
            t2 = compounds[compound]['t2'] + t2_offset
            t2 *= 1000
            t2_pixels = t2 // 5
            if t1 < min_ten_seconds:
                t1_pixels = t1 // 8
            else:
                t1 = t1 - min_ten_seconds
                t1_pixels = x_eight_seconds + t1 // 10
            t2_pixels = int(t2_pixels)
            t1_pixels = int(t1_pixels)
            compounds_pixels[compound] = (t1_pixels, t2_pixels)
    else:
        min_ten_seconds = x_six_seconds * 6 + x_eight_seconds * 8
        min_eight_seconds = x_six_seconds * 6
        for compound in compounds:
            t1 = compounds[compound]['t1'] + t1_offset + t1_shift
            t2 = compounds[compound]['t2'] + t2_offset
            t2 *= 1000
            t2_pixels = t2 // 5
            if t1 < min_eight_seconds:
                t1_pixels = t1 // 6
            elif t1 < min_ten_seconds:
                t1 = t1 - min_eight_seconds
                t1_pixels = x_six_seconds + t1 // 8
            else:
                t1 = t1 - min_ten_seconds
                t1_pixels = (x_six_seconds + x_eight_seconds) + t1 // 10
            t2_pixels = int(t2_pixels)
            t1_pixels = int(t1_pixels)
            compounds_pixels[compound] = (t1_pixels, t2_pixels)
    return compounds_pixels


import os

def load_compound_and_times(filepath: str) -> dict:
    """
    Reads a tab-separated file with columns:
      Name, 1st Dimension Time (s), 2nd Dimension Time (s), …
    Returns a dict mapping compound names to {'t1': int, 't2': float}.
    """
    compounds_times = {}

    # mappings for name corrections
    corrections = {
        '1-Octanol, 2-hexyl-': '2-Hexyl-1-octanol',
        'Benzenoic acid, tetradecyl ester': 'Benzoic acid, tetradecyl ester',
        "5,10-Diethoxy-2,3,7,8-tetrahydro-1H,6H-dipyrrolo[1,2-a:1',2'-d]pyrazine":
            "5,10-Diethoxy-2,3,7,8-tetrahydro-1H,6H-dipyrrolo[1,2-a_1',2'-d]pyrazine"
    }

    with open(filepath, 'r', encoding='utf-8') as f:
        # read header
        _ = f.readline().rstrip('\n').split('\t')
        # identify the indexes of the columns we care about
        idx_name, idx_t1, idx_t2 = 0, 1, 2

        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= idx_t2:
                # skip malformed lines
                continue

            name = parts[idx_name]
            # apply corrections if needed
            if name in corrections:
                name = corrections[name]

            # parse times
            try:
                t1 = int(parts[idx_t1])
            except ValueError:
                # skip or default if non-integer
                continue

            # convert comma to dot for float parsing
            t2_str = parts[idx_t2].replace(',', '.')
            try:
                t2 = float(t2_str)
            except ValueError:
                # skip or default if non-float
                continue

            compounds_times[name] = {'t1': t1, 't2': t2}

    return compounds_times


def save_hdf5(tensor, tensor_name, hdf5_dir, file_name, logger=None,
              compression="gzip",
              compression_opts=4,
              ):
    os.makedirs(hdf5_dir, exist_ok=True)

    hdf5_filepath = os.path.join(hdf5_dir, file_name)
    with h5py.File(hdf5_filepath, 'w') as _:
        pass

    group = tensor_name[0]

    if group in ['F', 'M']:
        identifier = ''
        for char in tensor_name[1:]:
            if char.isdigit():
                identifier += char
            else:
                break

        if tensor.is_cuda:
            tensor = tensor.cpu().numpy()
        else:
            tensor = tensor.numpy()

        chunk_size = (tensor.shape[0], 10, 10)

        success = False
        while not success:
            try:
                with h5py.File(hdf5_filepath, 'a') as f:
                    # Check if group exists, otherwise create it
                    if group not in f:
                        f.create_group(group)
                    if identifier not in f[group]:
                        f[group].create_group(identifier)
                    # Check if dataset exists, otherwise create it
                    if tensor_name not in f[group][identifier]:
                        dset = f[group][identifier].create_dataset(
                            name=tensor_name,
                            data=tensor,
                            chunks=chunk_size,
                            compression=compression,
                            compression_opts=compression_opts,
                            shuffle=True,
                            fletcher32=True
                        )
                success = True
            except Exception as e:
                print(f"HDF5 Error while processing {tensor_name}: {e}")
                sleep(1)
    if logger is not None:
        logger.debug(f"Dataset {tensor_name} saved to {hdf5_filepath}")
