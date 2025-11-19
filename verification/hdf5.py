from matplotlib import pyplot as plt

import os
import h5py
from time import sleep
import torch as th
import numpy as np
from tqdm import tqdm, trange

def load_hdf5(tensor_name, hdf5_dir, file_name):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    hdf5_filepath = os.path.join(hdf5_dir, file_name)
    if "race" in hdf5_filepath.lower():
        return none
    group = tensor_name[0]

    if group in ['F', 'M']:
        identifier = ''
        for char in tensor_name[1:]:
            if char.isdigit():
                identifier += char
            else:
                break
        success = False
        while not success:
            try:
                with h5py.File(hdf5_filepath, 'r') as f:
                    # Check if group and identifier exist
                    if group in f and identifier in f[group] and tensor_name in f[group][identifier]:
                        dset = f[group][identifier][tensor_name]
                        tensor_data = dset[:]
                        # Convert numpy array to torch tensor
                        tensor = th.tensor(tensor_data, device=device)
                        success = True
                        return tensor
                    else:
                        raise KeyError(f"Dataset {tensor_name} not found in file.")
            except Exception as e:
                print(f"HDF5 Error while loading {tensor_name}: {e}")
                sleep(1)
    return None