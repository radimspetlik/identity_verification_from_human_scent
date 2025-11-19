import glob
import sys
import time
import warnings
import os
import logging
from argparse import ArgumentParser

import matplotlib
import torch
from omegaconf import OmegaConf
from einops import rearrange
from scipy.interpolate import LinearNDInterpolator
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm, trange
import netCDF4 as nc

from utils import get_obj_from_str, custom_centers, custom_centered_centers, extract_global_min_max
from data import set_system


from help_functions import from_times_pixels, load_compound_and_times, save_hdf5


logging.basicConfig(level=logging.INFO)


class Settings:
    def __init__(self, box_add_x, box_add_y, metric, weighted, weighting_function, only_significant=False):
        self.box_add_x = box_add_x
        self.box_add_y = box_add_y
        self.metric = metric
        self.weighted = weighted
        self.weighting_function = weighting_function
        self.only_significant = only_significant

    def __str__(self):
        import inspect
        if self.weighting_function is not None:
            if inspect.isfunction(self.weighting_function) and self.weighting_function.__name__ == "<lambda>":
                lambda_source = inspect.getsource(self.weighting_function).strip()
                weighting_function_str = lambda_source.split(':', 1)[1].strip()
                weighting_function_str = weighting_function_str.split(',')[0]
            else:
                weighting_function_str = str(self.weighting_function)
        else:
            weighting_function_str = "None"
        return f"box size x: {2 * self.box_add_x + 1}, box size y: {2 * self.box_add_y + 1}, metric: {self.metric}, weighted: {self.weighted}, weighting function: {weighting_function_str}, only significant: {self.only_significant}"


def apply_color_map(grayscale_image):
    cm = matplotlib.colormaps['viridis']
    return cm(grayscale_image)[..., :3] * 255


def warp_tensor(reference_tensor, tensor_to_warp, reference_compound_pixels, compounds_pixels, name,
                save_dir=None, save_grids=False, suffix=None, viz=False):
    reference_tensor = reference_tensor.cpu().numpy()
    tensor_to_warp = tensor_to_warp.cpu().numpy()
    C, H, W = tensor_to_warp.shape
    t1_ref, t2_ref = [], []
    t1, t2 = [], []
    t1_ref_shift, t2_ref_shift = [], []
    compound_order = []
    for compound in compounds_pixels:
        if compound not in reference_compound_pixels:
            continue
        compound_order.append(compound)
        reference_t1 = reference_compound_pixels[compound][0]
        reference_t2 = reference_compound_pixels[compound][1]
        t1_ref.append(reference_t1)
        t2_ref.append(reference_t2)
        measure_t1 = compounds_pixels[compound][0]
        measure_t2 = compounds_pixels[compound][1]
        t1.append(measure_t1)
        t2.append(measure_t2)
        t1_shift = reference_t1 - measure_t1
        t2_shift = reference_t2 - measure_t2
        t1_ref_shift.append(t1_shift)
        t2_ref_shift.append(t2_shift)

    def preproces_tensor_to_tic_2(tensor):
        print("1. tensor min max", tensor.min(), tensor.max())
        tensor = tensor.sum(0, keepdims=True).transpose((1, 2, 0))
        # print('Check if tensor is the same', tensor_to_warp is tensor)
        tensor = tensor - tensor[tensor > 0].min()
        tensor[tensor < 0] = 0
        tensor = np.log(tensor + 1e-10)
        print("2. tensor min max", tensor.min(), tensor.max())
        min_greater_zero = tensor[tensor > 0].min()
        tensor[tensor < 0] = min_greater_zero
        tensor = tensor / tensor.max()
        tensor = apply_color_map(tensor)
        tensor = np.squeeze(tensor)
        return tensor

    def preproces_tensor_to_tic(tensor):
        """
        Args:
          tensor: <C×H×W> numpy array  (e.g. spectral channels × height × width)
        Returns:
          H×W×3 uint8 RGB image after log-scaling and colormap
        """
        # 1) Sum across channels → 1×H×W, then move to H×W×1
        tic = tensor.sum(axis=0, keepdims=True).transpose(1, 2, 0)

        # 2) Shift so the minimum is exactly 0
        gmin = tic.min()
        tic = tic - gmin

        # 3) Log-scale (add a tiny epsilon to avoid log(0))
        tic = np.log(tic + 1e-10)

        # 4) Shift again so the logged data starts at 0
        log_min = tic.min()
        tic = tic - log_min

        # 5) Normalize to [0,1]
        log_max = tic.max()
        tic = tic / log_max

        # 6) Apply your colormap and squeeze out the singleton channel
        tic = apply_color_map(tic)  # expects shape H×W×1 → returns H×W×3
        tic = np.squeeze(tic)  # final shape H×W×3

        return tic

    tic_to_warp = preproces_tensor_to_tic(tensor_to_warp)
    for i in range(len(t1)):
        cv2.circle(tic_to_warp, (int(t1[i]), int(t2[i])), radius=4, color=(255, 0, 0), thickness=1)

    reference_tic = preproces_tensor_to_tic(reference_tensor)
    c = 0
    for i in range(len(t1_ref)):
        # print("Reference", t1_ref[i], t2_ref[i])
        cv2.circle(reference_tic, (int(t1_ref[i]), int(t2_ref[i])), radius=4, color=(0, 0, 255), thickness=1)
        c += 1
    # print("Number of compounds", c)

    t1_ref.extend([0, 0, W - 1, W - 1])
    t2_ref.extend([0, H - 1, H - 1, 0])
    t1.extend([0, 0, W - 1, W - 1])
    t2.extend([0, H - 1, H - 1, 0])

    t1_ref, t2_ref = np.array(t1_ref), np.array(t2_ref)
    t1, t2 = np.array(t1), np.array(t2)
    x_coords, y_coords = np.meshgrid(np.linspace(0, W, W), np.linspace(0, H, H))
    x_coords_flat = x_coords.flatten()
    y_coords_flat = y_coords.flatten()

    reference_points = np.vstack((t1_ref, t2_ref)).T  # Original grid points
    new_positions = np.vstack((t1, t2)).T  # Modified grid points

    # Interpolate new x and y coordinates for the whole grid
    interp_x = LinearNDInterpolator(reference_points, new_positions[:, 0])
    interp_y = LinearNDInterpolator(reference_points, new_positions[:, 1])

    # Interpolated coordinates for the entire grid
    new_x_coords_flat = interp_x(x_coords_flat, y_coords_flat)
    new_y_coords_flat = interp_y(x_coords_flat, y_coords_flat)

    # Create new grid with normalized coordinates for grid_sample
    new_grid = np.stack([new_x_coords_flat, new_y_coords_flat], axis=1)
    new_grid = new_grid.reshape(H, W, 2)
    new_grid = torch.tensor(new_grid, dtype=torch.float32).unsqueeze(0)

    # Normalize grid to [-1, 1] range
    new_grid[..., 0] = (new_grid[..., 0] / (W - 1)) * 2 - 1
    new_grid[..., 1] = (new_grid[..., 1] / (H - 1)) * 2 - 1

    new_grid = new_grid.to(device)

    if save_grids:
        grid_dir = os.path.join(save_dir, 'grids')
        if not os.path.exists(grid_dir):
            os.makedirs(grid_dir)
        torch.save(new_grid, os.path.join(grid_dir, name + '.pt'))
        print("Saved grid to", os.path.join(grid_dir, name + '.pt'))

    def warp_tensor(tensor_to_warp, new_grid):
        tensor_to_warp_t = torch.tensor(tensor_to_warp, device=device)[None].float()
        warped_tensor = F.grid_sample(tensor_to_warp_t, new_grid, mode='bilinear', align_corners=True)
        warped_tensor = warped_tensor.squeeze(0).cpu().numpy()
        warped_tensor = np.nan_to_num(warped_tensor, nan=0)
        return warped_tensor

    tic_to_warp = tic_to_warp.transpose((2, 0, 1))
    # reference_tic = reference_tic.transpose((2, 0, 1))
    warped_tensor = warp_tensor(tensor_to_warp, new_grid)
    tic_to_warp = warp_tensor(tic_to_warp, new_grid).transpose((1, 2, 0)).astype(np.uint8)
    # reference_tic = warp_tensor(reference_tic, new_grid).transpose((1, 2, 0)).astype(np.uint8)

    reference_tic = reference_tic[::-1]
    tic_to_warp = tic_to_warp[::-1]

    def scale_t1(viz):
        result = np.zeros((viz.shape[0], 5 * viz.shape[1], 3))
        for i in range(5):
            result[:, i::5, :] = viz
        return result

    # make all same size)
    if reference_tic.shape[1] > tic_to_warp.shape[1]:
        larger = np.zeros_like(reference_tic)
        h, w, c = tic_to_warp.shape
        larger[:h, :w] = tic_to_warp
        tic_to_warp = larger
        larger = np.zeros_like(reference_tensor)
        larger[:, :h, :w] = warped_tensor
        warped_tensor = larger
    elif reference_tic.shape[1] < tic_to_warp.shape[1]:
        smaller = np.zeros_like(reference_tic)
        h, w, c = reference_tic.shape
        smaller[:, :] = tic_to_warp[:, :h, :w]
        reference_tic = smaller
        smaller = np.zeros_like(warped_tensor)
        smaller[:, :, :] = warped_tensor[:, :h, :w]
        warped_tensor = smaller

    reference_tic = scale_t1(reference_tic)
    tic_to_warp = scale_t1(tic_to_warp)
    # blend with original image
    blended = cv2.addWeighted(reference_tic, 0.5, tic_to_warp, 0.5, 0)
    # print("Saving to", save_dir)
    if viz:
        save_dir = '/mnt/personal/hlavsja3/new_registration_10/viz'
        save_dir = os.path.join(save_dir, suffix)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir, 'blend')):
            os.makedirs(os.path.join(save_dir, 'blend'))
        if not os.path.exists(os.path.join(save_dir, 'warp')):
            os.makedirs(os.path.join(save_dir, 'warp'))
        cv2.imwrite(os.path.join(save_dir, 'blend', name + '_blended.png'), blended[..., ::-1])
        # cv2.imwrite(os.path.join(save_dir, 'reference.png'), reference_tic[..., ::-1])
        cv2.imwrite(os.path.join(save_dir, 'warp', name + '_warped.png'), tic_to_warp[..., ::-1])
    return torch.from_numpy(warped_tensor)


def normalize_filename(file_name):
    file_name = file_name.replace('.cdf', '')
    file_name = file_name.replace('asi', '')
    file_name = file_name.replace('-', '_')
    file_name = file_name.replace('.', '_')
    file_name = file_name.replace('__', '_')
    return file_name


@torch.no_grad()
def fill_spectrogram_image_torch_fast(
        ds,                 # HDF5 / Arrow / NPZ dataset
        spectro_img,        # (C, H, W)  in‑place tensor on GPU
        data_ptr: int,
        x_start: int,
        x_len: int,
        y_len: int,
        model: torch.nn.Module,
        max_gpu_points: int = 20_000_000   # safety for very long blocks
):
    """
    Fully batched replacement for the original nested‑loop routine.
    Assumes `spectro_img` is already on the same device as `model`.
    Returns the updated data pointer.
    """
    device  = spectro_img.device
    img_dtype = spectro_img.dtype  # keep everything coherent
    compute_dtype = next(model.parameters()).dtype

    n_pix   = x_len * y_len            # total #scans we must draw
    idx0    = data_ptr                 # first scan id
    idx1    = data_ptr + n_pix         # one‑past‑last

    ## ------------------------------------------------------------------ ##
    # 1. Pull *metadata* for the whole slab in one shot (CPU RAM)
    scan_idx   = torch.as_tensor(ds['scan_index'][idx0:idx1],  dtype=torch.long)
    pt_cnt     = torch.as_tensor(ds['point_count'][idx0:idx1], dtype=torch.long)

    # 2. Read the *contiguous* span of m/z and intensity that covers **all**
    #    those scans – only ONE I/O + ONE host→device copy.
    ptr_first  = int(scan_idx[0])
    ptr_last   = int(scan_idx[-1] + pt_cnt[-1])
    mass_val   = torch.as_tensor(ds['mass_values'][ptr_first:ptr_last], dtype=compute_dtype, device=device)
    inten_val  = torch.as_tensor(ds['intensity_values'][ptr_first:ptr_last], dtype=compute_dtype, device=device)

    # 3. Build a map that tells, for every raw sample, which output pixel it
    #    belongs to.  (repeat_interleave is ~80 ns on 1 M points).
    pix_id     = torch.repeat_interleave(
                    torch.arange(n_pix, device=device), pt_cnt.to(device))

    # 4. Run the kernel *once*.  If the block is gigantic, chunk it.
    def run_kernel(mz, inten):
        T = model(mz)  # (K, 1, N) or (K, N, 1) or (N, K)
        if T.dim() == 3:  # handle (K,1,N) or (K,N,1)
            if T.size(1) == 1:
                T = T.squeeze(1)  # → (K, N)
            elif T.size(2) == 1:
                T = T.squeeze(2)  # → (K, N)
        elif T.dim() == 2 and T.size(0) != K:
            T = T.t()  # (N, K) → (K, N)

        return T * inten.unsqueeze(0)  # broadcasting to (K, N)

    K = model.centers.shape[0]
    mass_accum = torch.zeros(
        K,  # K  – or infer from a dummy pass
        n_pix,
        device=device,
        dtype=compute_dtype
    )

    # .expand() = Returns a new view of the self tensor with singleton dimensions expanded to a larger size.
    index = pix_id.unsqueeze(0).expand(K, -1)  # (K, N_points)

    # optional safety chunking
    if mass_val.numel() <= max_gpu_points:
        src = run_kernel(mass_val, inten_val)  # (K, N_points)
        mass_accum.scatter_add_(
            dim=1,
            index=index,
            src=src
        )
    else:
        # fall back to streaming in reasonably sized slices
        logger.debug("Filling spectrogram image in chunks, mass_val.numel() = %d", mass_val.numel())
        offset = 0
        while offset < mass_val.numel():
            take   = min(max_gpu_points, mass_val.numel() - offset)
            sl     = slice(offset, offset + take)
            mass_accum.scatter_add_(
                1,
                pix_id[sl].expand(K, -1),
                run_kernel(mass_val[sl], inten_val[sl])
            )
            offset += take

    # 5. Reshape and write the block to the image in one assignment
    mass_accum = mass_accum.view(-1, x_len, y_len).to(img_dtype)        # (C, H, W_block)
    mass_accum = mass_accum.permute(0, 2, 1)  # (C, W_block, H) for image
    spectro_img[:, :y_len, x_start:x_start + x_len] = mass_accum
    return idx1                                           # new data_ptr


@torch.no_grad()
def fill_spectrogram_image_torch(ds, spectrogram_image, data_pointer, x_start, x_length, y_length, model):
    for j in trange(x_start, x_start + x_length):
        for i in range(y_length):
            # Extract scan index range
            si_s = ds['scan_index'][data_pointer]
            si_t = si_s + ds['point_count'][data_pointer]

            # Load mass values and intensities
            indexes = ds['mass_values'][si_s:si_t]
            intensities = ds['intensity_values'][si_s:si_t]

            # Compute kernel weights for all K in batch
            T = model(indexes)  # Now processes all K at once
            T = T * intensities.unsqueeze(0)  # (K, num_samples, N)
            kernel_weights_per_position = T.squeeze(1) if T.size(1) == 1 else T
            mass_data = kernel_weights_per_position.sum(dim=1)  # Sum across mass index dimension

            # Store results in spectrogram image
            spectrogram_image[:, i, j] = mass_data

            # Move to next data point
            data_pointer += 1

    return data_pointer


def prepare_one_data_new_representation(model, path, data):
    start = time.time()
    logger.debug(f"Loading dataset from {path}...")

    logger.debug(f"Dataset loaded in {time.time() - start:.2f} seconds.")

    shape_ = data['scan_index'].shape[0]
    x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = \
        set_system(os.path.basename(path), shape_)

    # save image of spectrogram
    def store_image(spectrogram_image, filename='spectrogram_image.png'):
        spectrogram_image = rearrange(spectrogram_image, 'C H W -> H W C')
        spectrogram_image = spectrogram_image.cpu().numpy()
        spectrogram_image = (spectrogram_image - spectrogram_image.min()) / (
                    spectrogram_image.max() - spectrogram_image.min())
        spectrogram_image = (spectrogram_image * 255).astype(np.uint8)
        cv2.imwrite(filename, spectrogram_image)


    spectrogram_image = torch.zeros((model.centers.shape[0],
                                     y_ten_seconds,
                                     x_six_seconds + x_eight_seconds + x_ten_seconds),
                                    dtype=torch.float32).to(device)

    data_pointer_spectrogram = 0
    start = time.time()
    if x_six_seconds > 0:
        logger.debug("Loading first part")
        data_pointer_spectrogram = fill_spectrogram_image_torch_fast(data, spectrogram_image, data_pointer_spectrogram, 0,
                                                                x_six_seconds, y_six_seconds, model)
        logger.debug(f"First part loaded in {time.time() - start:.2f} seconds, data_pointer_spectrogram: {data_pointer_spectrogram}")

    if x_eight_seconds > 0:
        logger.debug("Loading second part")
        start = time.time()
        data_pointer_spectrogram = fill_spectrogram_image_torch_fast(data, spectrogram_image, data_pointer_spectrogram,
                                                                x_six_seconds, x_eight_seconds, y_eight_seconds, model)
        logger.debug(f"Second part loaded in {time.time() - start:.2f} seconds, data_pointer_spectrogram: {data_pointer_spectrogram}")

    if x_ten_seconds > 0:
        logger.debug("Loading third part")
        start = time.time()
        _ = fill_spectrogram_image_torch_fast(data, spectrogram_image, data_pointer_spectrogram, x_six_seconds + x_eight_seconds,
                                         x_ten_seconds, y_ten_seconds, model)
        logger.debug(f"Third part loaded in {time.time() - start:.2f} seconds, data_pointer_spectrogram: {data_pointer_spectrogram}")

    # if logger level is DEBUG, store the spectrogram image
    if logger.level == logging.DEBUG:
        store_image(spectrogram_image, 'spectrogram_image.png')
        exit()

    return spectrogram_image


DS = None
DS_FILEPATH = None

def get_ds(cdf_filepath):
    global DS, DS_FILEPATH
    if DS is None or DS_FILEPATH != cdf_filepath:
        with nc.Dataset(cdf_filepath) as ds:
            DS = {
                'mass_range_min': torch.tensor(ds['mass_range_min'][:].compressed(), device=device),
                'scan_index': torch.tensor(ds['scan_index'][:].compressed(), device=device),
                'point_count': torch.tensor(ds['point_count'][:].compressed(), device=device),
                'mass_values': torch.tensor(ds['mass_values'][:].compressed(), device=device),
                'intensity_values': torch.tensor(ds['intensity_values'][:].compressed(), device=device),
                'total_intensity': torch.tensor(ds['total_intensity'][:].compressed(), device=device),
                'mass_range_max': torch.tensor(ds['mass_range_max'][:].compressed(), device=device)
            }
        DS_FILEPATH = cdf_filepath
    return DS


@torch.no_grad
def store_one_file(cdf_filepath, settings, reference_positions):
    cdf_basename = os.path.basename(cdf_filepath)
    banned_compounds = ['Heptacosane', 'Nonacosane']
    reference_positions_without_banned = {k: v for k, v in reference_positions.items() if k not in banned_compounds}

    # K = 1
    if len(channel_experiments_dir) == 0:
        logger.info(f"Processing K=1 with Sum Kernel")

        model_classname = "models.SumBank"
        model = get_obj_from_str(model_classname)()

        reference_tensor = torch.load(reference_tensor_filepath)[:1].to(device)
        logger.debug(f"REFERENCE_TENSOR shape: {reference_tensor.shape}")

        representation_save_dir = os.path.join(output_dir, f"{model_classname}_{1}")
        save_name = normalize_filename(cdf_basename) + '.pt'

        # load state dict
        for split in range(1, 11):
            model = model.to(device)

            all_exist = check_if_all_exist(representation_save_dir, save_name, split)
            if all_exist:
                logger.debug(f"Representation for {save_name} already exists for split {split:02d}, skipping.")
                continue

            data = get_ds(cdf_filepath)

            run_single_split(split, cdf_filepath, banned_compounds, data, "NONE", model, model_classname,
                             reference_positions, reference_positions_without_banned, reference_tensor,
                             representation_save_dir, save_dir_detections, save_name, settings)

    for experiment_dirname in os.listdir(channel_experiments_dir):
        experiment_dir = os.path.join(channel_experiments_dir, experiment_dirname)
        if not os.path.isdir(experiment_dir):
            continue
        if experiment_dirname[0] not in ['1', '3', '5', '10', '20']:
            continue
        logger.info(f"Processing experiment directory: {experiment_dir}")
        experiment_config = OmegaConf.load(os.path.join(experiment_dir, 'config.yaml'))
        global_min, global_max = extract_global_min_max(os.path.join(experiment_dir, 'log'))

        if experiment_config.centers_mode == 'limits':
            if experiment_config.model.params.K <= 3:
                centers = torch.linspace(global_min, global_max, experiment_config.model.params.K, device=device)
            else:
                centers = custom_centers(K=experiment_config.model.params.K, max_val=global_max).to(device)
        elif experiment_config.centers_mode == 'centers':
            centers = custom_centered_centers(K=experiment_config.model.params.K, max_val=global_max).to(device)
        else:
            raise ValueError(f"Unknown centers_mode: {experiment_config.centers_mode}")

        model = get_obj_from_str(experiment_config.model.classname)(
            centers, **experiment_config.model.params
        )

        reference_tensor = torch.load(reference_tensor_filepath)[:experiment_config.model.params.K].to(device)
        logger.debug(f"REFERENCE_TENSOR shape: {reference_tensor.shape}")

        model_classname = experiment_config.model.classname.split('.')[-1]
        representation_save_dir = os.path.join(output_dir, f"{model_classname}_"
                                                                f"{experiment_config.loss}_"
                                                                f"{experiment_config.centers_mode}_"
                                                                f"{str(experiment_config.model.params.K)}")

        save_name = normalize_filename(cdf_basename) + '.pt'

        # load state dict
        for split in range(1, 11):
            split_experiment_dir = os.path.join(experiment_dir, f'split_{split:02d}')
            model_filepath = os.path.join(split_experiment_dir, 'model.pt')
            state_dict_with_module = torch.load(model_filepath, map_location='cpu')
            state_dict = {k.replace('module.', ''): v for k, v in state_dict_with_module.items()}
            model.load_state_dict(state_dict)
            model = model.to(device)

            all_exist = check_if_all_exist(representation_save_dir, save_name, split)
            if all_exist:
                logger.debug(f"Representation for {save_name} already exists for split {split:02d}, skipping.")
                continue

            data = get_ds(cdf_filepath)

            run_single_split(split, cdf_filepath, banned_compounds, data, experiment_config.loss, model, model_classname,
                             reference_positions, reference_positions_without_banned, reference_tensor,
                             representation_save_dir, save_dir_detections, save_name, settings)


def check_if_all_exist(representation_save_dir, save_name, split):
    subdir_names = ['noalign', 'bbox_11x101_plus_cos_all', 'bbox_11x101_plus_cos_22', 'fcn_plus_cos_all',
                    'fcn_plus_cos_22']
    all_exist = True
    for subdir_name in subdir_names:
        target_filepath = os.path.join(representation_save_dir, subdir_name, f"{split:02d}",
                                       save_name.replace('.pt', '.h5'))
        if not os.path.exists(target_filepath):
            all_exist = False
    return all_exist


def run_single_split(split, cdf_filepath, banned_compounds, data, experiment_loss, model, model_classname,
                     reference_positions, reference_positions_without_banned, reference_tensor, representation_save_dir,
                     save_dir_detections, save_name, settings):
        start = time.time()
        logger.debug(f"Preparing spectrogram_image: {cdf_filepath} with model: {model_classname} and loss: {experiment_loss}")
        spectrogram_image = prepare_one_data_new_representation(model, cdf_filepath, data)
        logger.debug(f"Prepared spectrogram_image: {cdf_filepath} in {time.time() - start:.2f} seconds")

        save_hdf5(spectrogram_image.clone(), save_name.replace('.pt', ''),
                  os.path.join(representation_save_dir, 'noalign', f"{split:02d}"),
                  save_name.replace('.pt', '.h5'), logger=logger)

        for i, set in enumerate(settings):
            if not os.path.exists(os.path.join(save_dir_detections, set.metric)):
                os.makedirs(os.path.join(save_dir_detections, set.metric))

            compounds_position = {}
            with open(os.path.join(save_dir_detections, set.metric, save_name.replace('.pt', '.txt')), 'r') as f:
                for line in f:
                    line = line.replace('\n', '')
                    parts = line.rsplit(maxsplit=2)
                    compound = parts[0]
                    t1, t2 = parts[1], parts[2]
                    compounds_position[compound] = (int(t1), int(t2))

            start = time.time()
            logger.debug(f"[{split:02d}/10] Warping tensor for {set.metric} with {len(compounds_position)} compounds")
            compounds_position_without_banned = {k: v for k, v in compounds_position.items() if k not in banned_compounds}
            warped_tensor = warp_tensor(reference_tensor, spectrogram_image.clone(), reference_positions,
                                        compounds_position,
                                        save_name,
                                        suffix=set.metric + '_all')
            logger.debug(f"[{split:02d}/10] Warped tensor for {set.metric} in {time.time() - start:.2f} seconds")

            subdir_name = 'bbox_11x101_plus_cos_all'
            if set.metric == 'cnn':
                subdir_name = 'fcn_plus_cos_all'

            save_hdf5(warped_tensor, save_name.replace('.pt', ''),
                      os.path.join(representation_save_dir, subdir_name, f"{split:02d}"),
                      save_name.replace('.pt', '.h5'), logger=logger)

            start = time.time()
            logger.debug(f"[{split:02d}/10] Warping tensor for {set.metric} with {len(compounds_position)} compounds (22)")
            warped_tensor = warp_tensor(reference_tensor, spectrogram_image.clone(),
                                        reference_positions_without_banned,
                                        compounds_position_without_banned,
                                        save_name,
                                        suffix=set.metric + '_without_banned')
            logger.debug(f"[{split:02d}/10] Warped tensor for {set.metric} (22) in {time.time() - start:.2f} seconds")

            subdir_name = 'bbox_11x101_plus_cos_22'
            if set.metric == 'cnn':
                subdir_name = 'fcn_plus_cos_22'

            save_hdf5(warped_tensor, save_name.replace('.pt', ''),
                      os.path.join(representation_save_dir, subdir_name, f"{split:02d}"),
                      save_name.replace('.pt', '.h5'), logger=logger)


def set_settings():
    settings1 = Settings(5, 50, 'dot', True, lambda x: x ** 4, True)
    settings2 = Settings(None, None, 'cnn', None, None, True)
    return [settings1, settings2]


def load_reference_positions(reference_system_number=1, t1_shift=-500):

    compounds_times = load_compound_and_times(reference_positions_filepath)

    with nc.Dataset(reference_cdf_filepath) as ds:
        shape_ = ds['scan_index'].shape[0]

    x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = \
        set_system(os.path.basename(reference_cdf_filepath), shape_)

    reference_compounds_pixels = from_times_pixels(compounds_times, int(reference_system_number),
                                                   x_six_seconds,
                                                   y_six_seconds,
                                                   x_eight_seconds, y_eight_seconds, x_ten_seconds,
                                                   y_ten_seconds,
                                                   t1_shift)
    return reference_compounds_pixels


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = ArgumentParser()
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes in the job')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank of the process')
    parser.add_argument('--logger_level', type=int, default=20, help='Local rank of the process')
    args = parser.parse_args()

    logger.setLevel(args.logger_level)

    dataset_dir = os.environ["DATASET_DIR"]
    hsd_dataset_dir = dataset_dir + "/HSD/"
    channel_experiments_dir = dataset_dir + "/channels/"
    reference_positions_filepath = channel_experiments_dir + '/system_1_target_F4_05.txt'
    reference_tensor_filepath = channel_experiments_dir + '/F4_05_system1.pt'
    reference_cdf_filepath = hsd_dataset_dir + '/F4_05_system1.cdf'
    save_dir_detections = channel_experiments_dir + '/detected_positions/'
    output_dir = dataset_dir + "/stored_channel_representation/"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    settings_array = set_settings()
    reference_positions = load_reference_positions()
    logger.info("Reference tensor and positions loaded")
    logger.info(f"World size: {args.world_size}, Local rank: {args.local_rank}")

    files = os.listdir(hsd_dataset_dir)[args.local_rank::args.world_size]
    for file_idx, basename in enumerate(files):
        file_start = time.time()
        logger.info(f"Processing file {file_idx + 1}/{len(files)}: {basename}")
        cdf_filepath = os.path.join(hsd_dataset_dir, basename)
        try:
            store_one_file(cdf_filepath, settings_array, reference_positions)
            logger.info(f"Processed file {basename} in {time.time() - file_start:.2f} seconds")
        except Exception as e:
            logger.error(f"Error processing file {basename}: {e}")
            continue
