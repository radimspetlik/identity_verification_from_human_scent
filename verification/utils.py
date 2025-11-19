import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from PIL import Image
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import random
from torch.utils.data import Sampler
import torch
from torchvision.transforms import transforms
from tqdm import tqdm, trange

from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
import torch.nn.functional as F
import hdf5

import torch as th
import numpy as np


def add_gaussian_noise_per_sample(x: torch.Tensor, frac: float = 0.1):
    """
    Element-wise Gauss. std = frac * (median|mean) ze vstupu.
    x: [C,H,W]
    frac=0.05 -> ~5 % relativní šum.
    """
    noise = torch.randn_like(x) * frac * x
    return x + noise


def add_contrast_per_channel(x: torch.Tensor, frac: float = 0.1, per_channel: bool = True):
    """
    Contrast augmentation per channel.

    Args:
        x: [C,H,W] tensor
        frac: maximum relative change of contrast (e.g. 0.1 = ±10%)
        per_channel: if True, contrast is relative to mean per channel,
                     if False, relative to global mean

    Returns:
        Tensor with contrast adjusted
    """
    # mean to adjust around
    if per_channel:
        mean = x.mean(dim=(1, 2), keepdim=True)  # [C,1,1]
    else:
        mean = x.mean()  # scalar

    # random factor in [1-frac, 1+frac]
    factor = 1.0 + (2 * torch.rand(1).item() - 1) * frac

    return (x - mean) * factor + mean

def add_brightness_per_channel(x: torch.Tensor, frac: float = 0.1, per_channel: bool = True):
    """
    Brightness augmentation per channel.

    Args:
        x: [C,H,W] tensor
        frac: maximum relative change of brightness (e.g. 0.1 = ±10%)
        per_channel: if True, brightness shift is applied per channel,
                     if False, one global shift is applied

    Returns:
        Tensor with brightness adjusted
    """
    if per_channel:
        # random shift factor per channel
        shift = 1.0 + (2 * torch.rand(x.shape[0], 1, 1) - 1) * frac
    else:
        # one global shift factor
        shift = 1.0 + (2 * torch.rand(1).item() - 1) * frac

    shift = shift.to(x.device)

    return x * shift


def load_data(device, root_dir, extensions=('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'pt', 'h5'), max_id=500, min_id=1,
              measurement_number_to_exclude=None, measurement_number_to_include=None, files_to_include=None,
              detector_files_to_include=None):
    """
    Loads and organizes image files from a directory into groups and subsets.

    Args:
        root_dir (str): Path to the directory containing image files.
        extensions (tuple): File extensions to include (default: common image and tensor formats).
        max_id (int): Maximum allowed identifier value (inclusive).
        min_id (int): Minimum allowed identifier value (inclusive).
        measurement_number_to_exclude (list or None): List of measurement numbers to exclude.
        measurement_number_to_include (list or None): List of measurement numbers to include.
        files_to_include (list or None): List of filenames to include; if None, all files are considered.

    Returns:
        tuple: (data, id_counts)
            data (dict): Nested dictionary with group ('F' or 'M') and identifier as keys, mapping to lists of filenames.
            id_counts (torch.Tensor): Tensor counting the number of files per identifier for each group.
    """
    data = {'F': {}, 'M': {}}
    if max_id is None:
        max_id = 500
    if min_id is None:
        min_id = 1
    id_counts = torch.zeros(2 * (max_id - min_id + 1)).to(device)
    # Load all images and split into groups and subsets
    for filename in tqdm(os.listdir(root_dir)):
        stem, _ = os.path.splitext(filename)
        if detector_files_to_include is not None and filename not in detector_files_to_include:
            continue

        if files_to_include is not None and filename not in files_to_include and stem not in files_to_include:
            continue
        if "race" in filename.lower():
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
                meassurement_number = filename.replace('.h5', '').replace('P', '').replace('(S)', '').split('_')[1]
                meassurement_number = int(meassurement_number)
                if measurement_number_to_exclude is not None:
                    if meassurement_number in measurement_number_to_exclude:
                        continue
                if measurement_number_to_include is not None:
                    if meassurement_number not in measurement_number_to_include:
                        continue
                identifier = int(identifier) if identifier else None
                if max_id is not None:
                    if identifier > max_id:
                        continue
                if min_id is not None:
                    if identifier < min_id:
                        continue
                if identifier is not None:
                    if identifier not in data[group]:
                        data[group][identifier] = []
                    data[group][identifier].append(filename)
                if group == 'M':
                    id_counts[identifier - min_id] += 1
                else:
                    id_counts[(max_id - min_id + 1) + identifier - min_id] += 1

    return data, id_counts


class GenderSplitDataset(Dataset):
    """
        A PyTorch Dataset for loading and processing gender-split image data with support for masks, caching,
        distributed loading, normalization.

        Args:
         root_dir (str): Root directory containing the image files.
         gender_and_identity_sorted_dict (dict): Nested dict with group ('F' or 'M') and identifier as keys, mapping to lists of filenames.
         master_mask_filepath (str or None): Filepath to the master mask image to apply to all samples.
         transform (callable, optional): Transform to apply to mask and images (not tensors).
         cache_dir (str or None): Directory to cache loaded images/tensors.
         transform_shape (callable, optional): Additional transform for image shape (only for images, not tensors).
         normalize (bool): Whether to normalize images using computed mean and std.
         only_n_measurements (int or None): If set, limits the number of measurements per identity.
         normalize_to_one (bool): If True, normalizes images to [0, 1].
         log (bool): If True, applies logarithm to images.
         exp (bool): If True, applies exponential to images.
         poly (bool): If True, squares the images.

        Attributes:
         data (dict): The gender and identity sorted dictionary.
         all_images (list): List of tuples (group, identifier, filename) for all images.
         master_mask (torch.Tensor or None): The loaded and processed master mask.
         mean (float): Mean of the dataset (if computed).
         std (float): Standard deviation of the dataset (if computed).
    """
    def __init__(self, device, root_dir, gender_and_identity_sorted_dict, master_mask_filepath, transform=None,
                 cache_dir=None, transform_shape=None, normalize=False, only_n_measurements=None,
                 normalize_to_one=False, log=False, exp=False, poly=False, gauss_noise=False, max_percentage=0.05,
                 contrast=False, contrast_frac=0.1, brightness=False, brightness_frac=0.1):
        self.transform = transform
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.transform_shape = transform_shape
        self.data = gender_and_identity_sorted_dict
        self.mean = 0.0
        self.std = 0.0
        self.max = 0.0
        self.normalize = normalize
        self.normalize_to_one = normalize_to_one
        self.log = log
        self.exp = exp
        self.poly = poly
        self.gauss_noise = gauss_noise
        self.max_percentage = max_percentage
        self.contrast = contrast
        self.contrast_frac = contrast_frac
        self.brightness = brightness
        self.brightness_frac = brightness_frac

        print("Gauss noise:", self.gauss_noise)
        print("Contrast:", self.contrast)
        print("Brightness:", self.brightness)

        self.all_images = [(group, identifier, filename)
                        for group in self.data
                        for identifier in self.data[group]
                        for filename in self.data[group][identifier]]

        if master_mask_filepath is not None:
            self.master_mask = self.transform(Image.open(master_mask_filepath).convert('F'))[:, :, ::5].float()
            self.master_mask = torch.flip(self.master_mask, [1])
            self.master_mask -= self.master_mask.min()
            self.master_mask /= self.master_mask.max()
            # move master mask to device
            self.master_mask = self.master_mask.to(device)

            # find first and last non-zero column and row
            col_sums = (self.master_mask[0].sum(0) > 0).float().nonzero()
            row_sums = (self.master_mask[0].sum(1) > 0).float().nonzero()
        self.first_dim = 0
        self.first_col = col_sums[0]
        self.last_col = col_sums[-1]
        self.first_row = row_sums[0]
        self.last_row = row_sums[-1]

        if only_n_measurements is not None:
            self.filter_n_measurements_per_id(only_n_measurements)
        #self.find_max()
        #print("Max max:", self.max)

    def find_max(self):
        global_max = None

        for idx in range(len(self.all_images)):
            image, group, identifier, image_filename, idx = self.get_one_item(idx)
            max_val_per_channel = image.amax(dim=(1, 2))  # shape [C]

            if global_max is None:
                global_max = max_val_per_channel
            else:
                global_max = torch.maximum(global_max, max_val_per_channel)

        self.max = global_max.view(-1, 1, 1).cpu()




    def filter_n_measurements_per_id(self, n_measurements):
        new_data = {'F': {}, 'M': {}}
        for group in self.data:
            for identifier in self.data[group]:
                new_data[group][identifier] = self.data[group][identifier][:n_measurements]
        self.data = new_data
        print(new_data)
        self.all_images = [(group, identifier, filename)
                           for group in self.data
                           for identifier in self.data[group]
                           for filename in self.data[group][identifier]]

    def compute_mean_std(self):
        for _, _, image_filename in self.all_images:
            if '.pt' == image_filename[-3:]:
                image = self.load(self.load_tensor, image_filename)
            else:
                image = self.load(self.load_image, image_filename)
            image = image.type(torch.float32)
            self.mean += image.mean()
            self.std += image.std()
        self.mean /= len(self.all_images)
        self.std /= len(self.all_images)
        return self.mean, self.std

    def __len__(self):
        # Calculate the total number of images across all groups and subsets
        # return sum(len(images) for group in self.data.values() for images in group.values())
        return len(self.all_images)

    def load_tensor(self, filepath):
        image = torch.load(filepath, weights_only=False)
        image = image[self.first_dim:, self.first_row:self.last_row, self.first_col:self.last_col]
        image = image.type(torch.float32)
        if self.master_mask is not None:
            image *= self.master_mask
        if self.master_mask is not None:
            _, h, w = self.master_mask.shape
        larger = torch.zeros((1, h, w))
        _, h, w = image.shape
        larger[:, :h, :w] = image
        image = larger.float()
        return image

    def load_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        # Apply any transformations if provided
        if self.transform:
            image = self.transform(image)
        image = image[:, :, ::5]
        if self.master_mask is not None:
            image *= self.master_mask
        if self.transform_shape:
            image = self.transform_shape(image)
        return image

    def load_tensor_h5(self, img_path):
        file_name = img_path.split('/')[-1]
        dir = img_path.split('/')[:-1]
        dir = '/'.join(dir)
        name = file_name.split('.')[0]
        image = hdf5.load_hdf5(name, dir, file_name)
        if image is None:
            return None
        image = image.type(torch.float32)
        _, h, w = self.master_mask.shape
        if image.shape[1] != h or image.shape[2] != w:
            #resize image to match master mask
            image = F.interpolate(image.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        image *= self.master_mask
        image = image[self.first_dim:, self.first_row:self.last_row, self.first_col:self.last_col]
        return image

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            return self.get_one_item(idx)
        else:
            image1, group1, identifier1, image_filename1, idx1 = self.get_one_item(idx[0])
            image2, group2, identifier2, image_filename2, idx2 = self.get_one_item(idx[1])
            return (image1, image2), (group1, group2), (identifier1, identifier2), (image_filename1, image_filename2), (idx1, idx2)

    def get_one_item(self, idx, load_image=True):
        # Get the image path based on the index
        group, identifier, image_filename = self.all_images[idx]

        image = None
        if load_image:
            if '.pt' == image_filename[-3:]:
                image = self.load(self.load_tensor, image_filename)
            elif '.h5' == image_filename[-3:]:
                image = self.load(self.load_tensor_h5, image_filename)
            else:
                image = self.load(self.load_image, image_filename)

            if self.log:
                # add to minimal value something to be over 1
                image += 1
                image = torch.log(image)
            elif self.exp:
                image = torch.exp(image)
            elif self.poly:
                image = image ** 2
            if self.normalize_to_one:
                # scale to 0-1
                image = image / image.max()

            if self.normalize:
                image = (image - self.mean) / self.std


            if self.brightness:
                image = add_brightness_per_channel(image, frac=self.brightness_frac, per_channel=True)


            if self.contrast:
                image = add_contrast_per_channel(image, frac=self.contrast_frac, per_channel=True)


            if self.gauss_noise:
                image = add_gaussian_noise_per_sample(image,self.max_percentage)

        group = 1.0 if group == 'M' else 0.0

        return image, group, identifier, image_filename, idx


    def load(self, loading_function, filename):
        root_image_filepath = os.path.join(self.root_dir, filename)
        if '.h5' in filename:
            filename_cache = filename.replace('.h5', '.pt')
        else:
            filename_cache = filename
        cached_filepath = os.path.join(self.cache_dir, filename_cache)
        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
            if os.path.exists(cached_filepath):
                image = torch.load(cached_filepath, weights_only=False)
            else:
                image = loading_function(root_image_filepath)
                torch.save(image, cached_filepath)
        else:
            image = loading_function(root_image_filepath)

        return image

    def num_identities(self):
        return sum(len(self.data[group]) for group in self.data)


class GenderSplitDatasetWeakFeatures(Dataset):
    def __init__(self, root_dir, gender_and_identity_sorted_dict):
        self.root_dir = root_dir

        # Dictionary to hold the split data
        self.data = gender_and_identity_sorted_dict
        self.mean = 0.0
        self.std = 0.0

        self.all_images = [(group, identifier, filename)
                           for group in self.data
                           for identifier in self.data[group]
                           for filename in self.data[group][identifier]]



    def __len__(self):
        # Calculate the total number of images across all groups and subsets
        # return sum(len(images) for group in self.data.values() for images in group.values())
        return len(self.all_images)


    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            return self.get_one_item(idx)
        else:
            group1, identifier1, image_filename1 = self.all_images[idx[0]]
            group2, identifier2, image_filename2 = self.all_images[idx[1]]
            return (group1,group2), (identifier1, identifier2), (image_filename1, image_filename2), (idx[0], idx[1])

    def get_one_item(self, idx):
        # Get the image path based on the index
        group, identifier, image_filename = self.all_images[idx]

        return group, identifier, image_filename


    def num_identities(self):
        return sum(len(self.data[group]) for group in self.data)

class BalancedSamplerIds(Sampler):
    def __init__(self, dataset, sampler_size=None):
        super().__init__()
        self.dataset = dataset
        self.samples = {'F': {}, 'M': {}}
        self.sampler_size = sampler_size
        self.m_ids = []
        self.f_ids = []

        # Organize dataset indices by group and identifier
        for idx in range(len(dataset)):
            group, identifier, _ = dataset.all_images[idx]
            if identifier not in self.samples[group]:
                self.samples[group][identifier] = []
            if group == 'M':
                self.m_ids.append(identifier)
            else:
                self.f_ids.append(identifier)
            self.samples[group][identifier].append(idx)
        self.min_samples = np.inf
        for group in ['F', 'M']:
            for identifier in self.samples[group]:
                if len(self.samples[group][identifier]) > 1:
                    self.min_samples = min(self.min_samples, len(self.samples[group][identifier]))
                else:
                    print(f"Warning: No samples for group {group}, identifier {identifier}")
        self.f_ids = [f_id for f_id in self.samples['F']]
        self.m_ids = [m_id for m_id in self.samples['M']]

    def __len__(self):
        return len(self.balanced_indices)

    def __iter__(self):
        self.balanced_indices = []
        if self.sampler_size is None or self.sampler_size > self.min_samples:
            self.sampler_size = self.min_samples
        if self.sampler_size % 2 != 0 and self.sampler_size > 1:
            self.sampler_size -= 1
        # print("Sampler size:", self.sampler_size)
        for f_id, m_id in zip(self.f_ids, self.m_ids):
            # should never be
            if self.sampler_size is not None:
                if len(self.samples['F'][f_id]) >= self.min_samples:
                    self.load_gender('F', f_id)
                if len(self.samples['M'][m_id]) >= self.min_samples:
                    self.load_gender('M', m_id)
            else:
                raise RuntimeError("Sampler size not set")

        return iter(self.balanced_indices)

    def num_identities(self):
        return len(self.f_ids) + len(self.m_ids)

    def load_gender(self, group, identifier):
        samples = random.sample(self.samples[group][identifier], self.sampler_size)
        # shuffle samples
        random.shuffle(samples)
        pos_sumples_tuples = [(samples[i], samples[i + 1]) for i in range(0, len(samples), 2)]
        self.balanced_indices.extend(pos_sumples_tuples)
        # pick random negative sample
        if len(samples) > 1:
            for i in range(len(samples) // 2):
                if i % 2 == 0:
                    # pick random men id different from current
                    m_diff_id = identifier
                    while m_diff_id == identifier:
                        m_diff_id = random.choice(self.m_ids)
                    self.balanced_indices.append((samples[i], random.choice(self.samples['M'][m_diff_id])))
                else:
                    f_diff_id = identifier
                    while f_diff_id == identifier:
                        f_diff_id = random.choice(self.f_ids)
                    self.balanced_indices.append((samples[i], random.choice(self.samples['F'][f_diff_id])))
        else:
            raise RuntimeError("Not enough samples for balanced sampler")


class OneMeassurementPerIDSampler(Sampler):
    """
    A PyTorch Sampler that selects one random measurement (sample) per identity for each group ('F' or 'M').

    This sampler is useful for tasks where only a single measurement per identity should be used, such as
    evaluation or when constructing a balanced set of unique identities.

    Args:
        dataset (Dataset): The dataset containing `all_images` as a list of (group, identifier, filename) tuples.

    Attributes:
        samples (dict): Dictionary mapping group and identifier to a list of dataset indices.
        f_ids (list): List of all female identifiers.
        m_ids (list): List of all male identifiers.
        balanced_indices (list): List of randomly selected indices, one per identity.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.samples = {'F': {}, 'M': {}}
        self.m_ids = []
        self.f_ids = []

        # Organize dataset indices by group and identifier
        for idx in range(len(dataset)):
            group, identifier, _ = dataset.all_images[idx]
            if identifier not in self.samples[group]:
                self.samples[group][identifier] = []
            if group == 'M':
                self.m_ids.append(identifier)
            else:
                self.f_ids.append(identifier)
            self.samples[group][identifier].append(idx)
        self.f_ids = [f_id for f_id in self.samples['F']]
        self.m_ids = [m_id for m_id in self.samples['M']]

        # pick one sample as an anchor for each id
        self.balanced_indices = []
        for group in ['F', 'M']:
            for identifier in list(self.samples[group].keys()):
                self.balanced_indices.append(random.choice(self.samples[group][identifier]))

    def __len__(self):
        return len(self.balanced_indices)

    def __iter__(self):
        self.balanced_indices = []
        # pick one sample as an anchor for each id
        for group in ['F', 'M']:
            for identifier in list(self.samples[group].keys()):
                self.balanced_indices.append(random.choice(self.samples[group][identifier]))
        return iter(self.balanced_indices)

    def num_identities(self):
        return len(self.f_ids) + len(self.m_ids)


def load_splits(path):
    """
    Parse a split.txt file into a list of dicts with keys 'trn' and 'val'.
    """
    splits = []
    with open(path, 'r', encoding='utf-8') as f:
        # strip only newline; preserve empty strings for blank lines
        lines = [line.rstrip('\n') for line in f]

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        # detect start of a new split
        if line.startswith("Split"):
            trn, val = [], []

            # advance to "Train files:"
            while i < n and not lines[i].strip().startswith("Train files:"):
                i += 1
            i += 1  # skip the "Train files:" line

            # collect all train filenames until blank or next section
            while i < n:
                l = lines[i].strip()
                if not l or l.startswith("Val files:"):
                    break
                trn.append(l)
                i += 1

            # advance to "Val files:" if we're not already on it
            while i < n and not lines[i].strip().startswith("Val files:"):
                i += 1
            i += 1  # skip the "Val files:" line

            # collect all val filenames until blank or next split
            while i < n:
                l = lines[i].strip()
                if not l or l.startswith("Split"):
                    break
                val.append(l)
                i += 1

            splits.append({'trn': trn, 'val': val})
        else:
            i += 1

    return splits

def load_old_detector_file(path):
    old = set()
    with open(path, "r") as f:
        for line in f:
            old.add(line.strip()+".h5")
    return old

def load_new_detector_file(path):
    new = set()
    with open(path, "r") as f:
        for line in f:
            new.add(line.strip()+".h5")
    return new

