import importlib
import re

import torch


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def custom_centered_centers(K: int, max_val: float = 800.0, low_range: float = 300.0) -> torch.Tensor:
    """
    Generate `K` Gaussian centers:
    - (K-2) lower centers: split [0, low_range] into (K-2) equal blocks,
      and place each center in the middle of its block.
    - 2 extra centers: split [low_range, max_val] into 2 equal blocks,
      and place each center in the middle of its block.

    Returns:
        torch.Tensor of shape [K]
    """
    if K < 3:
        raise ValueError("K must be at least 3.")

    # number of lower blocks
    n_low = K - 2
    if K == 3:
        n_low = 2
    low_block = low_range / n_low
    # midpoints of each lower block
    lower_centers = torch.tensor(
        [(i + 0.5) * low_block for i in range(n_low)],
        dtype=torch.float32
    )

    # split the remaining range into 2 blocks
    high_block = (max_val - low_range) / 2
    extra_centers = torch.tensor(
        [low_range + (i + 0.5) * high_block for i in range(K - n_low)],
        dtype=torch.float32
    )

    return torch.cat([lower_centers, extra_centers])


def custom_centers(K: int, max_val: float = 800.0, low_range: float = 300.0) -> torch.Tensor:
    """
    Generate `K` Gaussian centers:
    - (K-2) uniformly in [0, low_range]
    - 2 in [low_range, max_val] with equal spacing from last_low → first_extra → second_extra → max_val

    Returns:
        torch.Tensor of shape [K]
    """
    if K <= 3:
        raise ValueError("K must be at least 3.")

    # (K-2) lower centers
    lower_centers = torch.linspace(0, low_range, steps=K-2)
    last_low = lower_centers[-1].item()

    # Compute spacing
    d = (max_val - last_low) / 3

    # Final two centers
    extra_centers = torch.tensor([last_low + d, last_low + 2 * d])

    return torch.cat([lower_centers, extra_centers])


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


def extract_global_min_max(filepath):
    """
    Reads through a text file looking for a line of the form:
      Global min: <min_value>, Global max: <max_value>, ...
    and returns (min_value, max_value) as floats.

    Raises:
      ValueError: if no matching line is found.
    """
    # Compile a regex that captures the two numbers after “Global min:” and “Global max:”
    pattern = re.compile(
        r"Global\s+min:\s*([-+]?\d*\.?\d+),\s*Global\s+max:\s*([-+]?\d*\.?\d+).*",
        re.IGNORECASE
    )

    with open(filepath, 'r') as f:
        for lineno, line in enumerate(f, start=1):
            match = pattern.search(line)
            if match:
                gmin = float(match.group(1))
                gmax = float(match.group(2))
                return gmin, gmax

    raise ValueError(f"No line with ‘Global min/Global max’ found in {filepath}")