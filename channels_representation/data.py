import os, time, logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np, torch, netCDF4 as nc

import load


weird_files = ['M13_8.cdf', 'M20_07_system2.cdf', 'M15_7.cdf', 'M15_8.cdf', 'M19_07_system2.cdf', 'M16_7.cdf']

def set_system(file_name, shape_, first_chunk=False):
    if (file_name in weird_files):
        print("System WEIRD") if first_chunk else None
        x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = load.set_system_weird()
    elif (shape_ == load.SCAN_LENGHT_SYSTEM_1):
        print("System 1") if first_chunk else None
        x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = load.set_system_1()
    elif (shape_ == load.SCAN_LENGHT_SYSTEM_2):
        print("System 2") if first_chunk else None
        x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = load.set_system_2()
    elif (shape_ == load.SCAN_LENGHT_SYSTEM_3):
        print("System 3") if first_chunk else None
        x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = load.set_system_3()
    elif (shape_ == load.SCAN_LENGHT_SYSTEM_NEW1):
        print("System NEW1") if first_chunk else None
        x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = load.set_system_new1()
    elif (shape_ == load.SCAN_LENGHT_SYSTEM_NEW2):
        print("System NEW2") if first_chunk else None
        x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = load.set_system_new2()
    elif (shape_ == load.SCAN_LENGHT_SYSTEM_NEW3):
        print("System NEW3") if first_chunk else None
        x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = load.set_system_new3()
    elif (shape_ == load.SCAN_LENGHT_SYSTEM_NEW4):
        print("System NEW4") if first_chunk else None
        x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = load.set_system_new4()
    elif (shape_ == load.SCAN_LENGHT_SYSTEM_NEW5):
        print("System NEW5") if first_chunk else None
        x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = load.set_system_new5()
    elif (shape_ == load.SCAN_LENGHT_SYSTEM_NEW6):
        print("System NEW6") if first_chunk else None
        x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = load.set_system_new6()
    else:
        print("System WEIRD for", file_name) if first_chunk else None
        print("Shape:", shape_) if first_chunk else None
        print("System WEIRD for", file_name)
        print("Shape:", shape_)
        return None, None
    return x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds



# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def _vectorised_xy(n, x6, y6, x8, y8, x10, y10):
    """Return x, y for every data_pointer ∈ [0,n)."""
    idx  = np.arange(n, dtype=np.int64)

    cut1 = x6 * y6
    cut2 = cut1 + x8 * y8

    x = np.empty_like(idx,  dtype=np.int32)
    y = np.empty_like(idx,  dtype=np.int32)

    m1 = idx < cut1

    m2 = (idx >= cut1) & (idx < cut2)
    m3 = idx >= cut2

    # region 6 s
    y[m1] = idx[m1] % y6
    x[m1] = idx[m1] // y6

    # region 8 s
    t     = idx[m2] - cut1
    y[m2] = t % y8
    x[m2] = t // y8 + x6

    # region 10 s
    t     = idx[m3] - cut2
    y[m3] = t % y10
    x[m3] = t // y10 + x6 + x8
    return x, y

def _gather_slices(mass_values, intensity_values, starts, counts,
                   mass_offset):
    """Return two Python lists with sliced & offset‑corrected arrays."""
    out_pos, out_val = [], []
    for s, c in zip(starts, counts):
        sl = slice(s, s + c)
        out_pos.append(mass_values[sl] - mass_offset)
        out_val.append(intensity_values[sl])
    return out_pos, out_val

# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------
def prepare_one_data(path, logger: logging.Logger = None):
    start = time.time()
    basename = os.path.basename(path)
    logger.info("[%s] loading…", basename)

    with nc.Dataset(path) as ds:
        # ------- pull arrays once (NetCDF4 gives NumPy masked arrays) ------
        scan_index       = ds["scan_index"][:].compressed()
        point_count      = ds["point_count"][:].compressed()
        mass_values      = ds["mass_values"][:].compressed()
        intensity_values = ds["intensity_values"][:].compressed()
        N = scan_index.size

    logger.info("[%s] read arrays in %.1fs", basename, time.time() - start)

    # ------- coordinate map (pure NumPy) -------------------------------
    x6,y6,x8,y8,x10,y10 = set_system(basename, N, False)
    x, y = _vectorised_xy(N, x6, y6, x8, y8, x10, y10)

    # ------- boolean mask in one shot ----------------------------------
    master_mask = torch.zeros((2000, 460), dtype=torch.bool)
    master_mask[130:2000, 10:400] = True
    master_mask_np = master_mask.numpy()    # share memory, no copy

    inside = (y < master_mask_np.shape[0]) & (x < master_mask_np.shape[1])

    # allocate once, then fill only the safe locations
    hits = np.zeros_like(inside, dtype=bool)
    hits[inside] = master_mask_np[y[inside], x[inside]]

    valid_idx = np.nonzero(hits)[0]            # ≪ N

    # ------- gather only what survives ---------------------------------
    pos, val = _gather_slices(
        mass_values,
        intensity_values,
        scan_index[valid_idx],
        point_count[valid_idx],
        mass_offset=0      # == mass_range_min
    )

    logger.info("[%s] kept %d/%d scans in %.1fs",
             basename, len(valid_idx), N, time.time() - start)

    return pos, val          # <‑‑ your old position_list / value_list



def process_chunk(data_pointer, scan_index, point_count, mass_values, intensity_values, mass_range_min, mask,file_name,shape_,first_chunk=False):
    x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = set_system(file_name,shape_,first_chunk)
    if data_pointer < x_six_seconds * y_six_seconds:
        x = data_pointer // y_six_seconds
        y = data_pointer % y_six_seconds
    elif data_pointer < x_six_seconds * y_six_seconds + x_eight_seconds * y_eight_seconds:
        data_pointer -= x_six_seconds * y_six_seconds
        y = data_pointer % y_eight_seconds
        x = data_pointer // y_eight_seconds + x_six_seconds
    else:
        data_pointer -= x_six_seconds * y_six_seconds + x_eight_seconds * y_eight_seconds
        y = data_pointer % y_ten_seconds
        x = data_pointer // y_ten_seconds + x_six_seconds + x_eight_seconds

    #if x <= 459 and mask[y, x] == 1
    if mask.shape[0] > y and mask.shape[1] > x and mask[y, x] == 1:
        si_s = scan_index[data_pointer]
        si_t = si_s + point_count[data_pointer]
        indexes = np.array(mass_values[si_s:si_t]) - mass_range_min
        return indexes, intensity_values[si_s:si_t]
    else:
        return np.array([]), np.array([])
