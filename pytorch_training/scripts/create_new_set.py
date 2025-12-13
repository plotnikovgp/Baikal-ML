import os

import h5py as h5
import numpy as np
from scipy.spatial.distance import pdist
from tqdm import tqdm

MIN_HITS = 8
MIN_STRINGS = 2
# can read
# MEAN = np.array([7.4427495, -36.435776, 1.3464843,-0.39018127, 25.14138])
# STD = np.array([17.786774, 380.94046, 39.317795, 38.54446, 144.40602])
MIN_TRACK_LENGTH = 100
MAX_EVENTS = {"train": 2_000_000, "val": 200_000, "test": 200_000}


def calc_track_length(coords, is_track_mask):
    if is_track_mask.sum() < 2:
        return 0
    coords = coords[is_track_mask]
    distances = pdist(coords, "euclidean")
    return np.max(distances)


def copy_filtered_dataset(source_path, target_path):
    with h5.File(source_path, "r") as src, h5.File(target_path, "a") as dst:
        MEAN = np.array(src["norm_param/mean"])
        STD = np.array(src["norm_param/std"])
        splits = ["test", "val", "train"]
        for split in splits:
            print(f"Processing {split} split")
            indices = np.arange(len(src[f"{split}/ev_starts/data"]))
            appropriate_indices = [i for i in indices if i < MAX_EVENTS[split]]
            ev_starts_path = f"{split}/ev_starts/data"
            ev_starts = np.array(src[ev_starts_path])
            for i in tqdm(indices):
                if ev_starts[i + 1] - ev_starts[i] > MIN_HITS:
                    data = np.array(src[f"{split}/data/data"][ev_starts[i] : ev_starts[i + 1]])[:]
                    coords_meters = (data * STD + MEAN)[:, 2:]
                    labels = np.array(src[f"{split}/labels/data"][ev_starts[i] : ev_starts[i + 1]])
                    is_track_hit = labels < 0
                    track_length = calc_track_length(coords_meters, is_track_hit)
                    if track_length > MIN_TRACK_LENGTH:
                        appropriate_indices.append(i)

            indices = np.array(sorted(list(set(appropriate_indices))), dtype=np.int32)

            adjusted_ev_starts = [0]
            for ind in indices:
                adjusted_ev_starts.append(
                    adjusted_ev_starts[-1] + ev_starts[ind + 1] - ev_starts[ind]
                )
            adjusted_ev_starts = np.array(adjusted_ev_starts)
            # Copy the adjusted ev_starts first
            dst.create_dataset(f"{split}/ev_starts/data", data=adjusted_ev_starts)

            keys_to_slice = ["data", "labels", "t_res"]
            other_keys = [
                "prime_prty",
                "ev_ids",
                "num_un_strings",
            ]  # 'muons_prty/individ',
            # track_events = np.concatenate([data[ev_starts[ind]:ev_starts[ind + 1]] for ind in indices])
            # Iterate and copy all keys, applying slicing to necessary ones
            for name in keys_to_slice + other_keys:
                print(f"Copying {name}")
                sub_dataset_path = f"{split}/{name}/data"
                data = np.array(src[sub_dataset_path])

                if name in keys_to_slice:
                    # Filter data based on valid event starts
                    filtered_data = np.concatenate(
                        [data[ev_starts[ind] : ev_starts[ind + 1]] for ind in tqdm(indices)]
                    )
                else:
                    filtered_data = data[indices]  # For data not requiring slicing based on events
                dst.create_dataset(sub_dataset_path, data=filtered_data)


source_path = "/home2/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5"
target_path = "/home/plotnikovgp/baikal/data/baikal_2020_sig-noise_mid-eq_normed_2Mevs.h5"

if os.path.exists(target_path):
    os.remove(target_path)
# Function call to perform the operation
copy_filtered_dataset(source_path, target_path)
