import h5py as h5
import numpy as np
from tqdm import trange


def main():
    """
    Quick script to normalize coordinates for the specific file and save to H5.
    """
    # File paths
    input_file = "/home/plotnikovgp/baikal/data/baikal_mc2020_multi_split_0924h8s2_tl100_norm_wcluster_fix.h5"
    cluster_centers_file = (
        "/home/plotnikovgp/baikal/data/baikal_mc2020_multi_split_0924_clusters_centers.txt"
    )
    split = "train"

    print(f"Processing events from {split} split")
    print(f"Input file: {input_file}")

    cluster_centers = np.loadtxt(cluster_centers_file)
    print(f"Loaded {len(cluster_centers)} cluster centers")

    with h5.File(input_file, "a") as hfile:
        for split in ["train", "val", "test"]:
            print(f"Processing {split} split")
            cluster_ids = np.array(hfile[f"{split}/cluster_ids/data"])
            mean = np.array(hfile["norm_param/mean"])
            std = np.array(hfile["norm_param/std"])
            ev_starts = np.array(hfile[f"{split}/ev_starts/data"])

            total_size = len(cluster_ids)
            print(f"Total size needed: {total_size} points")

            output_path = f"{split}/muons_prty/individ_coords_norm"

            group_path = "/".join(output_path.split("/")[:-1])
            if group_path not in hfile:
                hfile.create_group(group_path)

            if f"{output_path}/data" in hfile:
                print(f"Dataset {output_path}/data already exists, will be overwritten")
                del hfile[f"{output_path}/data"]

            dataset = hfile.create_dataset(
                f"{output_path}/data", shape=(total_size, 3), dtype=np.float32, compression="gzip"
            )

            # Process events
            total_processed = 0
            for i in trange(len(ev_starts) - 1):
                start, end = ev_starts[i], ev_starts[i + 1]
                xyz = hfile[f"{split}/muons_prty/individ/data"][start:end, 2:5]

                # Apply normalization
                xyz = xyz - cluster_centers[cluster_ids[i]]
                xyz = (xyz - mean[2:5]) / std[2:5]

                # Write to dataset
                dataset[total_processed : total_processed + len(xyz)] = xyz
                total_processed += len(xyz)

        print(f"Processed and saved {total_processed} points for {split} split")


if __name__ == "__main__":
    main()
