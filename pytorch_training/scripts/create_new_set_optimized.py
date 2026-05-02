import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py as h5
import numpy as np
import psutil
from tqdm import tqdm

# Configuration parameters
MIN_STRINGS = 1
MAX_EVENTS = {"train": 2_000_000, "val": 200_000, "test": 200_000}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def monitor_memory(interval=5):
    """Monitor and log memory usage at specified intervals."""
    process = psutil.Process(os.getpid())
    while True:
        mem_info = process.memory_info()
        mem_usage_gb = mem_info.rss / (1024**3)  # Convert to GB
        logger.info(f"Memory usage: {mem_usage_gb:.2f} GB")
        time.sleep(interval)


def create_lookup_map(indices):
    """Create an efficient lookup map from original index to position in filtered array."""
    lookup = np.full(np.max(indices) + 1, -1, dtype=np.int32)
    for pos, idx in enumerate(indices):
        lookup[idx] = pos
    return lookup


def process_split(source_path, target_path, split, chunk_size=1000, compression=None):
    """
    Process a single data split efficiently by using chunking.

    Args:
        source_path: Path to source HDF5 file
        target_path: Path to target HDF5 file
        split: Data split name ('train', 'val', 'test')
        chunk_size: Size of chunks to process at once
        compression: Compression filter to use (None, 'gzip', 'lzf')
    """
    start_time = time.time()
    logger.info(f"Processing {split} split...")

    with h5.File(source_path, "r") as src, h5.File(target_path, "a") as dst:
        # Get num_un_strings and filter indices
        num_un_strings_path = f"{split}/num_un_strings/data"
        ev_starts_path = f"{split}/ev_starts/data"

        # Read data in one go to avoid multiple disk accesses
        num_un_strings = np.array(src[num_un_strings_path])
        ev_starts = np.array(src[ev_starts_path])

        # Filter indices efficiently using numpy operations
        indices = np.where(num_un_strings >= MIN_STRINGS)[0]
        indices = indices[indices < MAX_EVENTS[split]]
        indices = np.sort(indices)

        # Create efficient lookup map
        lookup_map = create_lookup_map(indices)

        logger.info(f"Found {len(indices)} valid events for {split} split")

        # Pre-compute adjusted event starts
        event_sizes = ev_starts[indices + 1] - ev_starts[indices]
        adjusted_ev_starts = np.zeros(len(indices) + 1, dtype=np.int32)
        np.cumsum(event_sizes, out=adjusted_ev_starts[1:])

        # Create dataset for adjusted event starts
        comp_opts = {"compression": compression} if compression else {}
        dst.create_dataset(f"{split}/ev_starts/data", data=adjusted_ev_starts, **comp_opts)

        # Process other datasets
        keys_to_slice = ["data", "labels", "t_res"]
        other_keys = ["prime_prty", "ev_ids", "num_un_strings"]

        # First handle datasets that don't need event-based slicing
        for name in other_keys:
            dataset_path = f"{split}/{name}/data"
            if dataset_path in src:
                logger.info(f"Copying {dataset_path}")
                data = src[dataset_path][indices]
                dst.create_dataset(dataset_path, data=data, **comp_opts)

        # Handle datasets that need event-based slicing with chunking for memory efficiency
        for name in keys_to_slice:
            dataset_path = f"{split}/{name}/data"
            if dataset_path not in src:
                logger.warning(f"Dataset {dataset_path} not found in source file")
                continue

            logger.info(f"Processing {dataset_path} with chunking")

            # Get the total size for pre-allocation
            total_size = adjusted_ev_starts[-1]

            # Get the datatype and shape from source dataset
            src_dataset = src[dataset_path]
            dtype = src_dataset.dtype
            shape = src_dataset.shape[1:] if len(src_dataset.shape) > 1 else ()

            # Create the target dataset with appropriate compression
            if shape:
                dst_shape = (total_size,) + shape
            else:
                dst_shape = (total_size,)

            # Add appropriate chunking for compressed datasets
            if compression:
                chunk_opts = calculate_optimal_chunks(dst_shape, dtype)
                comp_opts.update(chunk_opts)

            dst_dataset = dst.create_dataset(
                dataset_path, shape=dst_shape, dtype=dtype, **comp_opts
            )

            # Process in chunks to save memory
            chunk_indices = [
                indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)
            ]

            with tqdm(total=len(indices), desc=f"Processing {name}", unit="events") as pbar:
                for chunk_idx in chunk_indices:
                    # Process this chunk of indices
                    for idx in chunk_idx:
                        start_src = ev_starts[idx]
                        end_src = ev_starts[idx + 1]

                        # Calculate position in target dataset using the lookup map
                        idx_in_filtered = lookup_map[idx]
                        start_dst = adjusted_ev_starts[idx_in_filtered]
                        end_dst = start_dst + (end_src - start_src)

                        # Copy data
                        dst_dataset[start_dst:end_dst] = src_dataset[start_src:end_src]

                    # Update progress bar
                    pbar.update(len(chunk_idx))

                    # Report memory usage periodically
                    if pbar.n % (chunk_size * 10) == 0:
                        mem_info = psutil.Process(os.getpid()).memory_info()
                        mem_usage_gb = mem_info.rss / (1024**3)
                        logger.debug(f"Memory usage: {mem_usage_gb:.2f} GB")

    elapsed_time = time.time() - start_time
    logger.info(f"Completed {split} split in {elapsed_time:.2f} seconds")


def calculate_optimal_chunks(shape, dtype):
    """Calculate optimal chunk size for HDF5 dataset based on shape and dtype."""
    # Target chunk size ~1MB
    target_chunk_bytes = 1 * 1024 * 1024
    item_size = np.dtype(dtype).itemsize

    if len(shape) == 1:
        # 1D dataset
        chunk_size = min(shape[0], max(1, target_chunk_bytes // item_size))
        return {"chunks": (int(chunk_size),)}
    elif len(shape) == 2:
        # 2D dataset - try to keep rows intact
        rows = min(shape[0], max(1, target_chunk_bytes // (item_size * shape[1])))
        return {"chunks": (int(rows), shape[1])}
    else:
        # Higher dimensional - just use auto chunking
        return {"chunks": True}


def copy_filtered_dataset(
    source_path, target_path, parallel=False, max_workers=None, compression=None, chunk_size=1000
):
    """
    Copy and filter datasets from source to target HDF5 file.

    Args:
        source_path: Path to source HDF5 file
        target_path: Path to target HDF5 file
        parallel: Whether to process splits in parallel
        max_workers: Maximum number of worker processes (None = auto)
        compression: Compression filter to use (None, 'gzip', 'lzf')
        chunk_size: Number of events to process in each chunk
    """
    start_time = time.time()

    # Ensure the parent directory exists
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)

    # Create target file if it doesn't exist yet
    if os.path.exists(target_path):
        logger.warning(f"Target file {target_path} exists, removing it")
        os.remove(target_path)

    # Create target file with proper data structure
    with h5.File(target_path, "w"):
        pass

    splits = ["train", "val", "test"]

    if parallel and len(splits) > 1:
        # Process splits in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_split, source_path, target_path, split, chunk_size, compression
                ): split
                for split in splits
            }

            for future in as_completed(futures):
                split = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing {split} split: {str(e)}")
    else:
        # Process splits sequentially
        for split in splits:
            try:
                process_split(source_path, target_path, split, chunk_size, compression)
            except Exception as e:
                logger.error(f"Error processing {split} split: {str(e)}")
                logger.exception(e)

    # Verify file sizes
    src_size = os.path.getsize(source_path) / (1024 * 1024)  # MB
    dst_size = os.path.getsize(target_path) / (1024 * 1024)  # MB

    logger.info(f"Source file size: {src_size:.2f} MB")
    logger.info(f"Target file size: {dst_size:.2f} MB")
    logger.info(f"Size ratio: {dst_size / src_size:.2%}")

    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficiently filter and copy HDF5 datasets")
    parser.add_argument(
        "--source",
        type=str,
        default="/home2/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5",
        help="Source HDF5 file path",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="/home/plotnikovgp/baikal/data/baikal_2020_sig-noise_mid-eq_normed_2Mevs_opt.h5",
        help="Target HDF5 file path",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process splits sequentially instead of in parallel",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel execution",
    )
    parser.add_argument(
        "--min-strings", type=int, default=1, help="Minimum number of strings required"
    )
    parser.add_argument(
        "--compression", choices=["gzip", "lzf"], default=None, help="Compression filter to use"
    )
    parser.add_argument(
        "--compression-level", type=int, default=4, help="Compression level (1-9, only for gzip)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Number of events to process in each chunk"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Update global variables based on args
    MIN_STRINGS = args.min_strings

    # Configure compression options
    compression_opts = None
    if args.compression == "gzip":
        compression_opts = {"compression": "gzip", "compression_opts": args.compression_level}
    elif args.compression == "lzf":
        compression_opts = {"compression": "lzf"}

    logger.info(f"Starting dataset creation with min_strings={MIN_STRINGS}")
    logger.info(f"Source: {args.source}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Parallel execution: {not args.sequential}")
    if compression_opts:
        logger.info(f"Using compression: {args.compression}")

    try:
        copy_filtered_dataset(
            args.source,
            args.target,
            parallel=not args.sequential,
            max_workers=args.workers,
            compression=args.compression,
            chunk_size=args.chunk_size,
        )
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during dataset creation: {str(e)}")
        logger.exception(e)
        sys.exit(1)
