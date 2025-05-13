from torch.nn.utils.rnn import pad_sequence
from .readers import BaikalDataset, Dataset
from torch.utils.data import Dataset, Subset, DataLoader, IterableDataset
from torch_geometric.loader import DataLoader as GraphDataLoader
import typing as tp
import logging
import torch
import numpy as np
import random

SPLIT_TYPES = ["train", "val", "test"]


def create_datasets(
    path_to_data: str,
    use_val_subset: bool = True,
    DatasetType: tp.Type[Dataset] = BaikalDataset,
    batch_size: int = 128,
    is_graph: bool = False,
    val_subset_cut: int = 3,
    **kwargs,
):
    datasets = {}
    for split_type in SPLIT_TYPES:
        datasets[split_type] = DatasetType(
            path_to_data, split_type, batch_size=batch_size, is_graph=is_graph, **kwargs
        )
    if use_val_subset:
        datasets["val_subset"] = Subset(
            datasets["val"], list(range(0, len(datasets["val"]), val_subset_cut))
        )
    return datasets


def create_infnite_loader_generator(loader: tp.Iterable, buffer_size=2):
    buffer = []
    loader_iter = iter(loader)

    while True:
        while len(buffer) < buffer_size:
            try:
                item = next(loader_iter)
                buffer.append(item)
            except StopIteration:
                loader_iter = iter(loader)
                if not buffer:
                    buffer.append(next(loader_iter))
                break

        if buffer:
            yield buffer.pop(0)


class MultiDatasetSampler(IterableDataset):
    def __init__(
        self,
        datasets: list,
        probabilities: list | None = None,
        seed: int = 42,
        prefetch_size: int = 2,
    ):
        self.datasets = datasets
        if probabilities is None:
            self.probabilities = [1.0 / len(datasets)] * len(datasets)
        else:
            total = sum(probabilities)
            self.probabilities = [p / total for p in probabilities]

        self.random_gen = random.Random(seed)
        self.prefetch_size = prefetch_size
        self._prefetch_buffers = [[] for _ in datasets]
        self._iterators = None  # Will be initialized lazily

    def _get_iterator(self, dataset_idx):
        """Get or create an iterator for the specified dataset."""
        if self._iterators is None:
            self._iterators = [None] * len(self.datasets)

        if self._iterators[dataset_idx] is None:
            self._iterators[dataset_idx] = iter(self.datasets[dataset_idx])

        return self._iterators[dataset_idx]

    def _prefetch_from_dataset(self, dataset_idx):
        """Prefetch items from a dataset to fill the buffer."""
        buffer = self._prefetch_buffers[dataset_idx]
        iterator = self._get_iterator(dataset_idx)

        try:
            while len(buffer) < self.prefetch_size:
                buffer.append(next(iterator))
        except StopIteration:
            self._iterators[dataset_idx] = iter(self.datasets[dataset_idx])

            if not buffer:
                iterator = self._iterators[dataset_idx]
                buffer.append(next(iterator))

    def _get_sample_from_dataset(self, dataset_idx):
        """Get a sample from the specified dataset."""
        buffer = self._prefetch_buffers[dataset_idx]

        if not buffer:
            self._prefetch_from_dataset(dataset_idx)
        return buffer.pop(0)

    def __iter__(self):
        if self._iterators is None:
            self._iterators = [None] * len(self.datasets)
            self._prefetch_buffers = [[] for _ in self.datasets]

            for i in range(len(self.datasets)):
                self._prefetch_from_dataset(i)

        while True:
            dataset_idx = self.random_gen.choices(
                range(len(self.datasets)), weights=self.probabilities, k=1
            )[0]
            # print(f"Sampling from dataset {dataset_idx}")
            # a = input()
            try:
                sample = self._get_sample_from_dataset(dataset_idx)
                if isinstance(sample, tuple):
                    sample = sample + (dataset_idx,)
                else:
                    sample = (sample, dataset_idx)

                yield sample

                if len(self._prefetch_buffers[dataset_idx]) < self.prefetch_size:
                    self._prefetch_from_dataset(dataset_idx)

            except Exception as e:
                logging.error(f"Error sampling from dataset {dataset_idx}: {e}")
                raise e


def create_multi_dataset_dataloader(
    dataset_configs: list,
    probabilities: list = None,
    batch_size: int = 128,
    num_workers: int = 1,
    return_datasets: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    cache_datasets: bool = False,
    **kwargs,
):
    """
    Create a dataloader that samples from multiple datasets with specified probabilities.
    Optimized for performance with improved worker utilization.

    Args:
        dataset_configs (list): List of dataset configurations. Each config should have:
                                - path_to_data: path to the dataset
                                - DatasetType: type of dataset to create
                                - is_graph: whether the dataset is a graph dataset
                                - preprocessor: dataset-specific preprocessor
                                - Additional kwargs specific to the dataset
        probabilities (list, optional): Sampling probabilities for each dataset.
                                      If None, datasets will be sampled uniformly.
        batch_size (int, optional): Batch size for the dataloaders.
        num_workers (int, optional): Number of workers for the dataloaders.
        return_datasets (bool, optional): Whether to return the datasets in the result.
        set_tres_stats (bool, optional): Global flag for setting t_res statistics.
                                        Can be overridden by individual dataset configs.
        prefetch_factor (int, optional): Number of batches to prefetch per worker.
        persistent_workers (bool, optional): Keep worker processes alive after dataset exhaustion.
        pin_memory (bool, optional): Pin memory for faster GPU transfer.
        cache_datasets (bool, optional): Whether to cache datasets in memory (speeds up training
                                        at the cost of memory usage).
        **kwargs: Additional arguments to pass to all datasets.

    Returns:
        dict: Dictionary with train, val, and test dataloaders, and optionally the datasets.
    """
    train_datasets = []
    val_datasets = []
    test_datasets = []

    # Create datasets for each config
    for config in dataset_configs:
        path_to_data = config["path_to_data"]
        DatasetType = config.get("DatasetType", BaikalDataset)
        is_graph = config.get("is_graph", False)
        use_val_subset = config.get("use_val_subset", True)
        val_subset_cut = config.get("val_subset_cut", 3)

        preprocessor = config.get("preprocessor")

        datasets = create_datasets(
            path_to_data=path_to_data,
            use_val_subset=use_val_subset,
            DatasetType=DatasetType,
            batch_size=batch_size,
            is_graph=is_graph,
            val_subset_cut=val_subset_cut,
            preprocessor=preprocessor,
            **kwargs,
        )

        if cache_datasets:
            train_ds = CachingDatasetWrapper(datasets["train"])
            if use_val_subset:
                val_ds = CachingDatasetWrapper(datasets["val_subset"])
            else:
                val_ds = CachingDatasetWrapper(datasets["val"])
            test_ds = CachingDatasetWrapper(datasets["test"])

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)
        else:
            train_datasets.append(datasets["train"])
            if use_val_subset:
                val_datasets.append(datasets["val_subset"])
            else:
                val_datasets.append(datasets["val"])
            test_datasets.append(datasets["test"])

    train_sampler = MultiDatasetSampler(
        train_datasets, probabilities, prefetch_size=prefetch_factor
    )
    using_graph_data = any(config.get("is_graph", False) for config in dataset_configs)

    dataloader_common_args = {"num_workers": num_workers, "pin_memory": pin_memory}

    if persistent_workers and num_workers > 0:
        dataloader_common_args["persistent_workers"] = True

    if prefetch_factor > 2 and num_workers > 0:
        dataloader_common_args["prefetch_factor"] = prefetch_factor

    if not using_graph_data:
        train_loader = create_infnite_loader_generator(
            DataLoader(
                train_sampler, batch_size=None, shuffle=False, **dataloader_common_args
            ),
            buffer_size=prefetch_factor,
        )

        val_loaders = [
            DataLoader(dataset, batch_size=None, **dataloader_common_args)
            for dataset in val_datasets
        ]

        test_loaders = [
            DataLoader(dataset, batch_size=None, **dataloader_common_args)
            for dataset in test_datasets
        ]
    else:
        train_loader = create_infnite_loader_generator(
            GraphDataLoader(
                train_sampler, batch_size=batch_size, **dataloader_common_args
            ),
            buffer_size=prefetch_factor,
        )

        val_loaders = [
            GraphDataLoader(dataset, batch_size=batch_size, **dataloader_common_args)
            for dataset in val_datasets
        ]

        test_loaders = [
            GraphDataLoader(dataset, batch_size=batch_size, **dataloader_common_args)
            for dataset in test_datasets
        ]

    res = {"train": train_loader, "val": val_loaders, "test": test_loaders}

    if return_datasets:
        res["train_datasets"] = train_datasets
        res["val_datasets"] = val_datasets
        res["test_datasets"] = test_datasets

    return res


class CachingDatasetWrapper(Dataset):
    """
    A wrapper for datasets that caches items in memory for faster access.

    Args:
        dataset: The dataset to wrap
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]


def create_dataloaders(
    path_to_data: str,
    batch_size: int = 128,
    is_graph: bool = False,
    use_val_subset: bool = True,
    num_workers: int = 1,
    DatasetType: tp.Type[Dataset] = BaikalDataset,
    return_datasets: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    cache_datasets: bool = False,
    **kwargs,
):
    datasets = create_datasets(
        path_to_data,
        use_val_subset,
        DatasetType,
        is_graph=is_graph,
        batch_size=batch_size,
        **kwargs,
    )

    if cache_datasets:
        train_dataset = CachingDatasetWrapper(datasets["train"])
        val_dataset = CachingDatasetWrapper(
            datasets["val_subset" if use_val_subset else "val"]
        )
        test_dataset = CachingDatasetWrapper(datasets["test"])
    else:
        train_dataset = datasets["train"]
        val_dataset = datasets["val_subset" if use_val_subset else "val"]
        test_dataset = datasets["test"]

    dataloader_common_args = {"num_workers": num_workers, "pin_memory": pin_memory}

    if persistent_workers and num_workers > 0:
        dataloader_common_args["persistent_workers"] = True

    if prefetch_factor > 2 and num_workers > 0:
        dataloader_common_args["prefetch_factor"] = prefetch_factor

    if not is_graph:
        train_loader = create_infnite_loader_generator(
            DataLoader(
                train_dataset, batch_size=None, shuffle=False, **dataloader_common_args
            ),
            buffer_size=prefetch_factor,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=None,
            **dataloader_common_args,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, **dataloader_common_args
        )
    else:
        train_loader = create_infnite_loader_generator(
            GraphDataLoader(
                train_dataset, batch_size=batch_size, **dataloader_common_args
            ),
            buffer_size=prefetch_factor,
        )

        test_loader = GraphDataLoader(
            test_dataset, batch_size, **dataloader_common_args
        )

        val_loader = GraphDataLoader(val_dataset, batch_size, **dataloader_common_args)

    res = {"train": train_loader, "val": val_loader, "test": test_loader}
    if return_datasets:
        res["train_dataset"] = train_dataset
        res["val_dataset"] = val_dataset
        res["test_dataset"] = test_dataset
    return res
