import random
import typing as tp

from torch.utils.data import DataLoader, IterableDataset, Subset
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.loader import DataLoader as GraphDataLoader

from .readers import BaikalDataset, Dataset

SPLIT_TYPES = ["train", "val", "test"]


def create_datasets(
    path_to_data: str,
    DatasetType: tp.Type[Dataset] = BaikalDataset,
    batch_size: int = 128,
    is_graph: bool = False,
    val_subset_cut: int = 3,
    val_path: str | None = None,
    test_path: str | None = None,
    val_renorm_params: tuple | None = None,
    test_renorm_params: tuple | None = None,
    use_val_subset: bool = True,
    **kwargs,
):
    datasets = {}
    for split_type in SPLIT_TYPES:
        split_path = path_to_data
        split_kwargs = dict(kwargs)
        if split_type == "val" and val_path is not None:
            split_path = val_path
            split_kwargs["renorm_params"] = val_renorm_params
        elif split_type == "test" and test_path is not None:
            split_path = test_path
            split_kwargs["renorm_params"] = test_renorm_params
        datasets[split_type] = DatasetType(
            split_path, split_type, batch_size=batch_size, is_graph=is_graph, **split_kwargs
        )
    if use_val_subset and val_subset_cut > 1:
        datasets["val_subset"] = Subset(
            datasets["val"], list(range(0, len(datasets["val"]), val_subset_cut))
        )
    return datasets


def infinite_loader(loader: tp.Iterable):
    while True:
        for batch in loader:
            yield batch


class MultiDatasetSampler(IterableDataset):
    def __init__(self, datasets: list, probabilities: list = None, seed: int = 42):
        self.datasets = datasets
        self.probabilities = probabilities or [1.0 / len(datasets)] * len(datasets)
        total = sum(self.probabilities)
        self.probabilities = [p / total for p in self.probabilities]
        self.rng = random.Random(seed)
        self._iterators = None

    def __iter__(self):
        self._iterators = [iter(ds) for ds in self.datasets]

        while True:
            idx = self.rng.choices(range(len(self.datasets)), weights=self.probabilities, k=1)[0]
            try:
                sample = next(self._iterators[idx])
            except StopIteration:
                self._iterators[idx] = iter(self.datasets[idx])
                sample = next(self._iterators[idx])

            if isinstance(sample, tuple):
                yield sample + (idx,)
            else:
                yield (sample, idx)


def _get_dataloader_kwargs(
    num_workers: int, pin_memory: bool, prefetch_factor: int, persistent_workers: bool
):
    kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        if persistent_workers:
            kwargs["persistent_workers"] = True
        if prefetch_factor > 2:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def create_dataloaders(
    path_to_data: str,
    DatasetType: tp.Type[Dataset] = BaikalDataset,
    batch_size: int = 128,
    is_graph: bool = False,
    val_subset_cut: int = 3,
    val_path: str | None = None,
    test_path: str | None = None,
    val_renorm_params: tuple | None = None,
    test_renorm_params: tuple | None = None,
    num_workers: int = 1,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    **kwargs,
):
    datasets = create_datasets(
        path_to_data=path_to_data,
        DatasetType=DatasetType,
        batch_size=batch_size,
        is_graph=is_graph,
        val_subset_cut=val_subset_cut,
        val_path=val_path,
        test_path=test_path,
        val_renorm_params=val_renorm_params,
        test_renorm_params=test_renorm_params,
        **kwargs,
    )

    train_ds = datasets["train"]
    val_ds = datasets.get("val_subset", datasets["val"])
    test_ds = datasets["test"]

    dl_kwargs = _get_dataloader_kwargs(num_workers, pin_memory, prefetch_factor, persistent_workers)
    LoaderClass = GraphDataLoader if is_graph else DataLoader
    batch_arg = {"batch_size": batch_size} if is_graph else {"batch_size": None}

    train_loader = infinite_loader(LoaderClass(train_ds, shuffle=False, **batch_arg, **dl_kwargs))
    val_loader = LoaderClass(val_ds, **batch_arg, **dl_kwargs)
    test_loader = LoaderClass(test_ds, **batch_arg, **dl_kwargs)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
    }


def create_multi_dataset_dataloader(
    dataset_configs: list,
    probabilities: list = None,
    batch_size: int = 128,
    num_workers: int = 1,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    return_datasets: bool = False,
    **kwargs,
):
    train_datasets, val_datasets, test_datasets = [], [], []
    is_graph = any(cfg.get("is_graph", False) for cfg in dataset_configs)

    for config in dataset_configs:
        datasets = create_datasets(
            path_to_data=config["path_to_data"],
            DatasetType=config.get("DatasetType", BaikalDataset),
            batch_size=batch_size,
            is_graph=config.get("is_graph", False),
            val_subset_cut=config.get("val_subset_cut", 3),
            val_path=config.get("val_path"),
            test_path=config.get("test_path"),
            val_renorm_params=config.get("val_renorm_params"),
            test_renorm_params=config.get("test_renorm_params"),
            preprocessor=config.get("preprocessor"),
            **kwargs,
        )
        train_datasets.append(datasets["train"])
        val_datasets.append(datasets.get("val_subset", datasets["val"]))
        test_datasets.append(datasets["test"])

    dl_kwargs = _get_dataloader_kwargs(num_workers, pin_memory, prefetch_factor, persistent_workers)
    LoaderClass = GraphDataLoader if is_graph else DataLoader
    batch_arg = {"batch_size": batch_size} if is_graph else {"batch_size": None}

    train_sampler = MultiDatasetSampler(train_datasets, probabilities)
    train_loader = infinite_loader(
        LoaderClass(train_sampler, shuffle=False, **batch_arg, **dl_kwargs)
    )

    val_loaders = [LoaderClass(ds, **batch_arg, **dl_kwargs) for ds in val_datasets]
    test_loaders = [LoaderClass(ds, **batch_arg, **dl_kwargs) for ds in test_datasets]

    result = {"train": train_loader, "val": val_loaders, "test": test_loaders}

    if return_datasets:
        result["train_datasets"] = train_datasets
        result["val_datasets"] = val_datasets
        result["test_datasets"] = test_datasets

    return result
