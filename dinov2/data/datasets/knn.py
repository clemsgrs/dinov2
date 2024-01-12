import warnings
import numpy as np

from enum import Enum
from mmap import ACCESS_READ, mmap
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from .extended import ExtendedVisionDataset


_Labels = int


class _Subset:
    def __init__(self, value):
        self.value = value

    def entries_name(self, split):
        return f"{split.value}_entries_{self.value}.npy"


class _Split(Enum):
    query = "query"
    test = "test"

    def entries_name(self):
        return f"{self.value}_entries.npy"

    def file_indices_name(self):
        return f"{self.value}_file_indices.npy"


def _make_mmap_tarball(tarball_path: str) -> mmap:
    # since we only have one tarball, this function simplifies to mmap that single file
    with open(tarball_path) as f:
        return mmap(fileno=f.fileno(), length=0, access=ACCESS_READ)


class KNNDataset(ExtendedVisionDataset):
    Subset = _Subset
    Split = _Split
    Labels = _Labels

    def __init__(
        self,
        *,
        root: str,
        subset: Optional["KNNDataset.Subset"] = None,
        split: Optional["KNNDataset.Split"] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._subset = subset
        self._split = split
        self._get_entries()
        self._get_filepaths()
        self._get_mmap_tarball()
        self._get_class_ids()

    @property
    def subset(self) -> "KNNDataset.Subset":
        return self._subset

    @property
    def split(self) -> "KNNDataset.Split":
        return self._split

    @property
    def _entries_name(self) -> str:
        return self._subset.entries_name(self._split)

    @property
    def _file_indices_name(self) -> str:
        return self._split.file_indices_name()

    @property
    def _tarball_name(self) -> str:
        return "downstream_dataset.tar"

    @property
    def _class_ids_name(self) -> str:
        return "class-ids.npy"

    def _get_entries(self) -> np.ndarray:
        self._entries = self._load_entries(self._entries_name)

    def _load_entries(self, _entries_name: str) -> np.ndarray:
        entries_path = Path(self.root, _entries_name)
        return np.load(entries_path, mmap_mode="r")

    def _get_filepaths(self) -> np.ndarray:
        self._filepaths = self._load_filepaths(self._file_indices_name)

    def _load_filepaths(self, _file_indices_name: str) -> np.ndarray:
        file_indices_path = Path(self.root, _file_indices_name)
        return np.load(file_indices_path, allow_pickle=True).item()

    def _get_mmap_tarball(self) -> mmap:
        self._mmap_tarball = self._load_tarball(self._tarball_name)

    def _load_tarball(self, _tarball_name: str) -> mmap:
        tarball_path = Path(self.root, _tarball_name)
        return _make_mmap_tarball(tarball_path)

    def _get_class_ids(self):
        self._class_ids = self._load_class_ids(self._class_ids_name)

    def _load_class_ids(self, _class_ids_name: str) -> np.ndarray:
        class_ids_path = Path(self.root, _class_ids_name)
        return np.load(class_ids_path, allow_pickle=True)

    def get_image_data(self, index: int) -> bytes:
        entry = self._entries[index]
        start_offset, end_offset = entry[2], entry[3]
        mapped_data = self._mmap_tarball[start_offset:end_offset]
        return mapped_data

    def get_target(self, index: int) -> Any:
        return int(self._entries[index][0])

    def get_targets(self) -> np.ndarray:
        # return [entry[0] for entry in self._entries]
        return self._entries[0]

    def get_class_name(self, index: int) -> str:
        entry = self._entries[index]
        class_idx = entry[0]
        return self._class_ids[class_idx]

    def get_class_names(self) -> np.ndarray:
        # return [self._class_ids[entry[0]] for entry in self._entries]
        return self._class_ids[self._entries[0]]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return super().__getitem__(index)

    def __len__(self) -> int:
        return len(self._entries)
