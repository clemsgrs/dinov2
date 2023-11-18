import warnings
import numpy as np

from enum import Enum
from mmap import ACCESS_READ, mmap
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import VisionDataset

from .decoders import ImageDataDecoder
from .extended import ExtendedVisionDataset


_Labels = int


class _Fold(Enum):
    FOLD_0 = "0"
    FOLD_1 = "1"
    FOLD_2 = "2"
    FOLD_3 = "3"
    FOLD_4 = "4"

    def entries_name(self):
        return f"pretrain_entries_{self.value}.npy"


class _Split(Enum):
    TRAIN = "train"
    TEST = "test"

    def entries_name(self):
        return f"{self.value}_entries.npy"

    def file_indices_name(self):
        return f"{self.value}_file_indices.npy"

    def tarball_name(self):
        return f"{self.value}_dataset.tar"


def _make_mmap_tarball(tarball_path: str) -> mmap:
    # since we only have one tarball, this function simplifies to mmap that single file
    with open(tarball_path) as f:
        return mmap(fileno=f.fileno(), length=0, access=ACCESS_READ)


class PathologyDataset(VisionDataset):

    Fold = _Fold

    def __init__(
        self,
        *,
        root: str,
        fold: Optional["PathologyDataset.Fold"] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._fold = fold
        self._get_entries()
        self._filepaths = np.load(Path(root, "pretrain_file_indices.npy"), allow_pickle=True).item()
        self._mmap_tarball = _make_mmap_tarball(Path(root, "pretrain_dataset.tar"))

    @property
    def fold(self) -> "PathologyDataset.Fold":
        return self._fold

    @property
    def _entries_name(self) -> str:
        return self._fold.entries_name() if self._fold else "pretrain_entries.npy"

    def _get_entries(self) -> np.ndarray:
        self._entries = self._load_entries(self._entries_name)

    def _load_entries(self, _entries_name: str) -> np.ndarray:
        entries_path = Path(self.root, _entries_name)
        return np.load(entries_path, mmap_mode="r")

    def get_image_data(self, index: int) -> bytes:
        entry = self._entries[index]
        file_idx, start_offset, end_offset = entry[1], entry[2], entry[3]
        filepath = self._filepaths[file_idx]
        mapped_data = self._mmap_tarball[start_offset:end_offset]
        return mapped_data, Path(filepath)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data, img_path = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
            # image.save(f'/data/pathology/projects/ais-cap/clement/code/dinov2/tmp/{img_path.name}')
        except Exception as e:
            raise RuntimeError(f"Cannot read image for sample {index}") from e

        target = ()  # Empty target as per your requirement
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._entries)


class KNNDataset(ExtendedVisionDataset):

    Split = _Split
    Labels = _Labels

    def __init__(
        self,
        *,
        root: str,
        split: Optional["KNNDataset.Split"] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self._get_entries()
        self._get_filepaths()
        self._get_mmap_tarball()
        self._get_class_ids()

    @property
    def split(self) -> "KNNDataset.Split":
        return self._split

    @property
    def _entries_name(self) -> str:
        return self._split.entries_name()

    @property
    def _file_indices_name(self) -> str:
        return self._split.file_indices_name()

    @property
    def _tarball_name(self) -> str:
        return self._split.tarball_name()

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
        return np.load(class_ids_path, allow_pickle=True).item()

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
