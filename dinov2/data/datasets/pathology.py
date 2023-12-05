import numpy as np

from enum import Enum
from mmap import ACCESS_READ, mmap
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import VisionDataset

from .decoders import ImageDataDecoder


class _Fold(Enum):
    FOLD_0 = "0"
    FOLD_1 = "1"
    FOLD_2 = "2"
    FOLD_3 = "3"
    FOLD_4 = "4"

    def entries_name(self):
        return f"pretrain_entries_{self.value}.npy"


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
