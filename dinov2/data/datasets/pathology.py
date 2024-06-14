import numpy as np

from mmap import ACCESS_READ, mmap
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import VisionDataset

from .decoders import ImageDataDecoder


class _Subset:
    def __init__(self, value):
        self.value = value

    def entries_name(self):
        return f"pretrain_entries_{self.value}.npy"


def _make_mmap_tarball(tarball_path: str) -> mmap:
    # since we only have one tarball, this function simplifies to mmap that single file
    with open(tarball_path) as f:
        return mmap(fileno=f.fileno(), length=0, access=ACCESS_READ)


class PathologyDataset(VisionDataset):
    Subset = _Subset

    def __init__(
        self,
        *,
        root: str,
        extra: str,
        subset: Optional["PathologyDataset.Subset"] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.extra = extra
        self._subset = subset
        self._get_entries()
        self._mmap_tarball = _make_mmap_tarball(Path(root, "pretrain_dataset.tar"))

    @property
    def subset(self) -> "PathologyDataset.Subset":
        return self._subset

    @property
    def _entries_name(self) -> str:
        return self._subset.entries_name() if self._subset else "pretrain_entries.npy"

    def _get_entries(self) -> np.ndarray:
        self._entries = self._load_entries(self._entries_name)

    def _load_entries(self, _entries_name: str) -> np.ndarray:
        entries_path = Path(self.extra, _entries_name)
        return np.load(entries_path, mmap_mode="r")

    def get_image_data(self, index: int) -> bytes:
        entry = self._entries[index]
        start_offset, end_offset = entry[1], entry[2], entry[3]
        mapped_data = self._mmap_tarball[start_offset:end_offset]
        return mapped_data

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"Cannot read image for sample {index}") from e

        target = ()  # Empty target as per your requirement
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._entries)
