import numpy as np

from enum import Enum
from mmap import ACCESS_READ, mmap
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import VisionDataset

from .decoders import ImageDataDecoder

_DEFAULT_MMAP_CACHE_SIZE = 16  # Warning: This can exhaust file descriptors


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"

    @property
    def length(self) -> int:
        return {
            _Split.TRAIN: 11_797_647,
            _Split.VAL: 561_050,
        }[self]

    def entries_path(self):
        return f"imagenet21kp_{self.value}.txt"


def _make_mmap_tarball(tarball_path: str) -> mmap:
    # since we only have one tarball, this function simplifies to mmap that single file
    with open(tarball_path) as f:
        return mmap(fileno=f.fileno(), length=0, access=ACCESS_READ)


class PathologyDataset(VisionDataset):

    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        mmap_cache_size: int = _DEFAULT_MMAP_CACHE_SIZE,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._entries = self._load_entries(Path(root, "entries.npy"))
        self._paths = np.load(Path(root, "file_indices.npy"), allow_pickle=True).item()
        self._mmap_tarball = _make_mmap_tarball(Path(root, "dataset.tar"))

    def _load_entries(self, entries_path: str) -> np.ndarray:
        return np.load(entries_path, mmap_mode="r")

    def get_image_data(self, index: int) -> bytes:
        entry = self._entries[index]
        file_idx, start_offset, end_offset = entry[1], entry[2], entry[3]
        path = self._paths[file_idx]
        mapped_data = self._mmap_tarball[start_offset:end_offset]
        return mapped_data, Path(path)

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
