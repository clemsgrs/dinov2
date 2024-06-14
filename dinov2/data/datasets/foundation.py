from functools import lru_cache

from mmap import ACCESS_READ, mmap
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import VisionDataset
from pathlib import Path
import numpy as np

from .decoders import ImageDataDecoder, TargetDecoder


_DEFAULT_MMAP_CACHE_SIZE = 16  # Warning: This can exhaust file descriptors


class _Subset:
    def __init__(self, value):
        self.value = value

    def entries_name(self):
        return f"pretrain_entries_{self.value}.npy"


def _get_tarball_path(cohort_name: str) -> str:
    return f"{cohort_name}.tar"


def _make_mmap_tarball(tarballs_root: str, mmap_cache_size: int):
    @lru_cache(maxsize=mmap_cache_size)
    def _mmap_tarball(cohort_name: str) -> mmap:
        tarball_path = _get_tarball_path(cohort_name)
        tarball_full_path = Path(tarballs_root, tarball_path)
        with open(tarball_full_path) as f:
            return mmap(fileno=f.fileno(), length=0, access=ACCESS_READ)

    return _mmap_tarball


class PathologyFoundationDataset(VisionDataset):
    Subset = _Subset

    def __init__(
        self,
        *,
        root: str,
        subset: Optional["PathologyFoundationDataset.Subset"] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        mmap_cache_size: int = _DEFAULT_MMAP_CACHE_SIZE,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._subset = subset
        self._get_entries()
        self._get_cohort_names()
        self._mmap_tarball = _make_mmap_tarball(self._tarballs_root, mmap_cache_size)

    @property
    def _tarballs_root(self) -> str:
        return self.root

    @property
    def subset(self) -> "PathologyFoundationDataset.Subset":
        return self._subset

    @property
    def _entries_name(self) -> str:
        return self._subset.entries_name() if self._subset else "pretrain_entries.npy"

    def _get_entries(self) -> np.ndarray:
        self._entries = self._load_entries(self._entries_name)

    def _load_entries(self, _entries_name: str) -> np.ndarray:
        entries_path = Path(self.root, _entries_name)
        return np.load(entries_path, mmap_mode="r")

    def _get_cohort_names(self) -> dict:
        self._cohort_names = self._load_cohort_names()

    def _load_cohort_names(self) -> dict:
        cohort_dict_path = Path(self.root, "cohort_indices.npy")
        return np.load(cohort_dict_path, allow_pickle=True).item()

    def get_image_data(self, index: int) -> bytes:
        entry = self._entries[index]
        start_offset, end_offset, cohort_idx = entry[1], entry[2], entry[3], entry[4]
        cohort_name = self._cohort_names[cohort_idx]
        class_mmap = self._mmap_tarball(cohort_name)
        data = class_mmap[start_offset:end_offset]
        return data

    def get_target(self, index: int) -> Any:
        return int(self._entries[index][0])

    def get_targets(self) -> np.ndarray:
        return self._entries[:, 0]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index} ({e})") from e
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._entries)
