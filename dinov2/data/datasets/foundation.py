from enum import Enum
from functools import lru_cache

# from gzip import GzipFile
# from io import BytesIO
from mmap import ACCESS_READ, mmap
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import VisionDataset
from pathlib import Path
import numpy as np

from .decoders import ImageDataDecoder, TargetDecoder


_DEFAULT_MMAP_CACHE_SIZE = 16  # Warning: This can exhaust file descriptors


class _Fold(Enum):
    FOLD_0 = "0"
    FOLD_1 = "1"
    FOLD_2 = "2"
    FOLD_3 = "3"
    FOLD_4 = "4"

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
    Fold = _Fold

    def __init__(
        self,
        *,
        root: str,
        fold: Optional["PathologyFoundationDataset.Fold"] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        mmap_cache_size: int = _DEFAULT_MMAP_CACHE_SIZE,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._fold = fold
        self._get_entries()
        self._get_cohort_names()
        self._mmap_tarball = _make_mmap_tarball(self._tarballs_root, mmap_cache_size)

    @property
    def _tarballs_root(self) -> str:
        return self.root

    @property
    def fold(self) -> "PathologyFoundationDataset.Fold":
        return self._fold

    @property
    def _entries_name(self) -> str:
        return self._fold.entries_name() if self._fold else "pretrain_entries.npy"

    def _get_entries(self) -> np.ndarray:
        self._entries = self._load_entries(self._entries_name)

    def _load_entries(self, _entries_name: str) -> np.ndarray:
        entries_path = Path(self.root, _entries_name)
        return np.load(entries_path, mmap_mode="r")

    def _get_filepaths_dict(self, cohort_name: str):
        return self._load_filepaths_dict(cohort_name)

    def _load_filepaths_dict(self, cohort_name: str):
        filepaths_dict_path = Path(self.root, f"{cohort_name}_file_indices.npy")
        return np.load(filepaths_dict_path, allow_pickle=True).item()

    def _get_cohort_names(self) -> dict:
        self._cohort_names = self._load_cohort_names()

    def _load_cohort_names(self) -> dict:
        cohort_dict_path = Path(self.root, "cohort_indices.npy")
        return np.load(cohort_dict_path, allow_pickle=True).item()

    def get_image_data(self, index: int) -> bytes:
        entry = self._entries[index]
        file_idx, start_offset, end_offset, cohort_idx = entry[1], entry[2], entry[3], entry[4]
        cohort_name = self._cohort_names[cohort_idx]
        filepaths_dict = self._get_filepaths_dict(cohort_name)
        filepath = filepaths_dict[file_idx]
        class_mmap = self._mmap_tarball(cohort_name)
        # try:
        #     mapped_data = class_mmap[start_offset:end_offset]
        #     data = mapped_data[512:]  # Skip entry header block
        #     with GzipFile(fileobj=BytesIO(data)) as g:
        #         data = g.read()
        # except Exception as e:
        #     raise RuntimeError(
        #         f"can not retrieve image data for sample {index} " f'from "{cohort_name}" tarball'
        #     ) from e
        data = class_mmap[start_offset:end_offset]
        return data, Path(filepath)

    def get_target(self, index: int) -> Any:
        return int(self._entries[index]["class_index"])

    def get_targets(self) -> np.ndarray:
        return self._entries["class_index"]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._entries)
