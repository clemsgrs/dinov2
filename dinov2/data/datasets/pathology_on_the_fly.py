import h5py
import numpy as np
import multiresolutionimageinterface as mir

from PIL import Image
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, List
from torchvision.datasets import VisionDataset


class _Subset:
    def __init__(self, value):
        self.value = value

    def entries_name(self):
        return f"pretrain_entries_{self.value}.npy"


class PathologyOnTheFlyDataset(VisionDataset):
    Subset = _Subset

    def __init__(
        self,
        *,
        root: str,
        subset: Optional["PathologyOnTheFlyDataset.Subset"] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._subset = subset
        self._reader = mir.MultiResolutionImageReader()
        self._load_slide_names()
        self._get_entries()

    @property
    def subset(self) -> "PathologyOnTheFlyDataset.Subset":
        return self._subset

    @property
    def _entries_name(self) -> str:
        return self._subset.entries_name() if self._subset else "pretrain_entries.npy"

    def _get_entries(self) -> np.ndarray:
        self._entries = self._load_entries(self._entries_name)

    def _load_entries(self, _entries_name: str) -> np.ndarray:
        entries_path = Path(self.root, _entries_name)
        return np.load(entries_path, mmap_mode="r")

    def _load_slide_names(self) -> List[str]:
        slide_names_path = Path(self.root, "slide_names.hdf5")
        with h5py.File(slide_names_path, "r") as h5f:
            self._slide_names = list(h5f["slide_names"])  # Load all filenames into memory

    def get_slide_path(self, idx: int) -> str:
        return self._slide_names[idx]

    def get_slide(self, path: str) -> mir.MultiResolutionImage:
        return self._reader.open(path)

    def get_tile(self, wsi: mir.MultiResolutionImage, x: int, y: int, tile_size: int, level: int) -> np.ndarray:
        return wsi.getUCharPatch(x, y, tile_size, tile_size, level)

    def get_image_data(self, index: int) -> bytes:
        entry = self._entries[index]
        x, y, tile_size, level, rsize_factor, slide_idx = entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]
        slide_path = self.get_slide_path(slide_idx)
        slide = self.get_slide(slide_path)
        tile = self.get_tile(slide, x, y, tile_size * rsize_factor, level)
        return tile, tile_size

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data, target_size = self.get_image_data(index)
            image = Image.fromarray(image_data).resize((target_size, target_size))
        except Exception as e:
            raise RuntimeError(f"Cannot read image for sample {index}") from e

        target = ()  # Empty target as per your requirement
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._entries)
