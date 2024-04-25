import numpy as np
import multiresolutionimageinterface as mir

from pathlib import Path
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import VisionDataset

from .decoders import ImageDataDecoder


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
        # TODO: make this more memory & time efficient
        self._filepaths = np.load(Path(root, "pretrain_slide_indices.npy"), allow_pickle=True).item()
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

    def get_slide_path(self, idx: int) -> str:
        # TODO: make this more memory & time efficient
        return self._filepaths[idx]

    def get_slide(self, path: str) -> mir.MultiResolutionImage:
        return self._reader.open(path)

    def get_tile(self, wsi: mir.MultiResolutionImage, x: int, y: int, tile_size: int, level: int) -> np.ndarray:
        return wsi.getUCharPatch(x, y, tile_size, tile_size, level)

    def get_image_data(self, index: int) -> bytes:
        entry = self._entries[index]
        x, y, tile_size, level, slide_idx = entry[0], entry[1], entry[2], entry[3], entry[4]
        slide_path = self.get_slide_path(slide_idx)
        slide = self.get_slide(slide_path)
        tile = self.get_tile(slide, x, y, tile_size, level)
        return tile

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
