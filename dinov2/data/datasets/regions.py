import torch

from einops import rearrange
from pathlib import Path
from torchvision import transforms
from typing import Callable, Optional
from torchvision.datasets.folder import default_loader


class SlideIDsDataset(torch.utils.data.Dataset):
    """Dataset for iterating over slide IDs."""

    def __init__(self, slide_ids):
        """
        Args:
            slide_ids (list of str): List of slide IDs.
        """
        self.slide_ids = slide_ids

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, index):
        slide_id = self.slide_ids[index]
        return slide_id


class SlideRegionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: Path,
        slide_id: str,
        fmt: str = "jpg",
        image_size: int = 256,
        loader: Callable = default_loader,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.slide_id = slide_id
        self.format = fmt
        self.image_size = image_size
        self.transform = transform
        self.loader = loader
        self.region_paths = self._find_region_paths()

    def _find_region_paths(self):
        region_dir = Path(self.root_dir, self.slide_id, "imgs")
        sorted_region_paths = sorted([str(fp) for fp in region_dir.glob(f"*.{self.format}")])
        return sorted_region_paths

    def __len__(self):
        return len(self.region_paths)

    def __getitem__(self, idx):
        region_path = self.region_paths[idx]
        region = self.loader(region_path)
        region = transforms.functional.to_tensor(region)  # [3, region_size, region_size]
        region = region.unfold(1, self.image_size, self.image_size).unfold(
            2, self.image_size, self.image_size
        )  # [3, npatch, npatch, image_size, image_size]
        region = rearrange(region, "c p1 p2 w h -> (p1 p2) c w h")  # [num_patches, 3, image_size, image_size]
        if self.transform is not None:
            region = self.transform(region)
        return idx, region, region_path
