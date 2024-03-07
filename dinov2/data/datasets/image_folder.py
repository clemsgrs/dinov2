from pathlib import Path
from torchvision import datasets
from typing import Callable, Optional


class ImageFolderWithNameDataset(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            root,
            transform,
        )

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[idx]
        fname = Path(path).stem
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, fname
