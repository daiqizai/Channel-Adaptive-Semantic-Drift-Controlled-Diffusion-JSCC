from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FlatImageDataset(Dataset):
    def __init__(self, root: str | Path, transform=None) -> None:
        self.root = Path(root)
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(f"Image directory not found: {self.root}")
        self.paths = sorted(
            path for path in self.root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise FileNotFoundError(f"No image files found under: {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0
