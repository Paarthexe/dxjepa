from typing import List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import torch
from torch.utils.data import Dataset


class BigEarthNetDataset(Dataset):
    """Dataset for multimodal Sentinel-1 SAR and Sentinel-2 Optical satellite imagery."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        labels: Optional[List[str]] = None,
        image_size: int = 224,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size

        if labels is None:
            if "labels" in self.df.columns and len(self.df) > 0:
                self.labels = sorted(
                    {lbl for item in self.df["labels"] for lbl in item}
                )
            else:
                self.labels = []
        else:
            self.labels = labels

        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

    def __len__(self) -> int:
        return len(self.df)

    def load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load GeoTIFF raster image and bilinearly resample to (channels, image_size, image_size)."""
        path = str(path)
        try:
            with rasterio.open(path) as src:
                image = src.read(
                    out_shape=(src.count, self.image_size, self.image_size),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)
            return image
        except Exception as e:
            raise RuntimeError(f"Error loading image file {path}: {e}")

    def encode_labels(self, labels: List[str]) -> torch.Tensor:
        """Multi-hot binary vector encoding for labels."""
        target = torch.zeros(len(self.labels), dtype=torch.float32)
        for label in labels:
            if label in self.label_to_idx:
                target[self.label_to_idx[label]] = 1.0
        return target

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.df.iloc[idx]
        sar = self.load_image(sample.s1_path)
        optical = self.load_image(sample.s2_path)

        # Normalization exactly matching notebook
        sar = np.clip(sar, -25.0, 5.0)
        sar = (sar + 25.0) / 30.0

        optical = np.clip(optical, 0.0, 10000.0)
        optical = optical / 10000.0

        sar_tensor = torch.from_numpy(sar)
        optical_tensor = torch.from_numpy(optical)
        labels_tensor = self.encode_labels(sample.labels)

        return sar_tensor, optical_tensor, labels_tensor
