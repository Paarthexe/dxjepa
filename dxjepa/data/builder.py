import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from dxjepa.data.dataset import BigEarthNetDataset


def build_dataframe(
    data_root: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
    s1_root: Optional[Union[str, Path]] = None,
    s2_root: Optional[Union[str, Path]] = None,
    splits: Tuple[str, ...] = ("train", "validation", "test"),
) -> pd.DataFrame:
    """Build unified pandas DataFrame matching BigEarthNet-14k Sentinel-1 and Sentinel-2 pairs.

    Args:
        data_root: Base path to dataset root (e.g. /path/to/BEN_14k).
        metadata_path: Optional path to metadata.parquet. Defaults to data_root / 'metadata.parquet'.
        s1_root: Optional path to S1 root. Defaults to data_root / 'BigEarthNet-S1'.
        s2_root: Optional path to S2 root. Defaults to data_root / 'BigEarthNet-S2'.
        splits: Data splits to load.

    Returns:
        pd.DataFrame containing patch metadata, split designations, and resolved file paths.
    """
    data_root = Path(data_root)
    meta_path = Path(metadata_path) if metadata_path else data_root / "metadata.parquet"
    s1_dir = Path(s1_root) if s1_root else data_root / "BigEarthNet-S1"
    s2_dir = Path(s2_root) if s2_root else data_root / "BigEarthNet-S2"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata parquet file not found at: {meta_path}. "
            f"Please verify the dataset directory using --data_dir or set BIGEARTHNET_ROOT."
        )

    metadata = pd.read_parquet(meta_path)
    split_tables = []

    for split in splits:
        split_s1 = s1_dir / split
        split_s2 = s2_dir / split

        if not split_s1.exists() or not split_s2.exists():
            continue

        s1_files = {
            os.path.splitext(f)[0]
            for f in os.listdir(split_s1)
            if f.endswith(".tif")
        }
        s2_files = {
            os.path.splitext(f)[0]
            for f in os.listdir(split_s2)
            if f.endswith(".tif")
        }

        part = metadata[
            metadata["s1_name"].isin(s1_files) & metadata["patch_id"].isin(s2_files)
        ].copy()
        part["split"] = split
        part["s1_path"] = [split_s1 / f"{name}.tif" for name in part["s1_name"]]
        part["s2_path"] = [split_s2 / f"{name}.tif" for name in part["patch_id"]]
        split_tables.append(part)

    if not split_tables:
        raise RuntimeError(
            f"No matching Sentinel-1 / Sentinel-2 pairs found under {s1_dir} and {s2_dir}. "
            f"Check if image directories exist for splits: {splits}"
        )

    return pd.concat(split_tables, ignore_index=True)


class SyntheticBigEarthNetDataset(Dataset):
    """Synthetic dataset generating random tensors with identical shapes and ranges

    to BigEarthNet-14k for smoke testing, offline pipelines, and reproducibility verification.
    """

    def __init__(
        self,
        num_samples: int = 128,
        sar_channels: int = 2,
        optical_channels: int = 10,
        image_size: int = 224,
        num_classes: int = 19,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.image_size = image_size
        self.num_classes = num_classes

        rng = np.random.RandomState(seed)
        self.sar_data = rng.uniform(0.0, 1.0, size=(num_samples, sar_channels, image_size, image_size)).astype(np.float32)
        self.opt_data = rng.uniform(0.0, 1.0, size=(num_samples, optical_channels, image_size, image_size)).astype(np.float32)
        self.labels = (rng.uniform(0.0, 1.0, size=(num_samples, num_classes)) > 0.8).astype(np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.sar_data[idx]),
            torch.from_numpy(self.opt_data[idx]),
            torch.from_numpy(self.labels[idx]),
        )


def build_dataloaders(
    df: Optional[pd.DataFrame] = None,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = True,
    use_synthetic: bool = False,
    synthetic_samples: int = 512,
) -> Dict[str, DataLoader]:
    """Create train, validation, and test DataLoaders."""
    if use_synthetic or df is None:
        train_ds = SyntheticBigEarthNetDataset(num_samples=synthetic_samples, seed=42)
        val_ds = SyntheticBigEarthNetDataset(num_samples=synthetic_samples // 4, seed=43)
        test_ds = SyntheticBigEarthNetDataset(num_samples=synthetic_samples // 4, seed=44)
    else:
        labels = sorted({lbl for item in df["labels"] for lbl in item})
        train_ds = BigEarthNetDataset(df[df["split"] == "train"], labels=labels)
        val_ds = BigEarthNetDataset(df[df["split"] == "validation"], labels=labels)
        test_ds = BigEarthNetDataset(df[df["split"] == "test"], labels=labels)

    # Disable persistent_workers if num_workers == 0
    actual_persistent = persistent_workers if num_workers > 0 else False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last if len(train_ds) >= batch_size else False,
        persistent_workers=actual_persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last if len(val_ds) >= batch_size else False,
        persistent_workers=actual_persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=actual_persistent,
    )

    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader,
    }
