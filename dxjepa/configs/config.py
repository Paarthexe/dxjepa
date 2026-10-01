from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class DataConfig:
    data_root: str = field(
        default_factory=lambda: os.environ.get(
            "BIGEARTHNET_ROOT",
            "/kaggle/input/datasets/narendraaironi/bigearthnet-14k/BEN_14k",
        )
    )
    metadata_filename: str = "metadata.parquet"
    s1_dirname: str = "BigEarthNet-S1"
    s2_dirname: str = "BigEarthNet-S2"
    image_size: int = 224
    patch_size: int = 16
    sar_channels: int = 2
    optical_channels: int = 10
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = True


@dataclass
class MaskConfig:
    context_mask_ratio: float = 0.5
    target_mask_ratio: float = 0.5

    @property
    def num_patches(self) -> int:
        return (224 // 16) ** 2  # 196


@dataclass
class ModelConfig:
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    predictor_embed_dim: int = 384
    predictor_depth: int = 12
    predictor_heads: int = 12
    num_shared_queries: int = 16
    d_common: int = 512
    sar_channels: int = 2
    optical_channels: int = 10
    patch_size: int = 16
    image_size: int = 224

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def d_unique(self) -> int:
        return self.embed_dim - self.d_common


@dataclass
class LossConfig:
    lambda_psa: float = 0.1
    lambda_inv: float = 1.0
    lambda_var: float = 25.0
    lambda_cov: float = 25.0
    lambda_excl: float = 5.0
    lambda_var_min: float = 5.0
    lambda_cov_min: float = 2.0
    lambda_excl_min: float = 1.0
    eps_psa: float = 1e-4


@dataclass
class TrainingConfig:
    epochs: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.04
    warmup_epochs: int = 15
    grad_clip: float = 1.0
    seed: int = 67
    use_amp: bool = True
    eval_freq: int = 10
    checkpoint_freq: int = 50
    checkpoint_dir: str = "checkpoints"
    output_model_path: str = "dxjepa_model.pth"
    device: Optional[str] = None
    synthetic_data: bool = False


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load experiment configuration from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            raw_data = json.load(f)

        data_cfg = DataConfig(**raw_data.get("data", {}))
        mask_cfg = MaskConfig(**raw_data.get("mask", {}))
        model_cfg = ModelConfig(**raw_data.get("model", {}))
        loss_cfg = LossConfig(**raw_data.get("loss", {}))
        train_cfg = TrainingConfig(**raw_data.get("training", {}))

        return cls(
            data=data_cfg,
            mask=mask_cfg,
            model=model_cfg,
            loss=loss_cfg,
            training=train_cfg,
        )

    def to_json(self, path: Union[str, Path]) -> None:
        """Save experiment configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
