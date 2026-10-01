import argparse
import os
from pathlib import Path
import torch

from dxjepa.configs.config import ExperimentConfig
from dxjepa.data.builder import build_dataframe, build_dataloaders
from dxjepa.evaluation.retrieval import evaluate_retrieval_f1
from dxjepa.models.xjepa import XJEPA
from dxjepa.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained Decoupled X-JEPA (dxjepa) model on cross-modal retrieval. Reads config.json by default."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to JSON configuration file (default: config.json)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth). Defaults to output_model_path in config.json.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override dataset directory from config.json",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation", "train"],
        help="Split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--k_list",
        type=int,
        nargs="+",
        default=[5, 10],
        help="K values for retrieval F1@K (default: 5 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if config_path.exists():
        config = ExperimentConfig.from_json(config_path)
    else:
        config = ExperimentConfig()

    set_seed(config.training.seed)

    if config.training.device:
        device = torch.device(config.training.device)
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    print(f"Evaluating on device: {device}")

    # Determine checkpoint path
    checkpoint_path = args.checkpoint or config.training.output_model_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file '{checkpoint_path}' not found. Please provide via --checkpoint or train a model first."
        )

    # Build model
    model = XJEPA(
        embed_dim=config.model.embed_dim,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        predictor_depth=config.model.predictor_depth,
        predictor_heads=config.model.predictor_heads,
        predictor_embed_dim=config.model.predictor_embed_dim,
        num_shared_queries=config.model.num_shared_queries,
        num_patches=config.model.num_patches,
        patch_size=config.model.patch_size,
        sar_channels=config.model.sar_channels,
        optical_channels=config.model.optical_channels,
        context_mask_ratio=config.mask.context_mask_ratio,
        target_mask_ratio=config.mask.target_mask_ratio,
    ).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Prepare Data
    if config.training.synthetic_data:
        dataloaders = build_dataloaders(
            use_synthetic=True,
            batch_size=config.data.batch_size,
            num_workers=0 if device.type == "cpu" else config.data.num_workers,
            pin_memory=(device.type == "cuda"),
            synthetic_samples=256,
        )
        loader = dataloaders[args.split]
    else:
        data_root = Path(args.data_dir or config.data.data_root)
        print(f"Loading BigEarthNet data from: {data_root.resolve()}")
        df = build_dataframe(
            data_root=data_root,
            metadata_path=data_root / config.data.metadata_filename,
            s1_root=data_root / config.data.s1_dirname,
            s2_root=data_root / config.data.s2_dirname,
        )
        dataloaders = build_dataloaders(
            df=df,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        loader = dataloaders[args.split]

    print(f"Running retrieval evaluation on {args.split} set ({len(loader.dataset)} samples)...")
    metrics = evaluate_retrieval_f1(
        model=model,
        val_loader=loader,
        device=device,
        k_list=args.k_list,
        d_common=config.model.d_common,
    )

    print("\n" + "=" * 50)
    print(f"  Retrieval Evaluation Results ({args.split} set)")
    print("=" * 50)
    for task_name, score in metrics.items():
        print(f"  {task_name:24}: {score:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
