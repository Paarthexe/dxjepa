import argparse
import math
import os
from pathlib import Path
import torch
from tqdm.auto import tqdm

from dxjepa.configs.config import ExperimentConfig
from dxjepa.data.builder import build_dataframe, build_dataloaders
from dxjepa.evaluation.retrieval import evaluate_retrieval_f1
from dxjepa.losses.criterion import compute_loss
from dxjepa.models.xjepa import XJEPA
from dxjepa.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Decoupled X-JEPA (dxjepa) model. All parameters can be configured via config.json."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to JSON configuration file (default: config.json)",
    )
    # Optional CLI overrides (only used if explicitly provided)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override dataset directory from config.json",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total epochs from config.json",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config.json",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config.json",
    )
    parser.add_argument(
        "--synthetic_data",
        action="store_true",
        default=None,
        help="Override synthetic data flag for testing without full dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override target device ('cuda', 'mps', 'cpu')",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> ExperimentConfig:
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading configuration from: {config_path}")
        config = ExperimentConfig.from_json(config_path)
    else:
        print(f"Config file {config_path} not found. Using default ExperimentConfig.")
        config = ExperimentConfig()

    # Apply optional CLI overrides if provided
    if args.data_dir is not None:
        config.data.data_root = args.data_dir
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.training.lr = args.lr
    if args.synthetic_data is not None:
        config.training.synthetic_data = args.synthetic_data
    if args.device is not None:
        config.training.device = args.device

    return config


def train(args: argparse.Namespace) -> None:
    config = load_config(args)

    # 1. Reproducibility
    set_seed(config.training.seed)

    # 2. Device determination
    if config.training.device:
        device = torch.device(config.training.device)
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    print(f"Using device: {device}")

    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    # 3. Data Loading
    if config.training.synthetic_data:
        print("Running in synthetic data mode (benchmarking / testing).")
        dataloaders = build_dataloaders(
            use_synthetic=True,
            batch_size=config.data.batch_size,
            num_workers=0 if device.type == "cpu" else config.data.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            synthetic_samples=512,
        )
    else:
        data_root = Path(config.data.data_root)
        print(f"Loading BigEarthNet dataset from: {data_root.resolve()}")
        try:
            df = build_dataframe(
                data_root=data_root,
                metadata_path=data_root / config.data.metadata_filename,
                s1_root=data_root / config.data.s1_dirname,
                s2_root=data_root / config.data.s2_dirname,
            )
            print(f"Loaded {len(df)} samples across splits:")
            print(df["split"].value_counts())
            dataloaders = build_dataloaders(
                df=df,
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=config.data.drop_last,
                persistent_workers=(config.data.num_workers > 0),
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to load dataset from {data_root}: {e}")
            print(
                "\nHint: Update 'data_root' in config.json to your dataset path, e.g.:\n"
                '  "data": { "data_root": "/path/to/BEN_14k" }\n'
                "Or set 'synthetic_data': true in config.json to test the pipeline.\n"
            )
            raise

    train_loader = dataloaders["train"]
    val_loader = dataloaders["validation"]

    # 4. Model Construction
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized: {total_params:,} parameters")

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    if config.training.epochs > config.training.warmup_epochs:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.training.warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs - config.training.warmup_epochs,
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.training.warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.epochs, eta_min=1e-6
        )

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    # 6. Mixed Precision
    use_amp = (device.type == "cuda") and config.training.use_amp
    amp_dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # 7. Training Loop
    num_batches = max(1, len(train_loader))
    total_steps = config.training.epochs * num_batches

    print(f"\nStarting training for {config.training.epochs} epochs ({num_batches} batches/epoch)...")

    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        running_loss = 0.0
        running_l2 = 0.0
        running_psa = 0.0
        running_vicreg = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.training.epochs}",
            leave=True,
        )

        for step, batch in enumerate(pbar):
            sar_b = batch[0].to(device)
            optical_b = batch[1].to(device)

            optimizer.zero_grad()

            current_step = epoch * num_batches + step
            cosine_decay = (
                math.cos(math.pi * current_step / total_steps) * 0.5 + 0.5
            )
            sched_lambda_var = (
                config.loss.lambda_var_min
                + (config.loss.lambda_var - config.loss.lambda_var_min) * cosine_decay
            )
            sched_lambda_cov = (
                config.loss.lambda_cov_min
                + (config.loss.lambda_cov - config.loss.lambda_cov_min) * cosine_decay
            )
            sched_lambda_excl = (
                config.loss.lambda_excl_min
                + (config.loss.lambda_excl - config.loss.lambda_excl_min) * cosine_decay
            )

            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                outputs = model(sar_b, optical_b)
                loss_dict = compute_loss(
                    outputs,
                    model,
                    lambda_inv=config.loss.lambda_inv,
                    lambda_var=sched_lambda_var,
                    lambda_cov=sched_lambda_cov,
                    lambda_excl=sched_lambda_excl,
                    lambda_psa=config.loss.lambda_psa,
                    d_common=config.model.d_common,
                )
                loss = loss_dict["total_loss"]

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.grad_clip
                )
                optimizer.step()

            running_loss += loss.item()
            running_l2 += loss_dict["loss_l2"].item()
            running_psa += loss_dict["loss_psa"].item()
            running_vicreg += loss_dict["loss_vicreg"].item()

            lr_val = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                l2=f"{running_l2 / (step + 1):.4f}",
                psa=f"{running_psa / (step + 1):.4f}",
                vic=f"{running_vicreg / (step + 1):.4f}",
                lr=f"{lr_val:.2e}",
            )

        scheduler.step()

        steps_in_epoch = max(1, len(train_loader))
        epoch_loss = running_loss / steps_in_epoch
        epoch_l2 = running_l2 / steps_in_epoch
        epoch_psa = running_psa / steps_in_epoch
        epoch_vicreg = running_vicreg / steps_in_epoch

        print(
            f"Epoch {epoch + 1:3d}/{config.training.epochs} | Total Loss: {epoch_loss:.4f} | "
            f"L2: {epoch_l2:.4f} | PSA: {epoch_psa:.4f} | VICReg: {epoch_vicreg:.4f}"
        )

        # Periodic retrieval evaluation
        if (epoch + 1) % config.training.eval_freq == 0:
            print(f"\n--- Validation Retrieval Evaluation (Epoch {epoch + 1}) ---")
            val_metrics = evaluate_retrieval_f1(
                model,
                val_loader,
                device=device,
                d_common=config.model.d_common,
            )
            if val_metrics:
                print(
                    f"S1->S2 (SAR->Optical)  | F1@5: {val_metrics.get('S1->S2_F1@5', 0.0):.2f}% | "
                    f"F1@10: {val_metrics.get('S1->S2_F1@10', 0.0):.2f}%"
                )
                print(
                    f"S2->S1 (Optical->SAR)  | F1@5: {val_metrics.get('S2->S1_F1@5', 0.0):.2f}% | "
                    f"F1@10: {val_metrics.get('S2->S1_F1@10', 0.0):.2f}%"
                )
                print(
                    f"S1->S1 (SAR->SAR)      | F1@5: {val_metrics.get('S1->S1_F1@5', 0.0):.2f}% | "
                    f"F1@10: {val_metrics.get('S1->S1_F1@10', 0.0):.2f}%"
                )
                print(
                    f"S2->S2 (Optical->Opt)  | F1@5: {val_metrics.get('S2->S2_F1@5', 0.0):.2f}% | "
                    f"F1@10: {val_metrics.get('S2->S2_F1@10', 0.0):.2f}%\n"
                )

        # Periodic checkpoint
        if (epoch + 1) % config.training.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                config.training.checkpoint_dir, f"dxjepa_epoch_{epoch + 1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loss": epoch_loss,
                    "loss_l2": epoch_l2,
                    "loss_psa": epoch_psa,
                    "loss_vicreg": epoch_vicreg,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")

    # 8. Save Final Model
    torch.save(model.state_dict(), config.training.output_model_path)
    print(f"\nModel saved successfully to: {config.training.output_model_path}")


if __name__ == "__main__":
    train(parse_args())
