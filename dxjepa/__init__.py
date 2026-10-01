"""Decoupled X-JEPA (dxjepa) - Stage 3: Modular Architecture & Loss Engine."""

from dxjepa.models.xjepa import XJEPA
from dxjepa.models.stem import PatchEmbedding, ModalityStem
from dxjepa.models.transformer import (
    MLP,
    TransformerBlock,
    SharedTransformer,
    build_encoder_trunk,
)
from dxjepa.models.predictor import XJEPAPredictorBlock, XJEPAPredictor
from dxjepa.models.pooling import MeanPooling, compute_latent_embeddings
from dxjepa.losses.psa import XJEPAPSALoss
from dxjepa.losses.vicreg import (
    decoupled_vicreg_loss,
    vicreg_loss,
    split_common_unique,
    cross_covariance_penalty,
)
from dxjepa.losses.criterion import compute_loss, l2_prediction_loss

__all__ = [
    "XJEPA",
    "PatchEmbedding",
    "ModalityStem",
    "MLP",
    "TransformerBlock",
    "SharedTransformer",
    "build_encoder_trunk",
    "XJEPAPredictorBlock",
    "XJEPAPredictor",
    "MeanPooling",
    "compute_latent_embeddings",
    "XJEPAPSALoss",
    "decoupled_vicreg_loss",
    "vicreg_loss",
    "split_common_unique",
    "cross_covariance_penalty",
    "compute_loss",
    "l2_prediction_loss",
]
