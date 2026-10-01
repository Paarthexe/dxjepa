# DX-JEPA: Decoupled Cross-Modal Joint Embedding Predictive Architecture

DX-JEPA is a self-supervised foundation model for multimodal remote sensing image retrieval across Sentinel-1 SAR (VV, VH) and Sentinel-2 Multi-Spectral Optical imagery.

It extends X-JEPA (Choudhury et al., WACV 2026) by introducing latent subspace decoupling, subspace-selective invariance, a cross-covariance exclusion penalty, and dynamic regularization scheduling.

## Novelties over X-JEPA

1. **Subspace Decoupling ($d_{\text{common}} \oplus d_{\text{unique}}$)**:
   X-JEPA enforces global invariance across the entire 768-dimensional latent space, penalizing valid modality-specific physical phenomena (e.g. radar volume scattering vs optical surface reflectance). DX-JEPA splits the latent space into a shared common subspace ($d_{\text{common}} = 512$) and a modality-unique subspace ($d_{\text{unique}} = 256$).

2. **Subspace-Selective Invariance**:
   Invariance is applied only to the shared common features ($c_{\text{sar}}, c_{\text{opt}}$). Unique features ($u_{\text{sar}}, u_{\text{opt}}$) remain unconstrained by cross-modal alignment:

   $$
   \mathcal{L}_{\text{inv}} = \frac{1}{B} \sum_{i=1}^B \| c_{\text{sar}}^{(i)} - c_{\text{opt}}^{(i)} \|_2^2
   $$

3. **Cross-Subspace Mutual Exclusion Penalty ($\mathcal{L}_{\text{excl}}$)**:
   Prevents shared and unique subspaces from collapsing or leaking information into each other:

   $$
   \mathcal{L}_{\text{excl}} = \frac{1}{3} \left[ \mathcal{C}(c_{\text{sar}}, u_{\text{sar}}) + \mathcal{C}(c_{\text{opt}}, u_{\text{opt}}) + \mathcal{C}(u_{\text{sar}}, u_{\text{opt}}) \right]
   $$

   where:

   $$
   \mathcal{C}(A, B) = \frac{1}{\dim(A) \cdot \dim(B)} \left\| \frac{(A - \bar{A})^\top (B - \bar{B})}{N - 1} \right\|_F^2
   $$

4. **Dynamic Cosine Regularization Scheduling**:
   X-JEPA uses static VICReg weights throughout training. DX-JEPA anneals variance, covariance, and exclusion weights via cosine decay to promote manifold expansion early and fine-grained alignment late:
   - $\lambda_{\text{var}}: 25.0 \to 5.0$
   - $\lambda_{\text{cov}}: 25.0 \to 2.0$
   - $\lambda_{\text{excl}}: 5.0 \to 1.0$

5. **Parameter Efficiency**:
   DX-JEPA uses a shared online encoder with no-grad target generation, requiring 117.46M parameters compared to X-JEPA's 172.86M parameters (a 32% reduction).

## BigEarthNet-14K Benchmark Results

Models evaluated on the BigEarthNet-14k test set using graded multi-label retrieval F1 score (top-5 neighbors):

| Method | Backbone Params (M) | S1 -> S2 (SAR to Opt) | S2 -> S1 (Opt to SAR) | S1 -> S1 (SAR Unimodal) | S2 -> S2 (Opt Unimodal) |
|---|:---:|:---:|:---:|:---:|:---:|
| MAE (He et al., 2022) | 224.87 | 41.78% | 46.12% | 60.81% | 72.04% |
| MAE-RVSA (Reed et al., 2022) | 227.75 | 36.66% | 38.05% | 55.40% | 71.47% |
| SatMAE (Cong et al., 2022) | 329.40 | 49.57% | 52.48% | 70.86% | 78.71% |
| SatMAE++ (Farid et al., 2023) | 329.14 | 50.21% | 54.98% | 67.29% | 76.48% |
| CrossMAE (Sumbul et al., 2023) | 250.57 | 49.46% | 48.71% | 66.45% | 71.28% |
| CSMAE-SESD (Disjoint) (WACV 2025) | 210.64 | 38.74% | 38.42% | 70.62% | 39.01% |
| SkySense (Guo et al., 2024) | 398.04 | 50.26% | 52.11% | 69.87% | 73.42% |
| CROMA (Anthony et al., 2023) | 310.54 | 46.53% | 48.61% | 68.48% | 72.71% |
| DeCUR (Li et al., 2023) | 250.54 | 40.78% | 41.83% | 71.26% | 75.36% |
| REJEPA (Gao et al., 2024) | 197.09 | 55.46% | 56.32% | 76.38% | 75.42% |
| X-JEPA (WACV 2026) | 172.86 | 61.23% | 63.73% | 72.98% | 82.65% |
| **DX-JEPA (Ours)** | **117.46** | **67.11%** | **68.08%** | 66.73% | 70.67% |

DX-JEPA improves cross-modal retrieval by +5.88% on S1 -> S2 and +4.35% on S2 -> S1 over baseline X-JEPA while using 55.4M fewer parameters.

## Loss Formulation

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} + \mathcal{L}_{\text{d-VICReg}}
$$

### Prediction Loss

$$
\mathcal{L}_{\text{pred}} = \mathcal{L}_{L_2} + \lambda_{\text{PSA}} \mathcal{L}_{\text{PSA}}
$$

- $\mathcal{L}_{L_2}$: Symmetric MSE loss between predicted and target patch representations.
- $\mathcal{L}_{\text{PSA}}$: Prediction Space Alignment loss using a learned metric weight matrix $M \in \mathbb{R}^{D \times D}$ initialized to $0.1 \cdot I$:

  $$
  \text{PSA}(\hat{Y}, Y) = \mathbb{E} \left[ \sqrt{ \| (\hat{Y} - Y) M \|_2^2 + \epsilon \| \hat{Y} - Y \|_2^2 + 10^{-6}} \right]
  $$

### Decoupled VICReg Loss

$$
\mathcal{L}_{\text{d-VICReg}} = \lambda_{\text{inv}} \mathcal{L}_{\text{inv}}(c_a, c_b) + \lambda_{\text{var}}(t) \mathcal{L}_{\text{var}} + \lambda_{\text{cov}}(t) \mathcal{L}_{\text{cov}} + \lambda_{\text{excl}}(t) \mathcal{L}_{\text{excl}}
$$

- Variance loss: computed across all four subspaces ($c_a, c_b, u_a, u_b$):

  $$
  v(Z) = \frac{1}{D} \sum_{j=1}^D \max(0, 1 - \sqrt{\text{Var}(Z_j) + \epsilon})
  $$

- Covariance loss: penalizes off-diagonal covariance across all four subspaces:

  $$
  c(Z) = \frac{1}{D^2} \sum_{i \neq j} [\text{Cov}(Z)]_{ij}^2
  $$

- Exclusion loss: cross-covariance penalty between $(c_a, u_a)$, $(c_b, u_b)$, and $(u_a, u_b)$.

## Repository Structure

```text
.
├── config.json                 # Centralized configuration (data paths, model, training)
├── dxjepa/                     # Core library package
│   ├── configs/                # Configuration dataclasses and JSON loaders
│   ├── data/                   # Dataset loader, dataframe builder, disjoint masking
│   ├── evaluation/             # Cross-modal retrieval F1@K metrics
│   ├── losses/                 # PSA, Decoupled VICReg, composite loss
│   ├── models/                 # Stems, shared ViT trunk, cross-modal predictor, XJEPA
│   └── utils/                  # Seeding and EMA target model updates
├── dxjepa.ipynb                # Self-contained alternative notebook
├── evaluate.py                 # Standalone retrieval evaluation script
├── pyproject.toml              # Build and package configuration
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── tests/                      # Unit test suite
└── train.py                    # Pretraining script
```

## Running Instructions

### 1. Installation

```bash
pip install -r requirements.txt
```

Or install as an editable package:
```bash
pip install -e .
```

### 2. Configuration (`config.json`)

All paths, hyperparameters, and schedules are managed in `config.json`. Update `data_root` to point to your local BigEarthNet-14K installation:

```json
{
  "data": {
    "data_root": "/path/to/BEN_14k",
    "batch_size": 256,
    "num_workers": 4
  },
  "training": {
    "epochs": 200,
    "lr": 0.0003,
    "output_model_path": "dxjepa_model.pth"
  }
}
```

### 3. Training via CLI

Run training directly without CLI arguments:
```bash
python train.py
```

`train.py` reads `config.json` by default. To point to an alternate configuration file:
```bash
python train.py --config path/to/custom_config.json
```

### 4. Evaluation via CLI

Evaluate a saved checkpoint on the retrieval benchmark:
```bash
python evaluate.py
```

To evaluate a specific checkpoint file:
```bash
python evaluate.py --checkpoint checkpoints/dxjepa_epoch_200.pth
```

### 5. Running Tests

Execute the unit test suite:
```bash
pytest
```

### 6. Alternative: Interactive Notebook (`dxjepa.ipynb`)

As an alternative to the modular CLI scripts, `dxjepa.ipynb` contains the complete end-to-end pipeline in a single self-contained Jupyter notebook suitable for interactive environments (Colab, Kaggle, local JupyterLab).

To run the notebook:
1. Open `dxjepa.ipynb`.
2. Update the dataset directory paths in cell 9:
   ```python
   DATA_ROOT = Path('/path/to/BEN_14k')
   METADATA_PATH = DATA_ROOT / 'metadata.parquet'
   S1_ROOT = DATA_ROOT / 'BigEarthNet-S1'
   S2_ROOT = DATA_ROOT / 'BigEarthNet-S2'
   ```
3. Execute cells sequentially.
