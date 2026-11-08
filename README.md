# Noise-Regularized Training for Learned Image Compression

![NeurIPS 2026 Spotlight](https://img.shields.io/badge/NeurIPS%202026-Spotlight-6F42C1)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)

NoiseRegu-LIC adds Gaussian noise to the **encoder input** during training while keeping the clean image as the reconstruction target. The latent noise proxy, hyperprior, entropy coder, and inference pipeline remain unchanged.

- One training-only change, no inference overhead.
- DMCI Base, Medium, and Large checkpoints.
- Real entropy-coded evaluation, BD-Rate, VTM anchors, and 57 reference RD files.

## Use it in your codec

The complete training-side change is only a few lines:

```python
sigma_start = 0.5                      # recommended: [0.4, 0.8]
noise_epochs = int(0.85 * epochs)      # single-stage: stop at 80–90%

target = x
if epoch < noise_epochs:
    sigma = sigma_start * math.exp(-4.6 * epoch / max(noise_epochs - 1, 1))
    x = (x + torch.rand(()).item() * sigma * torch.randn_like(x)).clamp(0, 1)
x_hat, rate = codec(x)
loss = rate + lam * mse(x_hat, target)  # clean target
```

The latent noise proxy, entropy model, and inference code stay untouched. Anneal `sigma` during training, starting in `[0.4, 0.8]`:

- **Two stages (256×256 → 512×512):** anneal noise throughout the 256×256 stage, then fine-tune at 512×512 without noise.
- **One stage:** stop noise at roughly 80–90% of training, for example `noise_epochs = int(0.85 * epochs)`.

A second-order analysis shows that this produces an information-weighted encoder-Jacobian penalty and a contractive penalty on the reconstruction map.

## Results

BD-Rate results; negative values mean bitrate savings.

### MSE: NoiseRegu versus clean

| Backbone | Kodak | Tecnick | CLIC | Average |
|---|---:|---:|---:|---:|
| ELIC | **-2.10%** | **-3.28%** | **-2.42%** | **-2.60%** |
| DMCI-Base | **-0.90%** | **-0.85%** | **-0.80%** | **-0.85%** |
| DMCI-Medium | **-0.98%** | **-0.93%** | **-1.24%** | **-1.05%** |
| DMCI-Large | **-2.32%** | **-2.32%** | **-2.50%** | **-2.38%** |
| DMCI-Base (official release)† | +2.91% | +5.60% | +5.45% | +4.66% |

† Official DMCI-Base versus our reproduced clean DMCI-Base; the other rows compare NoiseRegu with the corresponding clean model.

### MSE: clean and NoiseRegu versus VTM-22.0

| Backbone | Variant | Kodak | Tecnick | CLIC | Average |
|---|---|---:|---:|---:|---:|
| ELIC | Clean | -5.87% | -8.11% | -4.42% | -6.13% |
| ELIC | **NoiseRegu** | **-7.98%** | **-11.35%** | **-6.79%** | **-8.71%** |
| DMCI-Base | Official release | -12.67% | -15.38% | -12.63% | -13.56% |
| DMCI-Base | Clean | -15.23% | -19.87% | -16.99% | -17.36% |
| DMCI-Base | **NoiseRegu** | **-16.04%** | **-20.59%** | **-17.70%** | **-18.11%** |
| DMCI-Medium | Clean | -17.37% | -21.85% | -18.45% | -19.23% |
| DMCI-Medium | **NoiseRegu** | **-18.24%** | **-22.63%** | **-19.53%** | **-20.13%** |
| DMCI-Large | Clean | -18.50% | -22.67% | -18.93% | -20.03% |
| DMCI-Large | **NoiseRegu** | **-20.47%** | **-24.55%** | **-21.06%** | **-22.02%** |

**Mixed perceptual objective: NoiseRegu versus clean**

| Dataset | PSNR | MS-SSIM | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| Kodak | -1.63% | -1.99% | -2.13% | -2.84% |
| Tecnick | -0.63% | -0.77% | -1.12% | -1.94% |
| CLIC | -1.01% | -1.49% | -2.83% | -4.85% |

**Mixed perceptual objective: clean and NoiseRegu versus VTM-22.0**

| Dataset | Variant | PSNR | MS-SSIM | LPIPS | DISTS |
|---|---|---:|---:|---:|---:|
| Kodak | Clean | -14.20% | -18.29% | -90.41% | -90.67% |
| Kodak | **NoiseRegu** | **-15.75%** | **-20.04%** | **-90.66%** | **-91.01%** |
| Tecnick | Clean | -17.10% | -14.62% | -87.34% | -88.29% |
| Tecnick | **NoiseRegu** | **-17.65%** | **-15.32%** | **-87.43%** | **-88.43%** |
| CLIC | Clean | -8.59% | -6.26% | -91.49% | -92.92% |
| CLIC | **NoiseRegu** | **-9.69%** | **-7.87%** | **-91.65%** | **-93.07%** |

## Get started

### 1. Install

Python 3.12, PyTorch, and a CUDA-capable GPU are recommended.

```bash
python -m pip install -r requirements.txt
bash scripts/build_rans.sh
```

Real entropy coding requires CUDA and the rANS extension. Custom CUDA inference kernels are optional; PyTorch fallback is the default.

### 2. Download checkpoints

Download the `.pth` files from the [release assets](https://github.com/Googolxx/NoiseRegu-LIC/releases) and place them in `checkpoints/`.

| Backbone | MSE | Perceptual |
|---|---|---|
| Base | `dmci_base_mse_{clean,noise}.pth` | — |
| Medium | `dmci_medium_mse_{clean,noise}.pth` | — |
| Large | `dmci_large_mse_{clean,noise}.pth` | `dmci_large_perceptual_noise.pth` |

Fixed checkpoints follow `dmci_{base,medium,large}_fixed.pth`; `dmci_base_fixed.pth` is currently included. `checkpoints/manifest.json` records file sizes and SHA256 hashes.

### 3. Evaluate

Edit the short configuration block at the top of `scripts/evaluate.sh`:

```bash
CHECKPOINT="checkpoints/dmci_base_mse_noise.pth"
IMAGE_DIR="/path/to/kodak"
MODEL="dmci"
METRICS="psnr"
SAVE_DIR=""  # Optional: save .bin streams and .png reconstructions
```

Then run:

```bash
bash scripts/evaluate.sh
```

The default QP grid is `0,9,18,27,36,45,54,63`. For perceptual evaluation, select the perceptual checkpoint, set `MODEL="dmci_large"`, and use `METRICS="psnr,ms_ssim,lpips,dists"`. Set `SAVE_DIR` to save each image as `SAVE_DIR/qp_QP/IMAGE.bin` and `IMAGE.png`; reconstructions use the same 8-bit representation as the quality metrics.

### 4. Calculate BD-Rate

Choose the two RD files in `scripts/bdrate.sh`, then run:

```bash
bash scripts/bdrate.sh
```

Use `METRIC="all"` for all four perceptual metrics.

## Train your own model

DMCI is variable-rate, but training starts from a fixed-QP model.

### Training data

We used [MLIC-Train-100K](https://huggingface.co/datasets/Whiteboat/MLIC-Train-100K). Everyone has a favorite training set; in our experience, we picked this one because training felt relatively stable

### 1. Fixed-QP pretraining

The default fixed rate is QP63; use `--fixed_qp N` to choose another QP. Every batch uses that QP and its corresponding RD weight.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
  train.py -d /path/to/train --val_dir /path/to/val \
  --save_dir outputs/fixed --model dmci --fixed_qp \
  --epochs 540 --lr_epoch 480 --patch-size 256 256 \
  --batch-size 8 --learning-rate 1e-4
```

The fixed model has one pair of quantization scales. Loading its checkpoint into the variable-rate model expands them to all 64 QPs. To skip fixed-QP pretraining, use the released `checkpoints/dmci_base_fixed.pth` as `CHECKPOINT` in the next step.

### 2. Variable-rate fine-tuning

Set the fixed checkpoint as `CHECKPOINT` in `scripts/train.sh`, along with the dataset, output directory, model, and GPUs. Then run:

```bash
bash scripts/train.sh
```

The variable-rate recipe first trains at 256×256, then fine-tunes at 512×512. The shell script includes the default schedule.

Use `INPUT_NOISE=true` for NoiseRegu and `false` for clean training. An initial sigma in `[0.4, 0.8]` is recommended. `--noise_epochs` controls how long input noise is applied and defaults to the total number of epochs.

## VTM baseline

Set `DATASET`, `DATASET_DIR`, and `VTM_BIN_DIR` in `scripts/vtm.sh`, then select the evaluation mode:

```bash
PERCEPTUAL=false  # RGB PSNR, QP 27,32,37,42,47,52
PERCEPTUAL=true   # PSNR, MS-SSIM, LPIPS, DISTS, perceptual QP grid
```

Run `bash scripts/vtm.sh`. VTM binaries and datasets are not redistributed. In perceptual mode, `SKIP_ENCODE=true` reuses existing reconstructions.

## Reference data and repository layout

All 57 reference JSON files are directly under `results/`: DMCI, ELIC, TCM, MLIC++, DCAE, LALIC, HPCM, and VTM share one compact schema:

```json
{
  "name": "dmci_base_noise_kodak",
  "dataset": "kodak",
  "qps": [0, 9, 18, 27, 36, 45, 54, 63],
  "results": {"bpp": ["..."], "psnr": ["..."]}
}
```

Perceptual files additionally contain `ms_ssim`, `lpips`, and `dists`. Arrays follow the order in `qps`. New evaluations go to `results/evaluations/`; BD-Rate outputs go to `results/bdrate/`.

```text
src/          DMCI models, layers, utilities, and rANS source
train.py      MSE training with optional input noise
evaluate.py   Real entropy-coded evaluation
scripts/      Small, editable shell entry points
checkpoints/  Checkpoints and manifest
results/      Reference RD curves
bench/        VTM benchmark runner and configuration
tools/        Data loading, BD-Rate, and VTM conversion
```

## License

NoiseRegu-LIC is released under the [MIT License](LICENSE).
