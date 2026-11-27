# NoiseRegu-LIC

**Noise-Regularized Training for Learned Image Compression**  
**NeurIPS 2026 Spotlight**

[![Paper](https://img.shields.io/badge/arXiv-2611.10073-b31b1b.svg)](https://arxiv.org/abs/2611.10073)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-Spotlight-6f42c1.svg)](https://arxiv.org/abs/2611.10073)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A lightweight noise regularizer, a strong end-to-end codec training framework, and reproducible evaluation tools—all in one place.

## What’s in this repository?

### 1. A noise regularizer you can drop into almost any codec

NoiseRegu adds Gaussian noise to the **encoder input** during training while keeping the clean image as the reconstruction target. It is architecture-agnostic, works with existing learned image compression frameworks, and introduces essentially no additional training or inference overhead. Porting it to another codec takes only a few lines of code.

#### Use in your own codec

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

That is the entire training-side change. The latent noise proxy, hyperprior, entropy model, and inference code stay untouched.

- **Two-stage training (256×256 → 512×512):** anneal noise throughout the 256×256 stage, then fine-tune at 512×512 without noise.
- **Single-stage training:** stop noise at roughly 80–90% of training.

### 2. A powerful end-to-end training framework

Beyond the regularizer, this repository provides the core algorithm and training framework behind our first-place solutions in two challenges:

- **First place, image track:** [1st Challenge on Perceptual Video & Image Coding](https://icme2026.labiip.com/), ICME 2026.
- **First place, objective track:** [LoViF 2026 AIGC Image Compression Challenge: Methods and Results](https://openreview.net/pdf?id=41hfEpLx7Z).

**Perceptual training vs. VTM-22.0**  
BD-Rate averaged over Kodak, Tecnick, and CLIC; negative values mean bitrate savings.

| PSNR | MS-SSIM | LPIPS | DISTS |
|---:|---:|---:|---:|
| **-14.36%** | **-14.41%** | **-89.91%** | **-90.84%** |

### 3. Tools for evaluation, comparison, and alignment

The repository also includes practical utilities for reproducing and extending our results:

- Real entropy-coded evaluation with PSNR, MS-SSIM, LPIPS, and DISTS.
- DMCI Base, Medium, and Large checkpoints, including perceptual training.
- VTM-22.0 anchors and scripts for consistent baseline comparisons.
- 57 reference rate–distortion files covering DMCI, ELIC, TCM, MLIC++, DCAE, LALIC, and HPCM.
- Small, editable scripts for training, evaluation, BD-Rate calculation, and result alignment.

## Why NoiseRegu works

NoiseRegu moves stochastic perturbation from the latent domain to the encoder input, while the reconstruction target remains clean. A second-order analysis shows that this creates two useful forms of regularization: an **information-weighted encoder-Jacobian penalty** on the rate term and a **contractive penalty** on the end-to-end reconstruction map. In practice, this improves rate–distortion performance without changing the codec architecture or inference pipeline.

## Results at a glance

- **ELIC:** **-2.60%** average BD-Rate compared with clean training.
- **DMCI Base / Medium / Large:** **-0.85% / -1.05% / -2.38%** compared with their clean counterparts.
- **DMCI-Large:** **-22.02%** average BD-Rate against VTM-22.0.
- **Perceptual training:** consistent gains over clean training across PSNR, MS-SSIM, LPIPS, and DISTS.

All results below report BD-Rate; negative values mean bitrate savings.

<details>
<summary><strong>Full MSE results</strong></summary>

### NoiseRegu versus clean training

| Backbone | Kodak | Tecnick | CLIC | Average |
|---|---:|---:|---:|---:|
| ELIC | **-2.10%** | **-3.28%** | **-2.42%** | **-2.60%** |
| DMCI-Base | **-0.90%** | **-0.85%** | **-0.80%** | **-0.85%** |
| DMCI-Medium | **-0.98%** | **-0.93%** | **-1.24%** | **-1.05%** |
| DMCI-Large | **-2.32%** | **-2.32%** | **-2.50%** | **-2.38%** |
| DMCI-Base (official release)† | +2.91% | +5.60% | +5.45% | +4.66% |

† Official DMCI-Base versus our reproduced clean DMCI-Base; the other rows compare NoiseRegu with the corresponding clean model.

### Clean and NoiseRegu versus VTM-22.0

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

</details>

<details>
<summary><strong>Full perceptual results</strong></summary>

### NoiseRegu versus clean training

| Dataset | PSNR | MS-SSIM | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| Kodak | -1.63% | -1.99% | -2.13% | -2.84% |
| Tecnick | -0.63% | -0.77% | -1.12% | -1.94% |
| CLIC | -1.01% | -1.49% | -2.83% | -4.85% |

### Clean and NoiseRegu versus VTM-22.0

| Dataset | Variant | PSNR | MS-SSIM | LPIPS | DISTS |
|---|---|---:|---:|---:|---:|
| Kodak | Clean | -14.20% | -18.29% | -90.41% | -90.67% |
| Kodak | **NoiseRegu** | **-15.75%** | **-20.04%** | **-90.66%** | **-91.01%** |
| Tecnick | Clean | -17.10% | -14.62% | -87.34% | -88.29% |
| Tecnick | **NoiseRegu** | **-17.65%** | **-15.32%** | **-87.43%** | **-88.43%** |
| CLIC | Clean | -8.59% | -6.26% | -91.49% | -92.92% |
| CLIC | **NoiseRegu** | **-9.69%** | **-7.87%** | **-91.65%** | **-93.07%** |

</details>

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

```text
MSE:         dmci_{base,medium,large}_mse_{clean,noise}.pth
Perceptual:  dmci_large_perceptual_noise.pth
Fixed-QP:    dmci_{base,medium,large}_fixed.pth
```

`dmci_base_fixed.pth` is currently included. `checkpoints/manifest.json` records file sizes and SHA256 hashes.

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

We used [MLIC-Train-100K](https://huggingface.co/datasets/Whiteboat/MLIC-Train-100K). You can use your own dataset; we chose this one because training was relatively stable in our experiments.

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

All 57 reference JSON files are directly under `results/`. New evaluations go to `results/evaluations/`; BD-Rate outputs go to `results/bdrate/`.

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

<details>
<summary><strong>Reference JSON format</strong></summary>

DMCI, ELIC, TCM, MLIC++, DCAE, LALIC, HPCM, and VTM share one compact schema:

```json
{
  "name": "dmci_base_noise_kodak",
  "dataset": "kodak",
  "qps": [0, 9, 18, 27, 36, 45, 54, 63],
  "results": {"bpp": ["..."], "psnr": ["..."]}
}
```

Perceptual files additionally contain `ms_ssim`, `lpips`, and `dists`. Arrays follow the order in `qps`.

</details>

## License

NoiseRegu-LIC is released under the [MIT License](LICENSE).

## Citation

If you find this work useful, please cite:

```bibtex
@misc{zou2026noisereg,
  title={Noise-Regularized Training for Learned Image Compression},
  author={Zou, Renjie and Huang, Zhiwei and Jiang, Dong},
  year={2026},
  eprint={2611.10073},
  archivePrefix={arXiv},
  primaryClass={eess.IV},
  url={https://arxiv.org/abs/2611.10073},
  note={NeurIPS 2026 Spotlight}
}
```
