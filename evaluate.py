import argparse
import json
import math
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.layers.cuda_inference import replicate_pad
from src.models.image_model import DMCI, DMCI_Large, DMCI_Medium
from src.utils.common import get_state_dict, set_torch_env

MODEL_REGISTRY = {
    "dmci": DMCI,
    "dmci_large": DMCI_Large,
    "dmci_medium": DMCI_Medium,
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
QUALITY_METRICS = ("psnr", "ms_ssim", "lpips", "dists")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DMCI checkpoint with real entropy coding")
    parser.add_argument("checkpoint")
    parser.add_argument("image_dir")
    parser.add_argument("--model", default="dmci", choices=list(MODEL_REGISTRY))
    parser.add_argument("--metrics", default="psnr", help="Comma-separated: psnr,ms_ssim,lpips,dists")
    parser.add_argument("--output", default="results/eval.json")
    parser.add_argument("--qp_list", default="0,9,18,27,36,45,54,63")
    parser.add_argument("--save_dir", help="Optional directory for .bin streams and 8-bit .png reconstructions")
    return parser.parse_args()


def parse_metrics(value):
    metrics = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    invalid = set(metrics) - set(QUALITY_METRICS)
    if not metrics or invalid:
        raise ValueError(f"Invalid metrics {sorted(invalid)}; choose from {QUALITY_METRICS}")
    return metrics


def load_image(image_path, device):
    with Image.open(image_path) as image:
        tensor = transforms.ToTensor()(image.convert("RGB"))
    return tensor.unsqueeze(0).to(device=device, dtype=torch.float32)


def quality_scores(x_hat, target, metrics, perceptual_metrics):
    target_8bit = (target * 255.0).clamp(0.0, 255.0).round()
    x_hat_8bit = (x_hat * 255.0).clamp(0.0, 255.0).round()
    scores = {}
    if perceptual_metrics is not None:
        scores.update(perceptual_metrics.evaluate(x_hat_8bit / 255.0, target_8bit / 255.0))
    if "psnr" in metrics:
        mse = max(F.mse_loss(x_hat_8bit, target_8bit).item(), 1e-10)
        scores["psnr"] = 10.0 * math.log10(255.0 * 255.0 / mse)
    return {metric: scores[metric] for metric in metrics}


def save_outputs(save_dir, image_path, qp, bit_stream, x_hat):
    output_dir = os.path.join(save_dir, f"qp_{qp}")
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    with open(os.path.join(output_dir, f"{stem}.bin"), "wb") as file:
        file.write(bit_stream)

    x_hat_8bit = (x_hat.squeeze(0).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()
    transforms.ToPILImage()(x_hat_8bit).save(os.path.join(output_dir, f"{stem}.png"))


def evaluate_image(model, image_path, qp, metrics, perceptual_metrics, save_dir=None):
    device = next(model.parameters()).device
    target = load_image(image_path, device)
    _, _, height, width = target.shape
    padding_r, padding_b = DMCI.get_padding_size(height, width, 16)
    padded = replicate_pad(target, padding_b, padding_r)
    _, _, padded_height, padded_width = padded.shape

    with torch.no_grad():
        output = model.compress(padded, qp)
        x_hat = output["x_hat"][:, :, :height, :width]
        result = quality_scores(x_hat, target, metrics, perceptual_metrics)
        result["bpp"] = len(output["bit_stream"]) * 8.0 / (padded_height * padded_width)
        if save_dir:
            save_outputs(save_dir, image_path, qp, output["bit_stream"], x_hat)
    return {"real": result}


def main():
    args = parse_args()
    metrics = parse_metrics(args.metrics)
    qp_list = [int(qp) for qp in args.qp_list.split(",")]
    image_files = [
        os.path.join(args.image_dir, name)
        for name in sorted(os.listdir(args.image_dir))
        if os.path.splitext(name.lower())[1] in IMAGE_EXTS
    ]
    if not image_files:
        raise RuntimeError(f"No images found in {args.image_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real entropy coding")

    set_torch_env()
    device = "cuda"
    model = MODEL_REGISTRY[args.model]()
    model.load_state_dict(get_state_dict(args.checkpoint), strict=True)
    model.to(device).eval()
    model.update(0.0)

    perceptual_metrics = None
    if any(metric != "psnr" for metric in metrics):
        from perceptual_metric import PerceptualQualityMetrics

        perceptual_metrics = PerceptualQualityMetrics(device)

    collected = defaultdict(lambda: defaultdict(list))
    for qp in tqdm(qp_list, desc="Processing QPs"):
        for image_path in tqdm(image_files, desc=f"QP {qp}", leave=False):
            result = evaluate_image(model, image_path, qp, metrics, perceptual_metrics, args.save_dir)
            for key, value in result["real"].items():
                if value is not None and not math.isnan(value):
                    collected[qp][key].append(value)

    summary = {
        qp: {key: sum(values) / len(values) for key, values in collected[qp].items()}
        for qp in qp_list
    }
    output = {
        "name": os.path.splitext(os.path.basename(args.output))[0],
        "dataset": os.path.basename(os.path.normpath(args.image_dir)),
        "qps": qp_list,
        "results": {
            key: [summary[qp][key] for qp in qp_list]
            for key in ("bpp", *metrics)
        },
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"Results saved to: {args.output}")
    for qp in qp_list:
        values = "  ".join(f"{key}={value:.6f}" for key, value in summary[qp].items())
        print(f"QP {qp}: {values}")


if __name__ == "__main__":
    main()
