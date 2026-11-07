"""Benchmark VTM: encode/decode, save recon images, then compute perceptual metrics."""

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from bench.codecs import VTM

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="VTM benchmark with PSNR / MS-SSIM / LPIPS / DISTS"
    )
    parser.add_argument("dataset", type=str, help="Image folder")
    parser.add_argument(
        "-q", "--qps", type=str, default="27,32,37,42,47,52",
        help="Comma-separated QP list (default: %(default)s)",
    )
    parser.add_argument("-b", "--build-dir", type=str, required=True, help="VTM binary dir")
    parser.add_argument("-c", "--config", type=str, required=True, help="VTM config file")
    parser.add_argument(
        "-j", "--num-jobs", type=int, default=1,
        help="Parallel encode/decode jobs (default: %(default)s)",
    )
    parser.add_argument(
        "--recon_dir", type=str, default=None,
        help="Directory to save reconstructed images (default: <output>_recon)",
    )
    parser.add_argument(
        "--output_json", type=str, required=True, help="Output results JSON path",
    )
    parser.add_argument("--rgb", action="store_true", help="Use RGB color space (over YCbCr)")
    parser.add_argument(
        "--bitdepth", type=int, choices=[8, 10], default=8,
        help="Input/output bit-depth (default: %(default)s)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for perceptual metrics (default: %(default)s)",
    )
    parser.add_argument(
        "--skip_encode", action="store_true",
        help="Skip encode/decode; only compute metrics on existing recon_dir",
    )
    return parser.parse_args()


def list_images(dataset):
    root = Path(dataset)
    if not root.is_dir():
        raise OSError(f"No such directory: {dataset}")
    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No images found in: {dataset}")
    return files


def _encode_one(payload):
    """Worker: run VTM on one (image, qp), save recon PNG, return bpp stats."""
    (
        img_path, qp, recon_path, build_dir, config, rgb, bitdepth,
    ) = payload
    codec = VTM(
        SimpleNamespace(
            build_dir=build_dir,
            config=config,
            rgb=rgb,
            bitdepth=bitdepth,
        )
    )
    info, rec = codec.run(str(img_path), qp, metrics=[], return_rec=True)
    os.makedirs(os.path.dirname(recon_path), exist_ok=True)
    rec.save(recon_path)
    return {
        "img_name": Path(img_path).name,
        "qp": qp,
        "bpp": info["bpp"],
        "encoding_time": info["encoding_time"],
        "decoding_time": info["decoding_time"],
        "recon_path": recon_path,
        "src_path": str(img_path),
    }


def encode_all(image_files, qps, args, recon_root):
    tasks = []
    for qp in qps:
        qp_dir = os.path.join(recon_root, f"qp{qp}")
        os.makedirs(qp_dir, exist_ok=True)
        for img_path in image_files:
            recon_path = os.path.join(qp_dir, f"{img_path.stem}.png")
            tasks.append(
                (
                    str(img_path),
                    qp,
                    recon_path,
                    args.build_dir,
                    args.config,
                    args.rgb,
                    args.bitdepth,
                )
            )

    print(f"Encoding {len(tasks)} (image, qp) pairs with {args.num_jobs} jobs ...")
    if args.num_jobs > 1:
        with mp.Pool(args.num_jobs) as pool:
            results = list(
                tqdm(pool.imap_unordered(_encode_one, tasks), total=len(tasks), desc="VTM")
            )
    else:
        results = [_encode_one(t) for t in tqdm(tasks, desc="VTM")]
    return results


def collect_existing(image_files, qps, recon_root):
    """Build result stubs from already-saved recon images (bpp unknown -> None)."""
    results = []
    for qp in qps:
        qp_dir = os.path.join(recon_root, f"qp{qp}")
        for img_path in image_files:
            recon_path = os.path.join(qp_dir, f"{img_path.stem}.png")
            if not os.path.isfile(recon_path):
                raise FileNotFoundError(f"Missing recon image: {recon_path}")
            results.append(
                {
                    "img_name": img_path.name,
                    "qp": qp,
                    "bpp": None,
                    "encoding_time": None,
                    "decoding_time": None,
                    "recon_path": recon_path,
                    "src_path": str(img_path),
                }
            )
    return results


def compute_metrics_all(encode_results, device):
    # Lazy import so encode workers (spawn) do not load pyiqa.
    from perceptual_metric import PerceptualQualityMetrics

    metrics = PerceptualQualityMetrics(device)
    to_tensor = transforms.ToTensor()
    summary_acc = defaultdict(lambda: defaultdict(list))

    for item in tqdm(encode_results, desc="Metrics"):
        src = Image.open(item["src_path"]).convert("RGB")
        rec = Image.open(item["recon_path"]).convert("RGB")
        gt = to_tensor(src).unsqueeze(0).to(device)
        hat = to_tensor(rec).unsqueeze(0).to(device)
        scores = metrics.evaluate(hat, gt)

        qp = item["qp"]
        entry = {
            "psnr": scores["psnr"],
            "ms_ssim": scores["ms_ssim"],
            "lpips": scores["lpips"],
            "dists": scores["dists"],
            "bpp": item["bpp"],
        }
        for key, value in entry.items():
            if value is not None:
                summary_acc[qp][key].append(value)

    return {
        qp: {
            key: sum(acc[key]) / len(acc[key])
            for key in ("psnr", "ms_ssim", "lpips", "dists", "bpp")
        }
        for qp, acc in sorted(summary_acc.items())
    }


def main():
    args = parse_args()
    qps = sorted(int(q) for q in args.qps.split(",") if q)
    image_files = list_images(args.dataset)

    recon_root = args.recon_dir
    if recon_root is None:
        stem = os.path.splitext(args.output_json)[0]
        recon_root = f"{stem}_recon"
    os.makedirs(recon_root, exist_ok=True)

    if args.skip_encode:
        encode_results = collect_existing(image_files, qps, recon_root)
    else:
        encode_results = encode_all(image_files, qps, args, recon_root)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"Computing perceptual metrics on {device} ...")
    summary = compute_metrics_all(encode_results, device)

    output = {
        "name": f"vtm_perceptual_{os.path.basename(os.path.normpath(args.dataset))}",
        "dataset": os.path.basename(os.path.normpath(args.dataset)),
        "qps": qps,
        "results": {
            key: [summary[qp][key] for qp in qps]
            for key in ("bpp", "psnr", "ms_ssim", "lpips", "dists")
        },
    }

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output_json}")
    print(f"Recon images under: {recon_root}")
    print("\n==== Summary ====")
    for qp in qps:
        r = summary[qp]
        print(
            f"QP {qp}: bpp={r['bpp']:.6f}  PSNR={r['psnr']:.4f}  "
            f"MS-SSIM={r['ms_ssim']:.6f}  LPIPS={r['lpips']:.6f}  "
            f"DISTS={r['dists']:.6f}"
        )


if __name__ == "__main__":
    # Required for CUDA + fork-safe multiprocessing on Linux.
    mp.set_start_method("spawn", force=True)
    main()
