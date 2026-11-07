import argparse
import json
from pathlib import Path

import numpy as np

METRICS = {
    "psnr": False,
    "ms_ssim": False,
    "lpips": True,
    "dists": True,
}


def bd_rate(bpp_ref, quality_ref, bpp_test, quality_test):
    log_ref = np.log(bpp_ref)
    log_test = np.log(bpp_test)
    fit_ref = np.polyfit(quality_ref, log_ref, 3)
    fit_test = np.polyfit(quality_test, log_test, 3)
    lower = max(quality_ref.min(), quality_test.min())
    upper = min(quality_ref.max(), quality_test.max())
    if lower >= upper:
        raise ValueError("The RD curves do not overlap")
    int_ref = np.polyint(fit_ref)
    int_test = np.polyint(fit_test)
    integral_ref = np.polyval(int_ref, upper) - np.polyval(int_ref, lower)
    integral_test = np.polyval(int_test, upper) - np.polyval(int_test, lower)
    return float((np.exp((integral_test - integral_ref) / (upper - lower)) - 1) * 100)


def read_curve(path, metric):
    data = json.loads(Path(path).read_text())
    negate = METRICS[metric]
    results = data["results"]
    if metric not in results:
        raise ValueError(f"Metric {metric} is not available in {path}")
    bpp = results["bpp"]
    quality = results[metric]
    if not len(bpp) == len(quality) == len(data["qps"]):
        raise ValueError(f"QP, bpp, and {metric} lengths differ in {path}")
    if len(bpp) < 4:
        raise ValueError(f"At least four RD points are required in {path}")
    quality = -np.asarray(quality) if negate else np.asarray(quality)
    return np.asarray(bpp), quality


def main():
    parser = argparse.ArgumentParser(description="Calculate BD-Rate between two RD JSON files")
    parser.add_argument("reference")
    parser.add_argument("test")
    parser.add_argument("--metric", default="psnr", choices=[*METRICS, "all"])
    parser.add_argument("--output")
    args = parser.parse_args()

    metrics = list(METRICS) if args.metric == "all" else [args.metric]
    results = {}
    for metric in metrics:
        ref_bpp, ref_quality = read_curve(args.reference, metric)
        test_bpp, test_quality = read_curve(args.test, metric)
        results[metric] = bd_rate(ref_bpp, ref_quality, test_bpp, test_quality)
        print(f"{metric}: {results[metric]:.4f}%")
    if args.output:
        output = results if len(results) > 1 else results[metrics[0]]
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
