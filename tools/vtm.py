import argparse
import json
from pathlib import Path

COUNTS = {"kodak": 24, "clic": 41, "tecnick": 100}


def main():
    parser = argparse.ArgumentParser(description="Convert raw VTM output to RD JSON")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("dataset", choices=sorted(COUNTS))
    parser.add_argument("--qps", default="27,32,37,42,47,52")
    args = parser.parse_args()

    raw = json.loads(Path(args.input).read_text())["results"]
    qps = [int(qp) for qp in args.qps.split(",")]
    if not (len(qps) == len(raw["bpp"]) == len(raw["psnr-rgb"])):
        raise ValueError("QP, bpp, and PSNR lengths differ")
    output = {
        "name": f"vtm_{args.dataset}",
        "dataset": args.dataset,
        "qps": qps,
        "results": {"bpp": raw["bpp"], "psnr": raw["psnr-rgb"]},
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
