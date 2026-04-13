"""
run_pipeline.py
Run predict + postprocess on all mosaic files in data/processed/.

Pre-processed mosaics (from ArcGIS) are placed in data/processed/:
    middle.tif, south.tif, upper-middle.tif

Usage:
    # Run all three mosaics
    python scripts/deployment/run_pipeline.py

    # Run a specific file
    python scripts/deployment/run_pipeline.py --inputs data/processed/middle.tif

    # Custom threshold and batch size
    python scripts/deployment/run_pipeline.py --threshold 0.40 --batch-size 32
"""

import subprocess
import sys
import argparse
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "deployment"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Default input mosaics (pre-processed in ArcGIS)
DEFAULT_INPUTS = [
    "middle.tif",
    "south.tif",
    "upper-middle.tif",
]


def run_step(description: str, cmd: list[str]) -> bool:
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\n  ✗ FAILED: {description}")
        return False
    print(f"  ✓ {description}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run deployment pipeline on mosaics")
    parser.add_argument("--inputs", nargs="*", default=None,
                        help="Specific mosaic .tif files (default: middle, south, upper-middle)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Inference batch size (default: 16)")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Binary threshold (default: 0.45)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()

    # Resolve input files
    if args.inputs:
        input_files = [Path(f) for f in args.inputs]
    else:
        input_files = [PROCESSED_DIR / f for f in DEFAULT_INPUTS]

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input files  : {[str(f) for f in input_files]}")

    py = sys.executable
    failed = []

    for input_file in input_files:
        stem = input_file.stem

        if not input_file.exists():
            print(f"\n✗ File not found: {input_file}")
            failed.append(stem)
            continue

        print(f"\n\n{'#'*60}")
        print(f"#  MOSAIC: {input_file.name}")
        print(f"{'#'*60}")

        # Step 1: Predict
        predict_cmd = [
            py, str(SCRIPT_DIR / "03_predict.py"),
            "--input", str(input_file),
            "--threshold", str(args.threshold),
            "--batch-size", str(args.batch_size),
        ]
        if args.output_dir:
            predict_cmd += ["--output-dir", args.output_dir]

        ok = run_step(f"[{stem}] UNet prediction", predict_cmd)
        if not ok:
            failed.append(stem)
            continue

        # Step 2: Post-process
        out_dir = Path(args.output_dir) if args.output_dir else input_file.parent
        bin_file = out_dir / f"{stem}_bin.tif"

        postprocess_cmd = [
            py, str(SCRIPT_DIR / "04_postprocess.py"),
            "--input", str(bin_file),
        ]
        if args.output_dir:
            postprocess_cmd += ["--output-dir", args.output_dir]

        ok = run_step(f"[{stem}] Post-process and vectorize", postprocess_cmd)
        if not ok:
            failed.append(stem)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")
    for f in input_files:
        status = "✗ FAILED" if f.stem in failed else "✓ OK"
        print(f"  {f.name:25s} {status}")

    if failed:
        print(f"\n{len(failed)} mosaic(s) failed.")
        sys.exit(1)
    else:
        print("\nAll mosaics completed successfully.")


if __name__ == "__main__":
    main()
