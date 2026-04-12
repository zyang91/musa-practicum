"""
run_pipeline.py
Orchestrate the full deployment pipeline for all target neighborhoods.

Usage:
    # Run all neighborhoods (Center City, Kensington, Point Breeze)
    python scripts/deployment/run_pipeline.py

    # Run a specific neighborhood
    python scripts/deployment/run_pipeline.py --neighborhoods "Center City"

    # Skip download step (if tiles already exist)
    python scripts/deployment/run_pipeline.py --skip-download

    # Run on GPU with larger batch size
    python scripts/deployment/run_pipeline.py --batch-size 32

Steps per neighborhood:
    1. Download aerial imagery tiles from PASDA
    2. Mosaic tiles and clip to neighborhood boundary
    3. Run UNet sliding-window prediction
    4. Post-process and vectorize to crosswalk polygons
"""

import subprocess
import sys
import argparse
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "deployment"
SHP_DIR = PROJECT_ROOT / "shp"

# Neighborhoods to deploy (name → shapefile stem)
NEIGHBORHOODS = {
    "Center_City": "Center City",
    "KENSINGTON": "KENSINGTON",
    "POINT_BREEZE": "POINT_BREEZE",
}


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
    parser = argparse.ArgumentParser(description="Run deployment pipeline for all neighborhoods")
    parser.add_argument("--neighborhoods", nargs="*", default=None,
                        help="Specific neighborhood names to process (default: all)")
    parser.add_argument("--skip-download", action="store_true", default=True,
                        help="Skip tile download — manual download is default (use existing tiles)")
    parser.add_argument("--skip-mosaic", action="store_true",
                        help="Skip mosaic step (use existing mosaics)")
    parser.add_argument("--skip-predict", action="store_true",
                        help="Skip prediction step")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Inference batch size (default: 16)")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Binary threshold (default: 0.45)")
    args = parser.parse_args()

    # Filter neighborhoods
    if args.neighborhoods:
        targets = {k: v for k, v in NEIGHBORHOODS.items()
                   if k in args.neighborhoods or v in args.neighborhoods}
        if not targets:
            print(f"No matching neighborhoods found. Available: {list(NEIGHBORHOODS.keys())}")
            sys.exit(1)
    else:
        targets = NEIGHBORHOODS

    print(f"Target neighborhoods: {list(targets.keys())}")
    print(f"Project root: {PROJECT_ROOT}")

    py = sys.executable
    failed = []

    for name, shp_stem in targets.items():
        shp_path = SHP_DIR / f"{shp_stem}.shp"
        if not shp_path.exists():
            print(f"\n✗ Shapefile not found: {shp_path}")
            failed.append(name)
            continue

        print(f"\n\n{'#'*60}")
        print(f"#  NEIGHBORHOOD: {name}")
        print(f"{'#'*60}")

        # Step 1: Download
        if not args.skip_download:
            ok = run_step(
                f"[{name}] Download imagery tiles",
                [py, str(SCRIPT_DIR / "01_download_imagery.py"),
                 "--shp", str(shp_path), "--name", name],
            )
            if not ok:
                failed.append(name)
                continue

        # Step 2: Mosaic & clip
        if not args.skip_mosaic:
            ok = run_step(
                f"[{name}] Mosaic and clip",
                [py, str(SCRIPT_DIR / "02_mosaic_and_clip.py"),
                 "--shp", str(shp_path), "--name", name],
            )
            if not ok:
                failed.append(name)
                continue

        # Step 3: Predict
        if not args.skip_predict:
            ok = run_step(
                f"[{name}] UNet prediction",
                [py, str(SCRIPT_DIR / "03_predict.py"),
                 "--name", name,
                 "--threshold", str(args.threshold),
                 "--batch-size", str(args.batch_size)],
            )
            if not ok:
                failed.append(name)
                continue

        # Step 4: Post-process
        ok = run_step(
            f"[{name}] Post-process and vectorize",
            [py, str(SCRIPT_DIR / "04_postprocess.py"), "--name", name],
        )
        if not ok:
            failed.append(name)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")
    for name in targets:
        status = "✗ FAILED" if name in failed else "✓ OK"
        print(f"  {name:20s} {status}")

    if failed:
        print(f"\n{len(failed)} neighborhood(s) failed.")
        sys.exit(1)
    else:
        print("\nAll neighborhoods completed successfully.")


if __name__ == "__main__":
    main()
