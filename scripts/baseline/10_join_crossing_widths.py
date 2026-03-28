import pandas as pd
import geopandas as gpd
from pathlib import Path

# paths
approaches_path = "../../data/vector/approach_segments.gpkg"
widths_path = "../../data/output/crossing_widths_baseline.csv"
output_dir = Path("../../data/output")
output_dir.mkdir(parents=True, exist_ok=True)

# read
approaches = gpd.read_file(approaches_path)
widths = pd.read_csv(widths_path)

print("Loaded approaches:", len(approaches))
print("Loaded width records:", len(widths))

# join
joined = approaches.merge(
    widths,
    on=["approach_id", "intersection_id"],
    how="left"
)

# optional unit conversion
joined["crossing_width_m"] = joined["crossing_width_ft"] * 0.3048
joined["width_iqr_m"] = joined["width_iqr_ft"] * 0.3048

# simple QC flags
joined["qc_flag"] = "ok"
joined.loc[joined["crossing_width_ft"].isna(), "qc_flag"] = "missing"
joined.loc[joined["crossing_width_ft"] < 8, "qc_flag"] = "too_small"
joined.loc[joined["crossing_width_ft"] > 80, "qc_flag"] = "too_large"
joined.loc[joined["valid_samples"] < 3, "qc_flag"] = "few_samples"

# save full output
out_full = output_dir / "approach_crossing_widths_baseline.gpkg"
joined.to_file(out_full, driver="GPKG")

# save a filtered version that is easier to inspect
filtered = joined[
    joined["crossing_width_ft"].notna() &
    (joined["crossing_width_ft"] >= 8) &
    (joined["crossing_width_ft"] <= 80) &
    (joined["valid_samples"] >= 3)
].copy()

out_filtered = output_dir / "approach_crossing_widths_baseline_filtered.gpkg"
filtered.to_file(out_filtered, driver="GPKG")

print("\nSaved:")
print(out_full)
print(out_filtered)

print("\nQC summary:")
print(joined["qc_flag"].value_counts(dropna=False))

print("\nFiltered count:", len(filtered))
print("\nCrossing width summary (filtered):")
print(filtered["crossing_width_ft"].describe())