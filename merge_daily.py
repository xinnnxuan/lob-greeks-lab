"""
One-time script: merge all per-minute loaded_lob_*.csv.gz files into
one file per trading day: data/day_YYYYMMDD.csv.gz

Run once from the project root:
    python merge_daily.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from data_loader import default_data_dir, group_files_by_date, load_session_gz


def merge_day(date: str, files: list[Path], out_dir: Path) -> Path:
    out_path = out_dir / f"day_{date}.csv.gz"
    if out_path.exists():
        print(f"  {out_path.name} already exists, skipping.")
        return out_path

    print(f"  Merging {len(files)} files for {date}...")
    frames = []
    for i, f in enumerate(files, 1):
        try:
            frames.append(load_session_gz(f))
        except Exception as e:
            print(f"    Warning: skipping {f.name} ({e})")
        if i % 20 == 0:
            print(f"    {i}/{len(files)} loaded...")

    if not frames:
        print(f"  No data for {date}, skipping.")
        return out_path

    merged = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    merged.to_csv(out_path, index=False, compression="gzip")
    mb = out_path.stat().st_size / 1e6
    print(f"  Saved {out_path.name} ({len(merged):,} rows, {mb:.1f} MB)")
    return out_path


def main():
    data_dir = default_data_dir()
    groups = group_files_by_date(data_dir)
    if not groups:
        print("No loaded_lob_*.csv.gz files found in data/.")
        return

    print(f"Found {sum(len(v) for v in groups.values())} files across {len(groups)} dates.\n")
    for date, files in groups.items():
        merge_day(date, files, data_dir)

    print("\nDone. You can now load full days instantly in the app.")


if __name__ == "__main__":
    main()
