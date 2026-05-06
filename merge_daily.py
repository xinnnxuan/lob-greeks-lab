"""
One-time script: merge all per-minute loaded_lob_*.csv.gz files into
one file per trading day: data/day_YYYYMMDD.csv.gz

Run once from the project root:
    python merge_daily.py
"""
from __future__ import annotations

import gc
import subprocess
import time
from pathlib import Path

import pandas as pd

from data_loader import default_data_dir, group_files_by_date, load_session_gz

DOWNLOAD_TIMEOUT = 60  # seconds to wait per file for iCloud download
CHUNK_SIZE = 50  # files to hold in memory at once before flushing to disk


def _ensure_local(f: Path) -> bool:
    """Trigger iCloud download and wait until file is local."""
    try:
        if f.stat().st_size > 1000:
            return True
    except OSError:
        return False
    try:
        subprocess.run(["brctl", "download", str(f)], capture_output=True, check=False)
    except FileNotFoundError:
        return False
    for _ in range(DOWNLOAD_TIMEOUT):
        time.sleep(1)
        try:
            if f.stat().st_size > 1000:
                return True
        except OSError:
            pass
    return False


def _evict(f: Path) -> None:
    """Evict file back to iCloud to free local disk space."""
    try:
        subprocess.run(["brctl", "evict", str(f)], capture_output=True, check=False)
    except FileNotFoundError:
        pass


def merge_day(date: str, files: list[Path], out_dir: Path) -> Path:
    out_path = out_dir / f"day_{date}.csv.gz"
    if out_path.exists():
        print(f"  {out_path.name} already exists, skipping.")
        return out_path

    print(f"  Merging {len(files)} files for {date}...")
    chunk_paths: list[Path] = []

    # Process in chunks to avoid loading all files into RAM at once
    for chunk_start in range(0, len(files), CHUNK_SIZE):
        chunk_files = files[chunk_start:chunk_start + CHUNK_SIZE]
        frames = []
        for i, f in enumerate(chunk_files, chunk_start + 1):
            try:
                _ensure_local(f)
                frames.append(load_session_gz(f))
                _evict(f)
            except Exception as e:
                print(f"    Warning: skipping {f.name} ({e})")
            if i % 20 == 0:
                print(f"    {i}/{len(files)} loaded...")

        if not frames:
            continue

        chunk_df = pd.concat(frames, ignore_index=True)
        del frames
        gc.collect()

        chunk_path = out_dir / f"_chunk_{date}_{chunk_start}.csv.gz"
        chunk_df.to_csv(chunk_path, index=False, compression="gzip")
        chunk_paths.append(chunk_path)
        del chunk_df
        gc.collect()
        print(f"    Flushed chunk {chunk_start}–{chunk_start + len(chunk_files) - 1} to disk.")

    if not chunk_paths:
        print(f"  No data for {date}, skipping.")
        return out_path

    print(f"  Combining {len(chunk_paths)} chunks and sorting...")
    merged = pd.concat(
        [pd.read_csv(p, compression="gzip") for p in chunk_paths],
        ignore_index=True,
    ).sort_values("timestamp").reset_index(drop=True)

    merged.to_csv(out_path, index=False, compression="gzip")
    del merged
    gc.collect()

    for p in chunk_paths:
        p.unlink(missing_ok=True)

    mb = out_path.stat().st_size / 1e6
    print(f"  Saved {out_path.name} ({mb:.1f} MB)")
    _evict(out_path)
    print(f"  Evicted {out_path.name} back to iCloud.")
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
