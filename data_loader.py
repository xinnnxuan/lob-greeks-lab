"""
Load SPX/ES loaded LOB data from double-gzipped JSON session files.
"""
from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path

import pandas as pd

GDRIVE_FOLDER_ID = "1j90YqlYwaeoV32ksz2-ruz11wxA9GNtl"
_DOWNLOAD_DONE = False


def _gdrive_file_ids(folder_id: str) -> list[tuple[str, str]]:
    """Return [(file_id, filename), ...] for files in a public Drive folder."""
    import re
    import urllib.request
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        html = r.read().decode("utf-8", errors="ignore")
    # Extract file IDs and names from the folder page JSON blob
    pairs = re.findall(r'"([\w-]{28,})".*?"([^"]+\.csv\.gz)"', html)
    return [(fid, name) for fid, name in pairs if name.startswith("day_")]


def _download_from_gdrive(data_dir: Path) -> None:
    """Download day_*.csv.gz files from Google Drive, once per process."""
    global _DOWNLOAD_DONE
    if _DOWNLOAD_DONE:
        return
    _DOWNLOAD_DONE = True

    existing = [f for f in data_dir.glob("day_*.csv.gz") if f.stat().st_size > 1000]
    if existing:
        return

    try:
        import gdown
    except ImportError:
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try listing individual day files and downloading each one
        pairs = _gdrive_file_ids(GDRIVE_FOLDER_ID)
    except Exception:
        pairs = []

    if pairs:
        for file_id, name in pairs:
            dest = data_dir / name
            if dest.exists() and dest.stat().st_size > 1000:
                continue
            try:
                gdown.download(id=file_id, output=str(dest), quiet=False, fuzzy=True)
            except Exception:
                pass
    else:
        # Fall back to full folder download and clean up non-day files
        try:
            gdown.download_folder(
                id=GDRIVE_FOLDER_ID,
                output=str(data_dir),
                quiet=False,
                use_cookies=False,
            )
            for f in data_dir.iterdir():
                if f.is_file() and not f.name.startswith("day_"):
                    f.unlink(missing_ok=True)
        except Exception:
            pass


def _mbo_depth(val) -> float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0
    if isinstance(val, list):
        return float(sum(val))
    if isinstance(val, str):
        s = val.strip()
        if not s or s == "[]":
            return 0.0
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return float(sum(parsed))
        except (ValueError, SyntaxError):
            pass
    return 0.0


def load_session_gz(path: str | Path) -> pd.DataFrame:
    """Double-gzip JSON array (string-encoded JSON) -> DataFrame."""
    path = Path(path)
    raw = path.read_bytes()
    inner = gzip.decompress(gzip.decompress(raw))
    s = json.loads(inner)
    if isinstance(s, str):
        records = json.loads(s)
    else:
        records = s
    df = pd.DataFrame(records)
    if "MBO" in df.columns:
        def ff(x):
            return float(sum(x)) if isinstance(x, list) else _mbo_depth(x)
        df["MBO_depth"] = df["MBO"].apply(ff)
    else:
        df["MBO_depth"] = 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def list_session_files(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    return sorted(data_dir.glob("loaded_lob_*.csv.gz"))


def group_files_by_date(data_dir: str | Path) -> dict[str, list[Path]]:
    """
    Return an ordered dict mapping date string (YYYYMMDD) -> sorted list of .csv.gz Paths.
    Expects filenames like loaded_lob_YYYYMMDD__YYYYMMDD_HHMM.csv.gz.
    """
    groups: dict[str, list[Path]] = {}
    for f in list_session_files(data_dir):
        # stem without the trailing .csv: loaded_lob_20250414__20250414_1015
        stem = f.name.replace(".csv.gz", "")
        parts = stem.split("__")
        date = parts[0].replace("loaded_lob_", "")
        groups.setdefault(date, []).append(f)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def load_sessions_concat(paths: tuple[str, ...]) -> pd.DataFrame:
    """Load and concatenate multiple session files, sorted by timestamp."""
    frames = [load_session_gz(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def load_merged_day(path: str | Path) -> pd.DataFrame:
    """Load a pre-merged daily CSV.gz produced by merge_daily.py."""
    import io as _io
    try:
        df = pd.read_csv(path, compression="gzip")
    except EOFError:
        # Truncated gzip — recover whatever bytes decompressed successfully.
        chunks: list[bytes] = []
        with open(path, "rb") as f:
            gz = gzip.GzipFile(fileobj=f)
            while True:
                try:
                    chunk = gz.read(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                except EOFError:
                    break
        df = pd.read_csv(_io.BytesIO(b"".join(chunks)), on_bad_lines="skip")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if "MBO_depth" not in df.columns:
        df["MBO_depth"] = 0.0
    return df


def list_merged_days(data_dir: str | Path) -> dict[str, Path]:
    """Return date -> Path for any pre-merged day_YYYYMMDD.csv.gz files."""
    data_dir = Path(data_dir)
    return {
        p.name.replace("day_", "").replace(".csv.gz", ""): p
        for p in sorted(data_dir.glob("day_*.csv.gz"))
    }


def default_data_dir() -> Path:
    d = Path(__file__).resolve().parent / "data"
    _download_from_gdrive(d)
    return d
