"""
Load SPX/ES loaded LOB data from CSV sample or double-gzipped JSON session files.
"""
from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path

import pandas as pd


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


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["MBO_depth"] = df["MBO"].apply(_mbo_depth)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


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


def default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"
