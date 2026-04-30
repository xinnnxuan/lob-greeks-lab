"""
Short-horizon forward realized volatility from SPX path + LOB depth + ATM call Greeks.

Target at time t: sqrt(sum of squared log returns over the next H snapshot-to-snapshot steps).
All features use only information available at or before t (no leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def _side_bid_mask(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    return s.str.contains("bid", na=False)


def _side_ask_mask(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    return s.str.contains("ask", na=False)


def _safe_log_returns(price: pd.Series) -> pd.Series:
    p = pd.to_numeric(price, errors="coerce")
    out = np.log(p / p.shift(1))
    return out.replace([np.inf, -np.inf], np.nan)


def _pick_return_price(spx: pd.Series, es: pd.Series | None) -> tuple[pd.Series, str]:
    """
    Many LOB snapshots fix spx_price for the whole session; ES ref often still moves.
    Prefer the series with meaningful step-to-step variation for log returns.
    """
    spx = pd.to_numeric(spx, errors="coerce")
    lr_s = _safe_log_returns(spx)
    spx_move = float(lr_s.abs().max(skipna=True)) if len(lr_s) else 0.0
    spx_nu = int(spx.nunique(dropna=True))

    if es is None or es.isna().all():
        return spx, "SPX (spx_price)"

    es = pd.to_numeric(es, errors="coerce")
    lr_e = _safe_log_returns(es)
    es_move = float(lr_e.abs().max(skipna=True)) if len(lr_e) else 0.0
    es_nu = int(es.nunique(dropna=True))

    spx_flat = spx_move < 1e-12 or spx_nu <= 2
    if spx_flat and (es_move > 1e-12 or es_nu > spx_nu):
        return es, "ES ref (current_es_price) — SPX was flat across snapshots in this file"

    if es_move > spx_move * 1.5 and es_nu >= spx_nu:
        return es, "ES ref (current_es_price) — stronger variation than SPX for this session"

    return spx, "SPX (spx_price)"


def build_snapshot_panel(df: pd.DataFrame) -> pd.DataFrame:
    """One row per timestamp: SPX, bid/ask depth, ATM call delta/vega."""
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()

    d = df[df["timestamp"].notna()].copy()
    if d.empty:
        return pd.DataFrame()

    ts_first = d.groupby("timestamp", sort=True)["spx_price"].first()
    es_first = (
        d.groupby("timestamp", sort=True)["current_es_price"].first()
        if "current_es_price" in d.columns
        else None
    )
    bid = d.loc[_side_bid_mask(d["Side"])].groupby("timestamp")["MBO_depth"].sum()
    ask = d.loc[_side_ask_mask(d["Side"])].groupby("timestamp")["MBO_depth"].sum()

    panel = pd.DataFrame({"spx": ts_first, "bid_depth": bid, "ask_depth": ask})
    if es_first is not None:
        panel["es_ref"] = es_first.reindex(panel.index)
    else:
        panel["es_ref"] = np.nan
    panel["bid_depth"] = panel["bid_depth"].fillna(0.0)
    panel["ask_depth"] = panel["ask_depth"].fillna(0.0)
    panel["total_depth"] = panel["bid_depth"] + panel["ask_depth"]
    dsum = panel["total_depth"].replace(0, np.nan)
    panel["depth_imbalance"] = (panel["bid_depth"] - panel["ask_depth"]) / dsum
    panel["depth_imbalance"] = panel["depth_imbalance"].fillna(0.0)

    atm_rows: list[dict] = []
    for ts, g in d.groupby("timestamp", sort=True):
        spx = float(g["spx_price"].iloc[0])
        u = g.drop_duplicates(subset=["future_strike"], keep="first")
        if u.empty:
            atm_rows.append({"timestamp": ts, "atm_call_delta": np.nan, "atm_call_vega": np.nan})
            continue
        strikes = pd.to_numeric(u["future_strike"], errors="coerce")
        if strikes.notna().sum() == 0:
            atm_rows.append({"timestamp": ts, "atm_call_delta": np.nan, "atm_call_vega": np.nan})
            continue
        idx = (strikes - spx).abs().idxmin()
        row = u.loc[idx]
        cd = row["call_delta"] if "call_delta" in row.index and pd.notna(row.get("call_delta")) else np.nan
        cv = row["call_vega"] if "call_vega" in row.index and pd.notna(row.get("call_vega")) else np.nan
        atm_rows.append(
            {
                "timestamp": ts,
                "atm_call_delta": float(cd) if pd.notna(cd) else np.nan,
                "atm_call_vega": float(cv) if pd.notna(cv) else np.nan,
            }
        )
    atm_df = pd.DataFrame(atm_rows).set_index("timestamp")
    panel["atm_call_delta"] = atm_df["atm_call_delta"].reindex(panel.index)
    panel["atm_call_vega"] = atm_df["atm_call_vega"].reindex(panel.index)

    panel = panel.sort_index()
    ret_px, note = _pick_return_price(panel["spx"], panel["es_ref"])
    panel["return_price"] = ret_px
    panel["log_ret"] = _safe_log_returns(panel["return_price"])
    panel.attrs["return_price_note"] = note
    return panel


def forward_realized_vol(log_ret: pd.Series, h: int) -> pd.Series:
    """
    At index i, sqrt(sum of squared log returns from i+1 .. i+h inclusive).
    Last h rows are NaN.
    """
    r = log_ret.to_numpy(dtype=float)
    n = len(r)
    out = np.full(n, np.nan)
    for i in range(n - h):
        window = r[i + 1 : i + 1 + h]
        if np.isfinite(window).all():
            out[i] = float(np.sqrt(np.sum(window**2)))
    return pd.Series(out, index=log_ret.index)


def past_realized_vol(log_ret: pd.Series, w: int) -> pd.Series:
    """sqrt(sum of squared returns over previous w steps, ending at t)."""
    r = log_ret.to_numpy(dtype=float)
    n = len(r)
    out = np.full(n, np.nan)
    for i in range(w, n):
        window = r[i - w : i]
        if np.isfinite(window).all():
            out[i] = float(np.sqrt(np.sum(window**2)))
    return pd.Series(out, index=log_ret.index)


def build_feature_matrix(panel: pd.DataFrame, lookback: int, horizon: int) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if panel.shape[0] < lookback + horizon + 5:
        return pd.DataFrame(), pd.Series(dtype=float), []

    y = forward_realized_vol(panel["log_ret"], horizon)
    past_rv = past_realized_vol(panel["log_ret"], lookback)

    feat = pd.DataFrame(
        {
            "past_rv": past_rv,
            "lag1_ret": panel["log_ret"].shift(1),
            "lag2_ret": panel["log_ret"].shift(2),
            "log_total_depth": np.log1p(panel["total_depth"]),
            "depth_imbalance": panel["depth_imbalance"],
            "atm_abs_delta": panel["atm_call_delta"].abs(),
            "atm_vega": panel["atm_call_vega"],
        },
        index=panel.index,
    )
    feat["target"] = y
    feat = feat.dropna()
    if feat.shape[0] < 20:
        return pd.DataFrame(), pd.Series(dtype=float), []

    y_clean = feat["target"]
    X = feat.drop(columns=["target"])
    names = list(X.columns)
    return X, y_clean, names


# ── IV Smile ────────────────────────────────────────────────────────────────

def _iv_from_call_delta(call_delta: float, log_s_over_k: float, t: float) -> float:
    """
    Back out annualized BSM IV from call delta using the quadratic formula.

    BSM: d1 = (ln(S/K) + σ²t/2) / (σ√t)
    Setting u = σ√t gives: u²/2 - d1·u + ln(S/K) = 0
    => u = d1 ± √(d1² - 2·ln(S/K));  σ = u / √t
    """
    if not (0.002 < call_delta < 0.998) or t <= 0:
        return float("nan")
    d1 = float(norm.ppf(call_delta))
    discriminant = d1 ** 2 - 2.0 * log_s_over_k
    if discriminant < 0:
        return float("nan")
    u = d1 + discriminant ** 0.5
    if u <= 0:
        u = d1 - discriminant ** 0.5
    if u <= 0:
        return float("nan")
    return u / (t ** 0.5)


def build_iv_smile(snap: pd.DataFrame) -> pd.DataFrame:
    """
    Compute call and put IV for every unique strike in a single snapshot.
    Returns a DataFrame sorted by strike with columns:
      future_strike, moneyness (ln K/S), call_iv, put_iv.
    """
    if snap.empty or "call_delta" not in snap.columns:
        return pd.DataFrame()

    spx = float(snap["spx_price"].iloc[0])
    t = float(snap["t"].iloc[0])
    if spx <= 0 or t <= 0:
        return pd.DataFrame()

    rows: list[dict] = []
    sub = snap.drop_duplicates(subset=["future_strike"], keep="first")
    for _, row in sub.iterrows():
        K = float(row["future_strike"])
        if K <= 0:
            continue
        log_s_k = float(np.log(spx / K))  # ln(S/K)
        call_iv = put_iv = float("nan")

        cd = row.get("call_delta")
        if pd.notna(cd):
            call_iv = _iv_from_call_delta(float(cd), log_s_k, t)

        pd_ = row.get("put_delta")
        if pd.notna(pd_):
            # put_delta = -N(-d1)  =>  N(d1) = 1 + put_delta
            put_iv = _iv_from_call_delta(float(pd_) + 1.0, log_s_k, t)

        rows.append(
            {
                "future_strike": K,
                "moneyness": float(np.log(K / spx)),
                "call_iv": call_iv,
                "put_iv": put_iv,
            }
        )

    return pd.DataFrame(rows).sort_values("future_strike").reset_index(drop=True)


# ── Imbalance signal ─────────────────────────────────────────────────────────

def build_imbalance_signal(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    For each snapshot, pair depth_imbalance with the cumulative log return
    over the next `horizon` steps.  Returns rows with:
      timestamp, imbalance, forward_return, bucket.
    """
    if panel.empty or "depth_imbalance" not in panel.columns:
        return pd.DataFrame()

    lr = panel["log_ret"].to_numpy(dtype=float)
    n = len(lr)
    fwd = np.full(n, np.nan)
    for i in range(n - horizon):
        window = lr[i + 1 : i + 1 + horizon]
        if np.isfinite(window).all():
            fwd[i] = float(np.sum(window))

    out = pd.DataFrame(
        {
            "timestamp": panel.index,
            "imbalance": panel["depth_imbalance"].values,
            "forward_return": fwd,
        }
    ).dropna()

    if out.empty:
        return out

    bins = [-1.01, -0.3, -0.1, 0.1, 0.3, 1.01]
    labels = ["Strongly Bearish (<-0.30)", "Bearish (-0.30–-0.10)", "Neutral (-0.10–0.10)", "Bullish (0.10–0.30)", "Strongly Bullish (>0.30)"]
    out["bucket"] = pd.cut(out["imbalance"], bins=bins, labels=labels)
    return out


def train_vol_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[GradientBoostingRegressor, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time-ordered split: last fraction held out as test."""
    if len(X) < 30:
        raise ValueError("Not enough rows to train.")

    n_test = max(int(len(X) * test_size), 5)
    n_train = len(X) - n_test
    if n_train < 15:
        raise ValueError("Not enough training rows.")

    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    model = GradientBoostingRegressor(
        random_state=random_state,
        max_depth=3,
        n_estimators=120,
        learning_rate=0.08,
        subsample=0.9,
    )
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    def _r2(y_true, y_pred) -> float:
        yt = np.asarray(y_true, dtype=float)
        if not np.isfinite(np.var(yt)) or np.var(yt) < 1e-20:
            return float("nan")
        return float(r2_score(y_true, y_pred))

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "train_r2": _r2(y_train, pred_train),
        "test_r2": _r2(y_test, pred_test),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "target_std_train": float(np.std(y_train)),
        "target_std_test": float(np.std(y_test)),
    }
    return model, metrics, y_train.values, pred_train, y_test.values, pred_test
