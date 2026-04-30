"""Streamlit: Options Microstructure & Greeks Lab."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import default_data_dir, group_files_by_date, load_sessions_concat
from vol_forecast import (
    build_feature_matrix,
    build_imbalance_signal,
    build_iv_smile,
    build_snapshot_panel,
    train_vol_model,
)

st.set_page_config(
    page_title="Options Microstructure & Greeks Lab",
    page_icon="📊",
    layout="wide",
)
DATA_DIR = default_data_dir()


@st.cache_data(show_spinner="Loading files…")
def _load_range_cached(paths: tuple[str, ...]) -> pd.DataFrame:
    return load_sessions_concat(paths)


def snapshot_at(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    return df[df["timestamp"] == ts].copy()


def _hhmm(path) -> str:
    """Extract HHMM from loaded_lob_YYYYMMDD__YYYYMMDD_HHMM.csv.gz."""
    stem = path.name.replace(".csv.gz", "")
    return stem.split("_")[-1]


def main():
    st.title("Options Microstructure & Greeks Lab")
    st.markdown(
        "**Focus:** Link **limit-order-book depth** (MBO) to **option Greeks** on ES/SPX. "
        "Load `loaded_lob_*.csv.gz` session files from `data/`."
    )
    with st.sidebar:
        st.header("Data")
        date_groups = group_files_by_date(DATA_DIR)
        if not date_groups:
            st.error("No loaded_lob_*.csv.gz in data/")
            st.stop()

        date_labels = {
            d: f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in date_groups
        }
        sel_date = st.selectbox(
            "Date",
            list(date_groups.keys()),
            format_func=lambda d: date_labels[d],
        )
        day_files = date_groups[sel_date]
        times = [_hhmm(f) for f in day_files]
        time_labels = [f"{t[:2]}:{t[2:]}" for t in times]

        if len(day_files) == 1:
            start_i, end_i = 0, 0
        else:
            start_i, end_i = st.select_slider(
                "Time window (UTC)",
                options=list(range(len(day_files))),
                value=(0, len(day_files) - 1),
                format_func=lambda i: time_labels[i],
            )

        selected_paths = tuple(str(f) for f in day_files[start_i : end_i + 1])
        n_files = len(selected_paths)
        st.caption(f"Loading {n_files} minute-file{'s' if n_files != 1 else ''} ({time_labels[start_i]}–{time_labels[end_i]} UTC).")
        choice = f"{sel_date}_{time_labels[start_i]}-{time_labels[end_i]}"

        df = _load_range_cached(selected_paths)
        ts_list = sorted(df["timestamp"].dropna().unique())
        if not ts_list:
            st.error("No valid timestamps in selected range.")
            st.stop()
        sel_ts = st.select_slider(
            "Snapshot time (UTC)",
            options=ts_list,
            value=ts_list[-1],
            format_func=lambda t: str(pd.Timestamp(t)),
        )
        snap = snapshot_at(df, pd.Timestamp(sel_ts))

    if snap.empty:
        st.warning("Empty snapshot.")
        st.stop()

    panel = build_snapshot_panel(df)

    meta = snap.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("SPX", f"{float(meta.get('spx_price', 0)):.2f}")
    with c2:
        st.metric("ES ref", f"{float(meta.get('current_es_price', 0)):,.0f}")
    with c3:
        st.metric("t (yrs)", f"{float(meta.get('t', 0)):.4f}")
    with c4:
        st.metric("Ladder rows", len(snap))

    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["Order book ladder", "Greeks vs strike", "Time series", "Volatility forecast", "IV Smile", "Imbalance Signal"]
    )
    with t1:
        st.subheader("MBO depth by strike")
        fig = px.bar(
            snap.sort_values(["Side", "future_strike"], ascending=[True, False]),
            x="MBO_depth",
            y="future_strike",
            color="Side",
            orientation="h",
            hover_data=["spx_strike", "MBO_pulling_stacking", "call_delta", "put_delta"],
        )
        fig.update_layout(height=700, yaxis_title="Future strike", xaxis_title="MBO depth")
        st.plotly_chart(fig, use_container_width=True)
        liq = snap[snap["MBO_depth"] > 0].sort_values("future_strike")
        cols = [c for c in ["Side", "future_strike", "MBO_depth", "call_delta", "call_vega", "put_delta", "put_vega"] if c in liq.columns]
        st.dataframe(liq[cols].head(80), use_container_width=True)

    with t2:
        sub = snap.drop_duplicates(subset=["future_strike"]).sort_values("future_strike")
        fig2 = go.Figure()
        for g in ["call_delta", "call_gamma", "call_vega", "call_theta"]:
            if g in sub.columns:
                fig2.add_trace(
                    go.Scatter(x=sub["future_strike"], y=sub[g], name=g.replace("call_", ""), mode="lines")
                )
        fig2.update_layout(title="Call Greeks", xaxis_title="Strike", height=480, hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)
        fig3 = go.Figure()
        for g in ["put_delta", "put_gamma", "put_vega", "put_theta"]:
            if g in sub.columns:
                fig3.add_trace(
                    go.Scatter(x=sub["future_strike"], y=sub[g], name=g.replace("put_", ""), mode="lines")
                )
        fig3.update_layout(title="Put Greeks", xaxis_title="Strike", height=480, hovermode="x unified")
        st.plotly_chart(fig3, use_container_width=True)

    with t3:
        st.subheader("Session dynamics")
        st.caption(
            "**ATM** curves use the **same rule as the Volatility forecast tab**: at each snapshot, pick the "
            "`future_strike` row closest to `spx_price`, then read call Δ and ν from that row."
        )
        if panel.empty:
            st.warning("Not enough data to build a per-timestamp panel.")
        else:
            pplot = panel.reset_index()
            fig4 = px.line(pplot, x="timestamp", y="spx", title="Spot through snapshots (SPX, first row per timestamp)")
            fig4.update_layout(height=380, xaxis_title="Time (UTC)")
            st.plotly_chart(fig4, use_container_width=True)

            if pplot["atm_call_delta"].notna().any():
                fig5 = px.line(
                    pplot,
                    x="timestamp",
                    y="atm_call_delta",
                    title="ATM call delta (rolling strike nearest spot)",
                )
                fig5.update_layout(height=380, xaxis_title="Time (UTC)", yaxis_title="Δ_call")
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("ATM call delta is all NaN for this session.")

            if "atm_call_vega" in pplot.columns and pplot["atm_call_vega"].notna().any():
                fig6 = px.line(
                    pplot,
                    x="timestamp",
                    y="atm_call_vega",
                    title="ATM call vega (same strike as Δ plot)",
                )
                fig6.update_layout(height=380, xaxis_title="Time (UTC)", yaxis_title="ν_call")
                st.plotly_chart(fig6, use_container_width=True)

        strikes_num = pd.to_numeric(df["future_strike"], errors="coerce")
        spx_num = pd.to_numeric(df["spx_price"], errors="coerce")
        if strikes_num.notna().any() and spx_num.notna().any() and "call_delta" in df.columns:
            med = float(spx_num.median())
            u = np.sort(np.unique(strikes_num.dropna().to_numpy()))
            if len(u) > 0:
                k = min(3, len(u))
                pick = u[np.argsort(np.abs(u - med))[:k]]
                pick = np.sort(pick)
                mask = np.isin(pd.to_numeric(df["future_strike"], errors="coerce"), pick)
                bundle = df.loc[mask].sort_values(["timestamp", "future_strike"])
                bundle = bundle.drop_duplicates(subset=["timestamp", "future_strike"], keep="first")
                if not bundle.empty and bundle["call_delta"].notna().any():
                    fig7 = px.line(
                        bundle,
                        x="timestamp",
                        y="call_delta",
                        color="future_strike",
                        title="Call delta at up to three strikes near session median spot",
                    )
                    fig7.update_layout(height=400, xaxis_title="Time (UTC)", legend_title="Strike")
                    st.plotly_chart(fig7, use_container_width=True)
                    st.caption(
                        f"Strikes shown: {', '.join(f'{float(s):g}' for s in pick)}. "
                        "These levels are **fixed** for the session (vs. median spot), unlike the rolling ATM series above."
                    )

    with t4:
        st.subheader("Short-horizon forward realized volatility")
        st.markdown(
            "**Target:** $\\sqrt{\\sum r_{t+1}^2 + \\cdots + r_{t+H}^2}$ over the next **H** snapshot steps using "
            "**log returns on the underlying price series** (see note below). "
            "**Features:** past **W**-step realized vol, lags, LOB depth / imbalance, ATM call |Δ| and vega. "
            "**Split:** time-ordered (last 20% test). For risk/execution intuition only—not trading advice."
        )
        c1, c2 = st.columns(2)
        with c1:
            lookback = st.slider(
                "Past window W (snapshots)",
                3,
                40,
                10,
                help="Used for backward realized vol and lags.",
                key=f"vol_W_{choice}",
            )
        with c2:
            horizon = st.slider(
                "Forward horizon H (snapshots)",
                2,
                30,
                8,
                help="How many future returns enter the target.",
                key=f"vol_H_{choice}",
            )

        note = panel.attrs.get("return_price_note", "")
        if note:
            st.info(note)

        X, y, feat_names = build_feature_matrix(panel, lookback, horizon)
        if X.empty:
            st.warning(
                "Not enough clean snapshots for this W/H (need more rows after dropping NaNs). "
                "Try smaller W or H, or load a longer session."
            )
        elif float(y.std()) < 1e-14:
            st.error(
                "Forward realized volatility is **constant (zero)** for this session: the underlying price used for "
                "returns does not move between snapshots. The model cannot learn anything useful. "
                "Try data where `spx_price` or `current_es_price` changes across timestamps."
            )
        else:
            try:
                model, metrics, y_tr, p_tr, y_te, p_te = train_vol_model(X, y, test_size=0.2)
            except ValueError as e:
                st.error(str(e))
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Test MAE", f"{metrics['test_mae']:.6e}")
                tr2 = metrics["test_r2"]
                m2.metric("Test R²", "N/A" if tr2 != tr2 else f"{tr2:.3f}")
                m3.metric("Train rows", metrics["n_train"])
                m4.metric("Test rows", metrics["n_test"])

                fig_f = go.Figure()
                fig_f.add_trace(
                    go.Scatter(
                        y=y_te,
                        x=p_te,
                        mode="markers",
                        name="Test",
                        marker=dict(size=8, opacity=0.65, color="#3B82F6"),
                    )
                )
                mx = max(float(np.nanmax(y_te)), float(np.nanmax(p_te)), 1e-12)
                mn = min(float(np.nanmin(y_te)), float(np.nanmin(p_te)), 0.0)
                pad = (mx - mn) * 0.05 + 1e-12
                fig_f.add_trace(
                    go.Scatter(
                        x=[mn - pad, mx + pad],
                        y=[mn - pad, mx + pad],
                        mode="lines",
                        name="Perfect",
                        line=dict(dash="dash", color="gray"),
                    )
                )
                fig_f.update_layout(
                    title="Test set: actual vs predicted forward realized vol",
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                    height=420,
                    xaxis=dict(tickformat=".2e", exponentformat="e"),
                    yaxis=dict(tickformat=".2e", exponentformat="e"),
                )
                st.plotly_chart(fig_f, use_container_width=True)

                imp = np.asarray(model.feature_importances_, dtype=float)
                imp_df = pd.DataFrame({"Importance": imp, "Feature": feat_names}).sort_values(
                    "Importance", ascending=True
                )
                fig_i = px.bar(
                    imp_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Gradient boosting feature importance",
                )
                fig_i.update_traces(marker_color="#3B82F6")
                fig_i.update_layout(
                    height=360,
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    xaxis=dict(range=[0, max(float(imp.max()), 1e-6) * 1.15]),
                )
                st.plotly_chart(fig_i, use_container_width=True)

    with t5:
        st.subheader("Implied Volatility Smile")
        st.markdown(
            "IV is back-calculated from each strike's **call delta** using the BSM quadratic: "
            r"given $\Delta_c = N(d_1)$, solve $\frac{u^2}{2} - d_1 u + \ln\frac{S}{K} = 0$ "
            r"where $u = \sigma\sqrt{t}$, yielding $\sigma = u / \sqrt{t}$. "
            r"Put IV uses $N(d_1) = 1 + \Delta_p$. "
            "Both curves should coincide by put-call parity; gaps reveal skew or data artifacts."
        )
        smile_df = build_iv_smile(snap)
        if smile_df.empty:
            st.warning("Cannot compute IV smile for this snapshot (check delta / t columns).")
        else:
            valid = smile_df.dropna(subset=["call_iv", "put_iv"], how="all")
            fig_smile = go.Figure()
            if valid["call_iv"].notna().any():
                fig_smile.add_trace(
                    go.Scatter(
                        x=valid["future_strike"],
                        y=valid["call_iv"] * 100,
                        mode="lines+markers",
                        name="Call IV",
                        line=dict(color="#3B82F6", width=2),
                        marker=dict(size=5),
                        hovertemplate="Strike %{x:.2f}<br>Call IV %{y:.2f}%<extra></extra>",
                    )
                )
            if valid["put_iv"].notna().any():
                fig_smile.add_trace(
                    go.Scatter(
                        x=valid["future_strike"],
                        y=valid["put_iv"] * 100,
                        mode="lines+markers",
                        name="Put IV",
                        line=dict(color="#F59E0B", width=2, dash="dot"),
                        marker=dict(size=5),
                        hovertemplate="Strike %{x:.2f}<br>Put IV %{y:.2f}%<extra></extra>",
                    )
                )
            spx_val = float(snap["spx_price"].iloc[0])
            fig_smile.add_vline(
                x=spx_val, line_dash="dash", line_color="gray",
                annotation_text=f"Spot {spx_val:.0f}", annotation_position="top right",
            )
            fig_smile.update_layout(
                xaxis_title="Strike",
                yaxis_title="Implied Volatility (%)",
                height=480,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_smile, use_container_width=True)

            st.markdown("#### IV vs Log-Moneyness ln(K/S)")
            st.caption(
                "Negative moneyness = in-the-money calls (K < S). "
                "The steeper the left tail, the stronger the put skew."
            )
            fig_mon = go.Figure()
            if valid["call_iv"].notna().any():
                fig_mon.add_trace(
                    go.Scatter(
                        x=valid["moneyness"],
                        y=valid["call_iv"] * 100,
                        mode="lines+markers",
                        name="Call IV",
                        line=dict(color="#3B82F6", width=2),
                        hovertemplate="ln(K/S) %{x:.4f}<br>IV %{y:.2f}%<extra></extra>",
                    )
                )
            if valid["put_iv"].notna().any():
                fig_mon.add_trace(
                    go.Scatter(
                        x=valid["moneyness"],
                        y=valid["put_iv"] * 100,
                        mode="lines+markers",
                        name="Put IV",
                        line=dict(color="#F59E0B", width=2, dash="dot"),
                        hovertemplate="ln(K/S) %{x:.4f}<br>IV %{y:.2f}%<extra></extra>",
                    )
                )
            fig_mon.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="ATM")
            fig_mon.update_layout(
                xaxis_title="Log-moneyness ln(K/S)",
                yaxis_title="Implied Volatility (%)",
                height=420,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_mon, use_container_width=True)
            st.dataframe(
                valid[["future_strike", "moneyness", "call_iv", "put_iv"]]
                .rename(columns={"call_iv": "Call IV", "put_iv": "Put IV", "moneyness": "ln(K/S)"})
                .assign(**{"Call IV": lambda d: (d["Call IV"] * 100).round(2),
                           "Put IV": lambda d: (d["Put IV"] * 100).round(2),
                           "ln(K/S)": lambda d: d["ln(K/S)"].round(5)})
                .reset_index(drop=True),
                use_container_width=True,
            )

    with t6:
        st.subheader("Order Book Imbalance as a Price Signal")
        st.markdown(
            "**Depth imbalance** = (bid depth − ask depth) / total depth. "
            "A strongly positive value means more liquidity is parked on the bid side (buyers defending). "
            "This tab tests whether that imbalance *predicts* the direction of the next price move."
        )
        c_h1, c_h2 = st.columns(2)
        with c_h1:
            imb_horizon = st.slider(
                "Forward horizon N (snapshots)",
                2, 30, 5,
                help="How many snapshot steps ahead to measure the cumulative log return.",
                key=f"imb_H_{choice}",
            )
        with c_h2:
            min_count = st.slider(
                "Min observations per bucket (filter)",
                1, 20, 3,
                key=f"imb_min_{choice}",
            )

        sig = build_imbalance_signal(panel, imb_horizon)
        if sig.empty:
            st.warning("Not enough snapshots to compute imbalance signal. Try a longer session or smaller horizon.")
        else:
            # ── scatter: imbalance vs forward return ─────────────────────────
            fig_sc = px.scatter(
                sig,
                x="imbalance",
                y="forward_return",
                color="bucket",
                opacity=0.65,
                trendline="ols",
                trendline_scope="overall",
                labels={"imbalance": "Depth Imbalance", "forward_return": f"Cumulative log-return (next {imb_horizon} steps)"},
                title=f"Imbalance vs {imb_horizon}-step forward cumulative log-return",
                hover_data={"timestamp": True},
            )
            fig_sc.update_layout(height=430, legend_title="Imbalance bucket")
            st.plotly_chart(fig_sc, use_container_width=True)

            # ── conditional mean bar chart ───────────────────────────────────
            bucket_stats = (
                sig.groupby("bucket", observed=True)["forward_return"]
                .agg(mean="mean", std="std", count="count")
                .reset_index()
            )
            bucket_stats = bucket_stats[bucket_stats["count"] >= min_count]
            if bucket_stats.empty:
                st.info("All buckets have fewer observations than the minimum filter. Lower the threshold.")
            else:
                bucket_stats["color"] = bucket_stats["mean"].apply(
                    lambda v: "#22C55E" if v > 0 else "#EF4444"
                )
                fig_bar = go.Figure()
                fig_bar.add_trace(
                    go.Bar(
                        x=bucket_stats["bucket"].astype(str),
                        y=bucket_stats["mean"] * 1e4,
                        error_y=dict(type="data", array=(bucket_stats["std"] * 1e4).tolist(), visible=True),
                        marker_color=bucket_stats["color"].tolist(),
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Mean fwd return: %{y:.2f} bps<br>"
                            "<extra></extra>"
                        ),
                    )
                )
                fig_bar.update_layout(
                    title=f"Mean {imb_horizon}-step forward return by imbalance bucket (±1 std, in basis points)",
                    xaxis_title="Imbalance bucket",
                    yaxis_title="Cumulative log-return (bps)",
                    height=400,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # ── summary table ────────────────────────────────────────────
                disp = bucket_stats.copy()
                disp["mean (bps)"] = (disp["mean"] * 1e4).round(2)
                disp["std (bps)"] = (disp["std"] * 1e4).round(2)
                corr = float(sig["imbalance"].corr(sig["forward_return"]))
                st.dataframe(
                    disp[["bucket", "count", "mean (bps)", "std (bps)"]].reset_index(drop=True),
                    use_container_width=True,
                )
                st.metric(
                    f"Pearson ρ (imbalance vs {imb_horizon}-step return)",
                    f"{corr:.4f}",
                    help="Values near ±1 indicate a strong linear relationship; near 0 means little linear predictability.",
                )

    st.divider()
    st.caption("Local: pip install -r requirements.txt then streamlit run app.py (http://localhost:8501)")


if __name__ == "__main__":
    main()
