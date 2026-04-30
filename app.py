"""Streamlit: Options Microstructure & Greeks Lab."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import default_data_dir, group_files_by_date, list_merged_days, load_merged_day, load_sessions_concat
from vol_forecast import (
    build_feature_matrix,
    build_imbalance_signal,
    build_iv_smile,
    build_snapshot_panel,
    train_vol_model,
)

st.set_page_config(
    page_title="Options Microstructure & Greeks Lab",
    page_icon=None,
    layout="wide",
)
DATA_DIR = default_data_dir()

# ── stock-market colour palette ───────────────────────────────────────────────
TEAL   = "#00d4aa"   # primary accent (like Bloomberg/TradingView)
GREEN  = "#00c805"   # up / bullish
RED    = "#ff4b4b"   # down / bearish
GOLD   = "#ffd700"   # second series / puts
BLUE   = "#0090ff"   # neutral info
GRAY   = "#8b949e"   # muted

CHART_TEMPLATE = "plotly_dark"


@st.cache_data(show_spinner="Loading files…")
def _load_range_cached(paths: tuple[str, ...]) -> pd.DataFrame:
    return load_sessions_concat(paths)


@st.cache_data(show_spinner="Loading merged day…")
def _load_merged_cached(path: str) -> pd.DataFrame:
    return load_merged_day(path)


def snapshot_at(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    return df[df["timestamp"] == ts].copy()



def _atm_iv(snap: pd.DataFrame) -> str:
    """Return ATM call IV as a formatted string, or '—' if unavailable."""
    from vol_forecast import build_iv_smile
    smile = build_iv_smile(snap)
    if smile.empty or "call_iv" not in smile.columns:
        return "—"
    valid = smile.dropna(subset=["call_iv"])
    if valid.empty:
        return "—"
    spx = float(snap["spx_price"].iloc[0])
    idx = (valid["future_strike"] - spx).abs().idxmin()
    iv = valid.loc[idx, "call_iv"]
    return f"{iv * 100:.1f}%" if pd.notna(iv) else "—"


def main():
    # ── page header ───────────────────────────────────────────────────────────
    st.title("Options Microstructure & Greeks Lab")
    st.markdown(
        "**Can the shape of the order book tell us where price is headed?**  \n"
        "We combine real SPX/ES **limit-order-book (LOB)** snapshots with **option Greeks** "
        "to explore market liquidity, risk pricing, and short-horizon predictability."
    )
    with st.expander("How to use this app"):
        st.markdown(
            "1. **Pick a trading date** in the sidebar (and a snapshot time to freeze the view).  \n"
            "2. **Work through the tabs left to right** — each one adds a layer of insight.  \n"
            "3. Every chart is interactive: hover for values, click the legend to hide/show series, drag to zoom."
        )
    st.divider()

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Data Selection")

        date_groups  = group_files_by_date(DATA_DIR)
        merged_days  = list_merged_days(DATA_DIR)
        all_dates    = sorted(set(date_groups.keys()) | set(merged_days.keys()))
        date_labels  = {d: f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in all_dates}

        if not all_dates:
            st.error("No data files found in `data/`.")
            st.stop()

        sel_date = st.selectbox(
            "Trading date",
            all_dates,
            format_func=lambda d: date_labels[d],
        )

        if sel_date in merged_days:
            df_full = _load_merged_cached(str(merged_days[sel_date]))
        else:
            day_files = date_groups.get(sel_date, [])
            df_full = _load_range_cached(tuple(str(f) for f in day_files))

        # always show time window slider
        ts_all = sorted(df_full["timestamp"].dropna().unique())
        if len(ts_all) > 1:
            t_start, t_end = st.select_slider(
                "Time window (UTC)",
                options=ts_all,
                value=(ts_all[0], ts_all[-1]),
                format_func=lambda t: pd.Timestamp(t).strftime("%H:%M"),
            )
            df = df_full[(df_full["timestamp"] >= t_start) & (df_full["timestamp"] <= t_end)].copy()
        else:
            t_start, t_end = ts_all[0], ts_all[0]
            df = df_full.copy()

        choice = f"{sel_date}_{pd.Timestamp(t_start).strftime('%H%M')}-{pd.Timestamp(t_end).strftime('%H%M')}"

        ts_list = sorted(df["timestamp"].dropna().unique())
        if not ts_list:
            st.error("No valid timestamps in selected range.")
            st.stop()

        sel_ts = st.select_slider(
            "Snapshot (UTC)",
            options=ts_list,
            value=ts_list[-1],
            format_func=lambda t: pd.Timestamp(t).strftime("%H:%M:%S"),
        )
        snap = snapshot_at(df, pd.Timestamp(sel_ts))

        # ── optional second session for comparison ────────────────────────────
        st.divider()
        with st.expander("Compare with another date"):
            compare_dates = [d for d in all_dates if d != sel_date]
            df2 = None
            if compare_dates:
                sel_date2 = st.selectbox(
                    "Second date",
                    compare_dates,
                    format_func=lambda d: date_labels[d],
                    key="compare_date",
                )
                if sel_date2 in merged_days:
                    df2_full = _load_merged_cached(str(merged_days[sel_date2]))
                else:
                    day_files2 = date_groups.get(sel_date2, [])
                    df2_full = _load_range_cached(tuple(str(f) for f in day_files2))

                ts2_all = sorted(df2_full["timestamp"].dropna().unique())
                if len(ts2_all) > 1:
                    t2_start, t2_end = st.select_slider(
                        "Time window (date 2, UTC)",
                        options=ts2_all,
                        value=(ts2_all[0], ts2_all[-1]),
                        format_func=lambda t: pd.Timestamp(t).strftime("%H:%M"),
                        key="compare_window",
                    )
                    df2 = df2_full[(df2_full["timestamp"] >= t2_start) & (df2_full["timestamp"] <= t2_end)].copy()
                else:
                    df2 = df2_full.copy()
            else:
                st.info("Only one date available — add more data to enable comparison.")

        with st.expander("Glossary"):
            st.markdown(
                "**MBO depth** — total quoted size at a strike on one side  \n"
                "**Delta (Δ)** — price sensitivity to a $1 move in the underlying  \n"
                "**Gamma (Γ)** — rate of change of delta  \n"
                "**Vega (ν)** — sensitivity to a 1-point move in implied vol  \n"
                "**Theta (Θ)** — daily time decay  \n"
                "**IV** — implied volatility back-calculated from delta  \n"
                "**Imbalance** — (bid depth − ask depth) / total depth  \n"
            )

        st.divider()
        st.caption(f"**{len(df):,}** rows · **{df['timestamp'].nunique()}** snapshots loaded")

    if snap.empty:
        st.warning("Empty snapshot — try a different time.")
        st.stop()

    panel = build_snapshot_panel(df)

    # ── top metrics row ───────────────────────────────────────────────────────
    meta  = snap.iloc[0]
    spx   = float(meta.get("spx_price", 0))
    spx0  = float(df.iloc[0].get("spx_price", spx))
    spx_chg = spx - spx0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("SPX Spot",      f"{spx:,.2f}",                  f"{spx_chg:+.2f} vs session open")
    c2.metric("ES Ref",        f"{float(meta.get('current_es_price', 0)):,.0f}")
    c3.metric("Time to Expiry",f"{float(meta.get('t', 0)):.4f} yrs")
    c4.metric("ATM IV (call)", _atm_iv(snap))
    c5.metric("LOB rows",      f"{len(snap):,}")

    st.divider()

    # ── tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
        "1. Market Liquidity",
        "2. Risk Pricing (IV Smile)",
        "3. Option Sensitivities",
        "4. Session Dynamics",
        "5. LOB as a Signal",
        "6. Predictive Model",
        "7. Cross-Day Comparison",
        "8. Summary",
    ])

    # ── Tab 1: Market Liquidity (Order Book) ─────────────────────────────────
    with t1:
        st.subheader("Market Liquidity — Where is the depth sitting?")
        st.markdown(
            "Each bar shows the total quoted size (**MBO depth**) at that strike. "
            "Bids (buyers) in blue; asks (sellers) in orange. "
            "A large cluster of bids below spot = support; a large cluster of asks above spot = resistance."
        )
        st.info("Look for where depth is concentrated relative to the spot line — that's where the market is defending a level.")

        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            min_depth = st.slider(
                "Minimum depth to show",
                0, int(snap["MBO_depth"].max() or 10), 0,
                help="Hide strikes with very thin liquidity.",
                key="lob_min_depth",
            )
        with col_f2:
            zoom_atm = st.checkbox("Zoom to ±50 strikes around spot", value=False)

        plot_snap = snap[snap["MBO_depth"] >= min_depth].copy()
        if zoom_atm:
            spx_val = float(snap["spx_price"].iloc[0])
            plot_snap = plot_snap[
                (plot_snap["future_strike"] >= spx_val - 50) &
                (plot_snap["future_strike"] <= spx_val + 50)
            ]

        if plot_snap.empty:
            st.info("No rows match the current filters.")
        else:
            fig = px.bar(
                plot_snap.sort_values(["Side", "future_strike"], ascending=[True, False]),
                x="MBO_depth",
                y="future_strike",
                color="Side",
                color_discrete_map={"Bid": BLUE, "Ask": GOLD},
                orientation="h",
                hover_data=["spx_strike", "MBO_pulling_stacking", "call_delta", "put_delta"],
                labels={"MBO_depth": "MBO Depth", "future_strike": "Strike"},
            )
            spx_val = float(snap["spx_price"].iloc[0])
            fig.add_hline(y=spx_val, line_dash="dot", line_color=RED,
                          annotation_text=f"Spot {spx_val:.0f}", annotation_position="right")
            fig.update_layout(height=650, hovermode="y unified", template=CHART_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)

        liq = snap[snap["MBO_depth"] > 0].sort_values("future_strike")
        cols = [c for c in ["Side", "future_strike", "MBO_depth", "call_delta", "call_vega", "put_delta", "put_vega"] if c in liq.columns]
        with st.expander("Show raw depth table"):
            st.dataframe(liq[cols].reset_index(drop=True), use_container_width=True)

    # ── Tab 3: Option Sensitivities (Greeks) ─────────────────────────────────
    with t3:
        st.subheader("Option Sensitivities — How do options react to market moves?")
        st.markdown(
            "Greeks measure how an option's price changes when market conditions shift. "
            "Select which Greeks to display and whether to show calls, puts, or both."
        )
        st.info("Notice how Delta crosses ~0.5 at the spot line — that's the at-the-money point. Gamma peaks there too, meaning options are most sensitive to price moves near the current level.")

        sub = snap.drop_duplicates(subset=["future_strike"]).sort_values("future_strike")
        spx_val = float(snap["spx_price"].iloc[0])

        g_col1, g_col2 = st.columns([2, 1])
        with g_col1:
            available = [g for g in ["delta", "gamma", "vega", "theta"] if f"call_{g}" in sub.columns]
            sel_greeks = st.multiselect(
                "Greeks to display",
                available,
                default=available,
                key="greek_select",
            )
        with g_col2:
            side_choice = st.radio("Option side", ["Calls", "Puts", "Both"], horizontal=True)

        def _greek_fig(prefix: str, label: str, color: str) -> go.Figure:
            fig = go.Figure()
            for g in sel_greeks:
                col = f"{prefix}_{g}"
                if col in sub.columns:
                    fig.add_trace(go.Scatter(
                        x=sub["future_strike"], y=sub[col],
                        name=g.capitalize(), mode="lines+markers",
                        marker=dict(size=4),
                        hovertemplate=f"Strike %{{x:.2f}}<br>{g.capitalize()} %{{y:.4f}}<extra></extra>",
                    ))
            fig.add_vline(x=spx_val, line_dash="dot", line_color=RED,
                          annotation_text=f"Spot {spx_val:.0f}", annotation_position="top right")
            fig.update_layout(
                title=f"{label} Greeks",
                xaxis_title="Strike",
                height=420,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1),
                template=CHART_TEMPLATE,
            )
            return fig

        if side_choice in ("Calls", "Both"):
            st.plotly_chart(_greek_fig("call", "Call", BLUE), use_container_width=True)
        if side_choice in ("Puts", "Both"):
            st.plotly_chart(_greek_fig("put", "Put", GOLD), use_container_width=True)

        with st.expander("What do these Greeks mean?"):
            st.markdown(
                "- **Delta**: how much the option price moves per $1 move in SPX. Calls range 0→1, puts −1→0.  \n"
                "- **Gamma**: how fast delta changes. Peaks near ATM (at-the-money) and near expiry.  \n"
                "- **Vega**: sensitivity to implied volatility — larger vega means bigger P&L from vol moves.  \n"
                "- **Theta**: daily time decay — the option loses this much value each day just from time passing."
            )

    # ── Tab 4: Session Dynamics (Time Series) ────────────────────────────────
    with t4:
        st.subheader("Session Dynamics — How did things evolve?")
        st.markdown("Track how spot price and key option metrics change snapshot by snapshot across the session.")
        st.info("Select **SPX Spot + Depth Imbalance** together to see if the order book shifted before price moved.")

        if panel.empty:
            st.warning("Not enough data to build a time-series panel.")
        else:
            pplot = panel.reset_index()

            ts_series = st.multiselect(
                "Series to plot",
                ["SPX Spot", "ATM Call Delta", "ATM Call Vega", "Bid Depth", "Ask Depth", "Depth Imbalance"],
                default=["SPX Spot", "ATM Call Delta"],
                key="ts_series",
            )

            series_map = {
                "SPX Spot":        ("spx",             "SPX Spot Price",       BLUE),
                "ATM Call Delta":  ("atm_call_delta",  "ATM Call Δ",           GOLD),
                "ATM Call Vega":   ("atm_call_vega",   "ATM Call ν",           GREEN),
                "Bid Depth":       ("bid_depth",       "Total Bid Depth",      BLUE),
                "Ask Depth":       ("ask_depth",       "Total Ask Depth",      RED),
                "Depth Imbalance": ("depth_imbalance", "Depth Imbalance",      GRAY),
            }

            for label in ts_series:
                col, yname, color = series_map[label]
                if col not in pplot.columns or not pplot[col].notna().any():
                    st.info(f"{label} has no data for this session.")
                    continue
                fig = px.line(
                    pplot, x="timestamp", y=col,
                    title=yname,
                    labels={"timestamp": "Time (UTC)", col: yname},
                )
                fig.update_traces(line_color=color)
                fig.update_layout(height=300, xaxis_title="Time (UTC)", yaxis_title=yname, template=CHART_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)

            # fixed-strike delta chart
            strikes_num = pd.to_numeric(df["future_strike"], errors="coerce")
            spx_num     = pd.to_numeric(df["spx_price"],    errors="coerce")
            if strikes_num.notna().any() and spx_num.notna().any() and "call_delta" in df.columns:
                med  = float(spx_num.median())
                u    = np.sort(np.unique(strikes_num.dropna().to_numpy()))
                k    = min(3, len(u))
                pick = np.sort(u[np.argsort(np.abs(u - med))[:k]])
                mask = np.isin(pd.to_numeric(df["future_strike"], errors="coerce"), pick)
                bundle = (
                    df.loc[mask]
                    .sort_values(["timestamp", "future_strike"])
                    .drop_duplicates(subset=["timestamp", "future_strike"], keep="first")
                )
                if not bundle.empty and bundle["call_delta"].notna().any():
                    with st.expander("Call delta at fixed strikes near median spot"):
                        fig7 = px.line(
                            bundle, x="timestamp", y="call_delta", color="future_strike",
                            title="Call Δ at up to 3 strikes nearest session median spot",
                            labels={"timestamp": "Time (UTC)", "call_delta": "Call Δ", "future_strike": "Strike"},
                        )
                        fig7.update_layout(height=380, legend_title="Strike", template=CHART_TEMPLATE)
                        st.plotly_chart(fig7, use_container_width=True)
                        st.caption(
                            f"Strikes: {', '.join(f'{float(s):g}' for s in pick)} — "
                            "fixed at session median spot, unlike the rolling ATM series above."
                        )

    # ── Tab 6: Predictive Model (Vol Forecast) ───────────────────────────────
    with t6:
        st.subheader("Predictive Model — Can LOB features forecast volatility?")
        st.markdown(
            "A gradient-boosting model predicts how much price will move over the next **H** snapshots, "
            "using past volatility, order book depth, imbalance, and ATM Greeks as features. "
            "Data is split in time order (last 20% = test) to avoid lookahead bias."
        )
        st.info("Check **Feature Importance** — if `depth_imbalance` ranks high, the order book structure is adding predictive power beyond just past vol.")

        c1, c2 = st.columns(2)
        with c1:
            lookback = st.slider("Past window W (snapshots)", 3, 40, 10,
                                 help="How many past returns to summarize into 'past_rv'.",
                                 key=f"vol_W_{choice}")
        with c2:
            horizon = st.slider("Forward horizon H (snapshots)", 2, 30, 8,
                                help="How many future returns define the target.",
                                key=f"vol_H_{choice}")

        note = panel.attrs.get("return_price_note", "")
        if note:
            st.info(f"Price series used for returns: **{note}**")

        X, y, feat_names = build_feature_matrix(panel, lookback, horizon)
        if X.empty:
            st.warning("Not enough snapshots for this W/H. Try smaller values or load more data.")
        elif float(y.std()) < 1e-14:
            st.error(
                "Realized volatility is zero — the underlying price is flat across all snapshots. "
                "Load a session where `spx_price` or `current_es_price` changes over time."
            )
        else:
            try:
                model, metrics, y_tr, p_tr, y_te, p_te = train_vol_model(X, y, test_size=0.2)
            except ValueError as e:
                st.error(str(e))
            else:
                tr2  = metrics["test_r2"]
                r2_display = "N/A" if tr2 != tr2 else f"{tr2:.3f}"
                r2_help = (
                    "R² = 1 is perfect prediction; R² = 0 means the model is no better than predicting the mean; "
                    "negative R² means it's worse than the mean."
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Test MAE",    f"{metrics['test_mae']:.2e}", help="Mean absolute error on the held-out test set.")
                m2.metric("Test R²",     r2_display,                  help=r2_help)
                m3.metric("Train rows",  metrics["n_train"])
                m4.metric("Test rows",   metrics["n_test"])

                col_a, col_b = st.columns(2)
                with col_a:
                    fig_f = go.Figure()
                    fig_f.add_trace(go.Scatter(
                        x=p_te, y=y_te, mode="markers", name="Test set",
                        marker=dict(size=7, opacity=0.65, color=TEAL),
                        hovertemplate="Predicted %{x:.2e}<br>Actual %{y:.2e}<extra></extra>",
                    ))
                    mx  = max(float(np.nanmax(y_te)), float(np.nanmax(p_te)), 1e-12)
                    mn  = min(float(np.nanmin(y_te)), float(np.nanmin(p_te)), 0.0)
                    pad = (mx - mn) * 0.05 + 1e-12
                    fig_f.add_trace(go.Scatter(
                        x=[mn - pad, mx + pad], y=[mn - pad, mx + pad],
                        mode="lines", name="Perfect fit",
                        line=dict(dash="dash", color=GRAY),
                    ))
                    fig_f.update_layout(
                        title="Actual vs Predicted (test set)",
                        xaxis_title="Predicted σ_fwd",
                        yaxis_title="Actual σ_fwd",
                        height=380,
                        xaxis=dict(tickformat=".1e"),
                        yaxis=dict(tickformat=".1e"),
                        template=CHART_TEMPLATE,
                    )
                    st.plotly_chart(fig_f, use_container_width=True)

                with col_b:
                    imp    = np.asarray(model.feature_importances_, dtype=float)
                    imp_df = pd.DataFrame({"Importance": imp, "Feature": feat_names}).sort_values("Importance")
                    fig_i  = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                                    title="Feature Importance")
                    fig_i.update_traces(marker_color=TEAL)
                    fig_i.update_layout(height=380,
                                        xaxis=dict(range=[0, float(imp.max()) * 1.15]),
                                        template=CHART_TEMPLATE)
                    st.plotly_chart(fig_i, use_container_width=True)

                with st.expander("How to interpret these results"):
                    st.markdown(
                        "- **High R²** (close to 1) → LOB features genuinely predict short-term vol.  \n"
                        "- **Low/negative R²** → vol is hard to predict from these features alone.  \n"
                        "- **Feature importance** shows which input drives predictions most. "
                        "`past_rv` dominating is expected; a high `depth_imbalance` rank suggests "
                        "the order book structure carries incremental signal beyond past vol."
                    )

    # ── Tab 2: Risk Pricing (IV Smile) ───────────────────────────────────────
    with t2:
        st.subheader("Risk Pricing — What does the market fear?")
        st.markdown(
            "Implied volatility (IV) is back-calculated from each strike's delta. "
            "The **shape** of the curve reveals how the market prices tail risk — "
            "a steep left tail (put skew) means the market is paying up for crash protection."
        )
        st.info("A positive **Put Skew** means out-of-the-money puts are more expensive than calls — the market fears a drop more than a rally.")

        smile_df = build_iv_smile(snap)
        if smile_df.empty:
            st.warning("Cannot compute IV smile for this snapshot.")
        else:
            valid = smile_df.dropna(subset=["call_iv", "put_iv"], how="all")

            # ── skew metric ───────────────────────────────────────────────────
            spx_val = float(snap["spx_price"].iloc[0])
            near_25d_put  = valid[valid["moneyness"] < -0.05]
            near_25d_call = valid[valid["moneyness"] >  0.05]
            if not near_25d_put.empty and not near_25d_call.empty:
                otm_put_iv  = float(near_25d_put["put_iv"].dropna().iloc[-1]) * 100
                otm_call_iv = float(near_25d_call["call_iv"].dropna().iloc[0]) * 100
                skew        = otm_put_iv - otm_call_iv
                sk1, sk2, sk3 = st.columns(3)
                sk1.metric("OTM Put IV",  f"{otm_put_iv:.1f}%",  help="IV of first strike >5% below spot")
                sk2.metric("OTM Call IV", f"{otm_call_iv:.1f}%", help="IV of first strike >5% above spot")
                sk3.metric("Put Skew",    f"{skew:+.1f}%",
                           help="Positive = puts are more expensive than calls (typical for equities — crash fear)")

            iv_col1, iv_col2 = st.columns([3, 1])
            with iv_col2:
                show_put_iv  = st.checkbox("Show Put IV",  value=True,  key="smile_put")
                show_call_iv = st.checkbox("Show Call IV", value=True,  key="smile_call")
                mon_range    = st.slider(
                    "Log-moneyness range",
                    float(valid["moneyness"].min()), float(valid["moneyness"].max()),
                    (float(valid["moneyness"].min()), float(valid["moneyness"].max())),
                    key="smile_mon_range",
                )

            filtered = valid[
                (valid["moneyness"] >= mon_range[0]) &
                (valid["moneyness"] <= mon_range[1])
            ]

            with iv_col1:
                fig_smile = go.Figure()
                if show_call_iv and filtered["call_iv"].notna().any():
                    fig_smile.add_trace(go.Scatter(
                        x=filtered["future_strike"], y=filtered["call_iv"] * 100,
                        mode="lines+markers", name="Call IV",
                        line=dict(color=TEAL, width=2), marker=dict(size=5),
                        hovertemplate="Strike %{x:.2f}<br>Call IV %{y:.2f}%<extra></extra>",
                    ))
                if show_put_iv and filtered["put_iv"].notna().any():
                    fig_smile.add_trace(go.Scatter(
                        x=filtered["future_strike"], y=filtered["put_iv"] * 100,
                        mode="lines+markers", name="Put IV",
                        line=dict(color=GOLD, width=2, dash="dot"), marker=dict(size=5),
                        hovertemplate="Strike %{x:.2f}<br>Put IV %{y:.2f}%<extra></extra>",
                    ))
                fig_smile.add_vline(x=spx_val, line_dash="dash", line_color=RED,
                                    annotation_text=f"Spot {spx_val:.0f}", annotation_position="top right")
                fig_smile.update_layout(
                    xaxis_title="Strike",
                    yaxis_title="Implied Volatility (%)",
                    height=450,
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.08),
                    template=CHART_TEMPLATE,
                )
                st.plotly_chart(fig_smile, use_container_width=True)

            # ── moneyness view ────────────────────────────────────────────────
            st.markdown("#### IV vs Log-Moneyness ln(K/S)")
            st.caption(
                "ln(K/S) < 0 → in-the-money calls (K below spot).  "
                "The steeper the left tail, the stronger the **put skew** — "
                "the market is paying up for downside protection."
            )
            fig_mon = go.Figure()
            if show_call_iv and filtered["call_iv"].notna().any():
                fig_mon.add_trace(go.Scatter(
                    x=filtered["moneyness"], y=filtered["call_iv"] * 100,
                    mode="lines+markers", name="Call IV", line=dict(color=TEAL, width=2),
                    hovertemplate="ln(K/S) %{x:.4f}<br>IV %{y:.2f}%<extra></extra>",
                ))
            if show_put_iv and filtered["put_iv"].notna().any():
                fig_mon.add_trace(go.Scatter(
                    x=filtered["moneyness"], y=filtered["put_iv"] * 100,
                    mode="lines+markers", name="Put IV", line=dict(color=GOLD, width=2, dash="dot"),
                    hovertemplate="ln(K/S) %{x:.4f}<br>IV %{y:.2f}%<extra></extra>",
                ))
            fig_mon.add_vline(x=0, line_dash="dash", line_color=RED, annotation_text="ATM")
            fig_mon.update_layout(
                xaxis_title="Log-moneyness ln(K/S)",
                yaxis_title="Implied Volatility (%)",
                height=380, hovermode="x unified",
                legend=dict(orientation="h", y=1.08),
                template=CHART_TEMPLATE,
            )
            st.plotly_chart(fig_mon, use_container_width=True)

            with st.expander("Show IV data table"):
                st.dataframe(
                    filtered[["future_strike", "moneyness", "call_iv", "put_iv"]]
                    .rename(columns={"call_iv": "Call IV (%)", "put_iv": "Put IV (%)", "moneyness": "ln(K/S)"})
                    .assign(**{
                        "Call IV (%)": lambda d: (d["Call IV (%)"] * 100).round(2),
                        "Put IV (%)":  lambda d: (d["Put IV (%)"]  * 100).round(2),
                        "ln(K/S)":     lambda d: d["ln(K/S)"].round(5),
                    })
                    .reset_index(drop=True),
                    use_container_width=True,
                )

    # ── Tab 5: LOB as a Signal (Imbalance) ───────────────────────────────────
    with t5:
        st.subheader("LOB as a Signal — Does order book shape predict price direction?")
        st.markdown(
            "**Depth imbalance** = (bid depth − ask depth) / total depth.  \n"
            "Ranges from **−1** (all sell-side) to **+1** (all buy-side).  \n"
            "This tab tests whether the imbalance at time *t* predicts the price move over the next *N* snapshots."
        )
        st.info("If the bar chart shows green on the right (bullish) and red on the left (bearish), the order book has directional signal in this session.")

        i_col1, i_col2 = st.columns(2)
        with i_col1:
            imb_horizon = st.slider(
                "Forward horizon N (snapshots)",
                2, 30, 5,
                help="How many snapshots ahead to measure the cumulative log return.",
                key=f"imb_H_{choice}",
            )
        with i_col2:
            min_count = st.slider(
                "Min observations per bucket",
                1, 20, 3,
                help="Buckets with fewer points are hidden.",
                key=f"imb_min_{choice}",
            )

        sig = build_imbalance_signal(panel, imb_horizon)
        if sig.empty:
            st.warning("Not enough snapshots. Try a smaller horizon or load more data.")
        else:
            corr = float(sig["imbalance"].corr(sig["forward_return"]))

            # interpretation badge
            if abs(corr) < 0.05:
                interp = "**Weak signal** — imbalance has little linear relationship with forward returns in this session."
            elif corr > 0:
                interp = f"**Positive correlation ({corr:+.3f})** — higher bid depth tends to precede upward moves."
            else:
                interp = f"**Negative correlation ({corr:+.3f})** — higher bid depth actually precedes downward moves (possible absorption)."
            st.info(interp)

            col_sc, col_bar = st.columns(2)

            with col_sc:
                fig_sc = px.scatter(
                    sig, x="imbalance", y="forward_return",
                    color="bucket", opacity=0.6,
                    trendline="ols", trendline_scope="overall",
                    labels={
                        "imbalance":      "Depth Imbalance",
                        "forward_return": f"Cum. log-return (next {imb_horizon} steps)",
                    },
                    title=f"Imbalance vs {imb_horizon}-step forward return",
                    hover_data={"timestamp": True},
                    template=CHART_TEMPLATE,
                )
                fig_sc.update_layout(height=400, legend_title="Bucket", showlegend=False, template=CHART_TEMPLATE)
                st.plotly_chart(fig_sc, use_container_width=True)

            with col_bar:
                bucket_stats = (
                    sig.groupby("bucket", observed=True)["forward_return"]
                    .agg(mean="mean", std="std", count="count")
                    .reset_index()
                )
                bucket_stats = bucket_stats[bucket_stats["count"] >= min_count]
                if bucket_stats.empty:
                    st.info("All buckets below minimum count. Lower the filter.")
                else:
                    bucket_stats["color"] = bucket_stats["mean"].apply(
                        lambda v: GREEN if v > 0 else RED
                    )
                    fig_bar = go.Figure(go.Bar(
                        x=bucket_stats["bucket"].astype(str),
                        y=bucket_stats["mean"] * 1e4,
                        error_y=dict(type="data", array=(bucket_stats["std"] * 1e4).tolist(), visible=True),
                        marker_color=bucket_stats["color"].tolist(),
                        hovertemplate="<b>%{x}</b><br>Mean: %{y:.2f} bps<extra></extra>",
                    ))
                    fig_bar.update_layout(
                        title=f"Mean {imb_horizon}-step return by bucket (bps, ±1 std)",
                        xaxis_title="Imbalance bucket",
                        yaxis_title="Return (basis points)",
                        height=400,
                        template=CHART_TEMPLATE,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            st.metric(
                f"Pearson ρ  (imbalance → {imb_horizon}-step return)",
                f"{corr:.4f}",
                help="−1 to +1. Values near 0 mean the order book imbalance has little predictive power here.",
            )

            with st.expander("Show raw signal data"):
                st.dataframe(
                    sig[["timestamp", "imbalance", "forward_return", "bucket"]]
                    .assign(
                        imbalance=lambda d: d["imbalance"].round(4),
                        forward_return=lambda d: (d["forward_return"] * 1e4).round(3),
                    )
                    .rename(columns={"forward_return": "fwd_return (bps)"})
                    .reset_index(drop=True),
                    use_container_width=True,
                )

    # ── Tab 7: Cross-Day Comparison ───────────────────────────────────────────
    with t7:
        st.subheader("Cross-Day Comparison — How did the market change between sessions?")
        st.markdown("Select a second date in the sidebar expander, then compare IV smiles and ATM dynamics side by side.")

        if df2 is None:
            st.info("Open **'Compare with another date'** in the sidebar to load a second session.")
        else:
            panel2 = build_snapshot_panel(df2)
            snap2  = snapshot_at(df2, df2["timestamp"].dropna().max())

            label1 = date_labels[sel_date]
            label2 = date_labels.get(sel_date2, "Session 2")

            # ── IV smile overlay ──────────────────────────────────────────────
            st.markdown("#### Implied Volatility Smile")
            smile1 = build_iv_smile(snap).dropna(subset=["call_iv"])
            smile2 = build_iv_smile(snap2).dropna(subset=["call_iv"])

            fig_cmp = go.Figure()
            if not smile1.empty:
                fig_cmp.add_trace(go.Scatter(
                    x=smile1["moneyness"], y=smile1["call_iv"] * 100,
                    mode="lines+markers", name=f"Call IV — {label1}",
                    line=dict(color=TEAL, width=2),
                    hovertemplate="ln(K/S) %{x:.4f}<br>IV %{y:.2f}%<extra></extra>",
                ))
            if not smile2.empty:
                fig_cmp.add_trace(go.Scatter(
                    x=smile2["moneyness"], y=smile2["call_iv"] * 100,
                    mode="lines+markers", name=f"Call IV — {label2}",
                    line=dict(color=GOLD, width=2, dash="dot"),
                    hovertemplate="ln(K/S) %{x:.4f}<br>IV %{y:.2f}%<extra></extra>",
                ))
            fig_cmp.add_vline(x=0, line_dash="dash", line_color=RED, annotation_text="ATM")
            fig_cmp.update_layout(
                xaxis_title="Log-moneyness ln(K/S)",
                yaxis_title="Implied Volatility (%)",
                height=400, hovermode="x unified",
                legend=dict(orientation="h", y=1.08),
                template=CHART_TEMPLATE,
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
            st.caption("A higher or steeper curve on one date = the market was pricing more risk that day.")

            # ── ATM delta over time overlay ───────────────────────────────────
            st.markdown("#### ATM Call Delta Over Time")
            if not panel.empty and not panel2.empty:
                p1 = panel.reset_index()[["timestamp", "atm_call_delta"]].dropna()
                p2 = panel2.reset_index()[["timestamp", "atm_call_delta"]].dropna()

                fig_atm = go.Figure()
                fig_atm.add_trace(go.Scatter(
                    x=p1["timestamp"], y=p1["atm_call_delta"],
                    mode="lines", name=label1, line=dict(color=TEAL),
                ))
                fig_atm.add_trace(go.Scatter(
                    x=p2["timestamp"], y=p2["atm_call_delta"],
                    mode="lines", name=label2, line=dict(color=GOLD, dash="dot"),
                ))
                fig_atm.update_layout(
                    xaxis_title="Time (UTC)", yaxis_title="ATM Call Δ",
                    height=350, hovermode="x unified",
                    legend=dict(orientation="h", y=1.08),
                    template=CHART_TEMPLATE,
                )
                st.plotly_chart(fig_atm, use_container_width=True)

            # ── metrics comparison table ──────────────────────────────────────
            st.markdown("#### Key Metrics Side by Side")
            atm_iv1 = _atm_iv(snap)
            atm_iv2 = _atm_iv(snap2)
            spx1    = float(snap["spx_price"].iloc[0])
            spx2    = float(snap2["spx_price"].iloc[0])
            t1_val  = float(snap["t"].iloc[0])
            t2_val  = float(snap2["t"].iloc[0])
            cmp_df  = pd.DataFrame({
                "Metric":          ["SPX Spot", "Time to Expiry (yrs)", "ATM Call IV"],
                label1:            [f"{spx1:,.2f}", f"{t1_val:.4f}", atm_iv1],
                label2:            [f"{spx2:,.2f}", f"{t2_val:.4f}", atm_iv2],
            })
            st.dataframe(cmp_df.set_index("Metric"), use_container_width=True)

    # ── Tab 8: Summary ────────────────────────────────────────────────────────
    with t8:
        st.subheader(f"Summary — {date_labels[sel_date]}")
        st.markdown("Key findings from this session at a glance.")

        # compute metrics
        spx_now  = float(snap["spx_price"].iloc[0])
        spx_open = float(df.iloc[0].get("spx_price", spx_now))
        spx_chg  = spx_now - spx_open
        atm_iv_str = _atm_iv(snap)

        smile_s  = build_iv_smile(snap).dropna(subset=["call_iv", "put_iv"], how="all")
        skew_str = "—"
        if not smile_s.empty:
            otm_p = smile_s[smile_s["moneyness"] < -0.05]["put_iv"].dropna()
            otm_c = smile_s[smile_s["moneyness"] >  0.05]["call_iv"].dropna()
            if not otm_p.empty and not otm_c.empty:
                skew_val = (float(otm_p.iloc[-1]) - float(otm_c.iloc[0])) * 100
                skew_str = f"{skew_val:+.1f}%"

        sig_s = build_imbalance_signal(panel, horizon=5)
        if not sig_s.empty:
            corr_s = float(sig_s["imbalance"].corr(sig_s["forward_return"]))
            imb_mean = float(sig_s["imbalance"].mean())
            corr_str = f"{corr_s:+.3f}"
            imb_str  = f"{imb_mean:+.3f} ({'buy-side' if imb_mean > 0 else 'sell-side'})"
        else:
            corr_str = "—"
            imb_str  = "—"

        # metrics row
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("SPX",          f"{spx_now:,.2f}", f"{spx_chg:+.2f} vs open")
        s2.metric("ATM IV",       atm_iv_str)
        s3.metric("Put Skew",     skew_str)
        s4.metric("Imbalance ρ",  corr_str)

        st.divider()

        # auto-generated bullet findings
        st.markdown("#### Findings")
        findings = []

        # SPX
        if abs(spx_chg) < 1:
            findings.append(f"SPX was essentially flat at **{spx_now:,.2f}** over the loaded window (change: {spx_chg:+.2f} pts).")
        elif spx_chg > 0:
            findings.append(f"SPX **gained {spx_chg:+.2f} pts** from session open to the selected snapshot ({spx_now:,.2f}).")
        else:
            findings.append(f"SPX **fell {spx_chg:.2f} pts** from session open to the selected snapshot ({spx_now:,.2f}).")

        # IV
        if atm_iv_str != "—":
            iv_num = float(atm_iv_str.replace("%", ""))
            level  = "elevated" if iv_num > 25 else "moderate" if iv_num > 15 else "low"
            findings.append(f"ATM implied volatility is **{atm_iv_str}** — {level} uncertainty priced by the market.")

        # Skew
        if skew_str != "—":
            sv = float(skew_str.replace("%", ""))
            if sv > 2:
                findings.append(f"Put skew of **{skew_str}** indicates the market is paying a significant premium for downside protection (crash fear).")
            elif sv > 0:
                findings.append(f"A mild put skew of **{skew_str}** shows slightly elevated demand for downside hedges.")
            else:
                findings.append(f"Skew is near flat (**{skew_str}**) — calls and puts are priced similarly, suggesting balanced risk perception.")

        # Imbalance
        if imb_str != "—":
            findings.append(f"Average LOB depth imbalance is **{imb_str}** — the order book is leaning toward the {'buy' if imb_mean > 0 else 'sell'} side.")
        if corr_str != "—":
            cv = float(corr_str)
            strength = "strong" if abs(cv) > 0.2 else "moderate" if abs(cv) > 0.05 else "weak"
            direction = "positive" if cv > 0 else "negative"
            findings.append(f"Imbalance has **{strength} {direction} predictive power** over 5-step forward returns (ρ = {corr_str}).")

        for f in findings:
            st.markdown(f"- {f}")

        st.divider()
        st.markdown("#### Data Loaded")
        st.dataframe(pd.DataFrame({
            "Field":  ["Date", "Snapshots", "Strikes covered", "Time to expiry"],
            "Value":  [
                date_labels[sel_date],
                str(df["timestamp"].nunique()),
                str(snap["future_strike"].nunique()),
                f"{float(snap['t'].iloc[0]):.4f} yrs",
            ],
        }).set_index("Field"), use_container_width=True)

    st.divider()
    st.caption("Run locally: `pip install -r requirements.txt` → `streamlit run app.py` → http://localhost:8501")


if __name__ == "__main__":
    main()
