"""Streamlit: Options Microstructure & Greeks Lab."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import default_data_dir, list_session_files, load_csv, load_session_gz

st.set_page_config(
    page_title="Options Microstructure & Greeks Lab",
    page_icon="📊",
    layout="wide",
)
DATA_DIR = default_data_dir()


@st.cache_data(show_spinner=True)
def _load_csv_cached(p: str) -> pd.DataFrame:
    return load_csv(p)


@st.cache_data(show_spinner="Loading session…")
def _load_gz_cached(p: str) -> pd.DataFrame:
    return load_session_gz(p)


def snapshot_at(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    return df[df["timestamp"] == ts].copy()


def main():
    st.title("Options Microstructure & Greeks Lab")
    st.markdown(
        "**Focus:** Link **limit-order-book depth** (MBO) to **option Greeks** on ES/SPX. "
        "Uses `loaded_lob_sample_for_excel.csv` or `loaded_lob_*.csv.gz` session files."
    )
    with st.sidebar:
        st.header("Data")
        mode = st.radio("Source", ["Sample CSV (fast)", "Session file (.csv.gz)"], index=0)
        if mode == "Sample CSV (fast)":
            csv_path = DATA_DIR / "loaded_lob_sample_for_excel.csv"
            if not csv_path.exists():
                st.error(f"Missing {csv_path}")
                st.stop()
            df = _load_csv_cached(str(csv_path))
        else:
            sessions = list_session_files(DATA_DIR)
            if not sessions:
                st.error("No loaded_lob_*.csv.gz in data/")
                st.stop()
            choice = st.selectbox("Session", [s.name for s in sessions], index=0)
            df = _load_gz_cached(str(DATA_DIR / choice))
            st.caption("First load of a large session may take a minute.")
        ts_list = sorted(df["timestamp"].dropna().unique())
        if not ts_list:
            st.error("No valid timestamps.")
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

    t1, t2, t3 = st.tabs(["Order book ladder", "Greeks vs strike", "Time series"])
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
        per_ts = df.sort_values(["timestamp", "future_strike"]).groupby("timestamp", as_index=False).first()
        fig4 = px.line(per_ts, x="timestamp", y="spx_price", title="SPX through snapshots")
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
        if "call_delta" in per_ts.columns:
            fig5 = px.line(per_ts, x="timestamp", y="call_delta", title="Call delta (first row per timestamp)")
            fig5.update_layout(height=400)
            st.plotly_chart(fig5, use_container_width=True)

    st.divider()
    st.caption("Local: pip install -r requirements.txt then streamlit run app.py (http://localhost:8501)")


if __name__ == "__main__":
    main()
