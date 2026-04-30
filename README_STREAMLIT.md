# Options Microstructure & Greeks Lab

**Idea:** An interactive lab that ties **limit-order-book (MBO) depth** on ES/SPX strikes to **option Greeks** (delta, gamma, vega, theta, etc.). Good for a course project: it uses your real LOB + Greeks fields and is more distinctive than a generic stock chart.

## Data

- `data/loaded_lob_*.csv.gz` — sessions (**double-gzip** + JSON array inside; loader handles this)

## Run locally

```bash
cd Group   # this folder
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## Features

1. **Order book ladder** — horizontal bars of MBO depth by strike; Bid vs Ask
2. **Greeks vs strike** — call and put sensitivities for the chosen snapshot
3. **Time series** — SPX and call-delta across timestamps in the loaded file
4. **Volatility forecast** — short-horizon **forward realized volatility** of SPX (next *H* snapshot returns), with features from past vol, LOB depth, and ATM call Greeks; time-ordered train/test (illustration only)

Use the sidebar to pick a session `.csv.gz`, then scrub snapshot time.
