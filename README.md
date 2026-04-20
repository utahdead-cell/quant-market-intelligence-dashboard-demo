# Quant Market Intelligence — Dashboard Demo

Public Streamlit demo for the local MVP of **Quant Market Intelligence**.

This repository does **not** contain the full private project pipeline.  
It contains the **dashboard layer**, the **minimal service layer**, and the **materialized artifacts** required to demonstrate the product story end-to-end.

## What this demo shows

The dashboard is built around one narrow product question:

**Does a machine learning model add consistent out-of-sample value over a serious classical benchmark for 5-day volatility forecasting?**

The current demo compares:

- **Benchmark:** GARCH(1,1) with Student-t innovations
- **ML model:** XGBoost Regressor
- **Universe:** SPY, TLT, GLD, HYG
- **Main target:** `future_rv_5d`
- **Discrete regime view:** `future_regime_5d`

## Dashboard views

The public demo includes four main views:

- **Home / Executive Summary**
- **Market & Forecast**
- **Model Comparison**
- **Structural Changes**

## Included artifacts

This demo repository includes only the artifacts required to render the dashboard:

- normalized daily bars
- feature snapshots
- target snapshots
- benchmark forecasts
- ML forecasts
- model comparison outputs
- decision outputs
- structural break outputs

## What is intentionally excluded

This public demo does **not** expose the full internal project repository.  
It excludes most of the operational pipeline, training scripts, tests, internal docs, and other implementation details that are not required to run the dashboard.

## Repository structure

```bash
quant-market-intelligence-dashboard-demo/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
├── data/
│   ├── normalized/
│   ├── features/
│   └── targets/
├── artifacts/
│   └── evaluations/
├── src/
│   └── quant_platform/
│       └── services/
└── streamlit_app/
    ├── Home.py
    ├── charts.py
    ├── ui.py
    └── pages/
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app/Home.py