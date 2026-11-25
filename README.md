# Basketwise

Basketwise is a market-basket analysis project that turns transaction data into
interactive, evidence-led product-pairing recommendations. It combines an
Apriori training pipeline, an indexed FastAPI service, and a Streamlit
exploration dashboard.

## Get started

Requirements: Python 3.10 or later.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the project dependencies and launch the app:

```bash
python -m pip install -r requirements.txt
python run.py
```

- Dashboard: <http://127.0.0.1:8501>
- API: <http://127.0.0.1:8000>
- Interactive API docs: <http://127.0.0.1:8000/docs>

The launcher starts the API first, checks that it is healthy, and then starts
the dashboard. Press `Ctrl+C` in the terminal to stop both services.

## Retrain recommendations

```bash
python run.py --train
```

This command reruns Apriori on `data/raw/basket_analysis.csv`, generates the
association rules using a minimum support of `0.05` and minimum confidence of
`0.5`, and atomically replaces `artifacts/rules.csv` before starting both
services. If training fails, the previous rules artifact is preserved. The
normal `python run.py` command reuses the saved rules without retraining.

## Explore

Choose a product, set confidence and lift thresholds, rank by confidence or
lift, and download the filtered rules as CSV. Each result includes its trigger
itemset, recommended pairing, confidence, lift, and support.

| Metric | Interpretation |
| --- | --- |
| Support | Share of baskets containing the full rule |
| Confidence | Share of trigger baskets that also contain the recommended item |
| Lift | Relative frequency versus the recommended item's baseline rate |

These are observed associations, not causal claims or guaranteed sales lifts.

## API

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/products
curl "http://127.0.0.1:8000/recommend?product=chocolate&min_confidence=0.5&min_lift=1&sort_by=lift&limit=10"
```

Product matching is case-insensitive. Recommendation rules are parsed and
indexed once at API startup, so lookups operate on the selected product's
candidate rules rather than rescanning the CSV for every request.

## Project layout

```text
.
├── artifacts/                 Generated rules consumed by the API
├── data/
│   └── raw/                   Source transaction data
├── notebooks/                 Exploratory analysis
├── src/
│   └── market_basket/
│       ├── api.py             FastAPI app and indexed recommendation catalog
│       ├── dashboard.py       Streamlit application
│       └── training.py        Reproducible Apriori pipeline
├── .streamlit/                Dashboard theme
├── run.py                     Root launcher
└── requirements.txt
```
