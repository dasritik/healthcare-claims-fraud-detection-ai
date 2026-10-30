# Healthcare Insurance Claims Fraud Detection POC

## Scope
A fully synthetic, reproducible POC for investigating potentially anomalous healthcare insurance claims.
It is not a production fraud adjudication system and does not establish actual fraud.

## Initial configuration
- Claims: 50,000
- Synthetic anomaly labels: 1,500 (3.0%)
- Random seed: 20261029
- Storage: CSV
- Detection design: deterministic rules + provider peer groups + Isolation Forest + risk scoring
- UI: Streamlit (planned next step)

## Project structure
- `data/synthetic/`: generated CSV data
- `src/data_generation/generate_synthetic_data.py`: data generator
- `docs/`: architecture and rulebook PDFs
- `tests/`: starter tests
- `outputs/charts/`: documentation charts

## Run locally
1. Create a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Generate data from the project root:
   `python src/data_generation/generate_synthetic_data.py --output-dir data/synthetic --seed 20261029 --n-claims 50000 --anomaly-rate 0.03`
4. Start the app after the Streamlit implementation is added:
   `streamlit run app.py`

## Label leakage prevention
`synthetic_labels.csv` contains ground-truth synthetic anomaly labels and scenario names.
Do not use either column as an input feature for rules, Isolation Forest, or risk scoring.
Join labels only after scoring for evaluation.
