from pathlib import Path
import pandas as pd

DATA = Path("data/synthetic")

def test_expected_files_exist():
    for name in ["claims.csv","members.csv","providers.csv","procedures.csv","claim_line_items.csv","synthetic_labels.csv"]:
        assert (DATA/name).exists()

def test_claims_and_labels_are_separate():
    claims = pd.read_csv(DATA/"claims.csv")
    labels = pd.read_csv(DATA/"synthetic_labels.csv")
    assert "synthetic_anomaly_label" not in claims.columns
    assert "synthetic_scenario" not in claims.columns
    assert len(claims) == len(labels)
    assert claims["claim_id"].is_unique
    assert labels["claim_id"].is_unique

def test_injected_anomaly_rate():
    labels = pd.read_csv(DATA/"synthetic_labels.csv")
    assert labels["synthetic_anomaly_label"].sum() == 1500
    assert abs(labels["synthetic_anomaly_label"].mean() - 0.03) < 1e-9
