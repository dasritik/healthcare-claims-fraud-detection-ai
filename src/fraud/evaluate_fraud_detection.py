
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data.load_data import load_csv_datasets


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SCORES_FILE = OUTPUT_DIR / "claim_fraud_scores.csv"

METRICS_FILE = OUTPUT_DIR / "fraud_model_metrics.csv"
SCENARIO_FILE = OUTPUT_DIR / "fraud_scenario_evaluation.csv"
CLAIM_EVALUATION_FILE = OUTPUT_DIR / "claim_level_evaluation.csv"

# Explicitly list the actual Boolean rule flags.
# Do NOT include rule_risk_points or rule_evidence.
RULE_COLUMNS = [
    "rule_duplicate_claim",
    "rule_high_amount",
    "rule_excessive_units",
    "rule_provider_billing_outlier",
    "rule_negative_submission_lag",
    "rule_line_amount_mismatch",
]

LABEL_COLUMN = "synthetic_anomaly_label"
CLAIM_ID_COLUMN = "claim_id"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def to_binary_flag(series: pd.Series) -> pd.Series:
    """
    Convert a prediction column to a Boolean flag safely.
    Handles Boolean, numeric 0/1, and common string values.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).gt(0).astype(int)

    normalized = series.fillna("").astype(str).str.strip().str.lower()

    return normalized.isin(
        ["true", "1", "yes", "y"]
    ).astype(int)


def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
) -> dict:
    """
    Calculate binary classification metrics.

    Positive class (1) = synthetic anomaly.
    Negative class (0) = normal claim.
    """
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "model": model_name,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": round(
            precision_score(y_true, y_pred, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_true, y_pred, zero_division=0), 4
        ),
        "f1_score": round(
            f1_score(y_true, y_pred, zero_division=0), 4
        ),
        "accuracy": round(
            accuracy_score(y_true, y_pred), 4
        ),
    }


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: list,
    dataframe_name: str,
) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {missing}\n"
            f"Available columns: {dataframe.columns.tolist()}"
        )


# ============================================================
# 3. LOAD MODEL SCORES AND SYNTHETIC LABELS
# ============================================================

def main():
    print("\n" + "=" * 65)
    print("HEALTHCARE CLAIMS FRAUD DETECTION - MODEL EVALUATION")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SCORES_FILE.exists():
        raise FileNotFoundError(
            f"Fraud scores file not found: {SCORES_FILE}\n"
            "Run src/fraud/fraud_engine.py first."
        )

    print("\nLoading fraud detection scores...")
    scores = pd.read_csv(SCORES_FILE)

    print(f"[LOADED] Fraud scores: {len(scores):,} rows")

    print("\nLoading synthetic evaluation labels...")
    datasets, manifest = load_csv_datasets()

    if "synthetic_labels" not in datasets:
        raise KeyError(
            "Dataset 'synthetic_labels' was not loaded. "
            "Check src/data/load_data.py and the generated CSV files."
        )

    labels = datasets["synthetic_labels"].copy()

    require_columns(
        scores,
        [CLAIM_ID_COLUMN, "flag_for_review", "ml_anomaly_flag"],
        "Fraud scores",
    )

    require_columns(
        labels,
        [CLAIM_ID_COLUMN, LABEL_COLUMN],
        "Synthetic labels",
    )

    # Keep only the evaluation columns from the label dataset.
    # Labels are joined AFTER detection; they are not model inputs.
    labels = labels[
        [CLAIM_ID_COLUMN, LABEL_COLUMN]
        + (
            ["synthetic_scenario"]
            if "synthetic_scenario" in labels.columns
            else []
        )
    ].copy()

    # Ensure claim IDs are compatible for joining.
    scores[CLAIM_ID_COLUMN] = scores[CLAIM_ID_COLUMN].astype(str)
    labels[CLAIM_ID_COLUMN] = labels[CLAIM_ID_COLUMN].astype(str)

    if scores[CLAIM_ID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate claim_id values found in fraud scores. "
            "Expected one scored row per claim."
        )

    if labels[CLAIM_ID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate claim_id values found in synthetic labels. "
            "Expected one label row per claim."
        )

    evaluation_df = scores.merge(
        labels,
        on=CLAIM_ID_COLUMN,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int(
        (evaluation_df["_merge"] != "both").sum()
    )

    if unmatched:
        raise ValueError(
            f"{unmatched:,} scored claims have no matching synthetic label. "
            "Check claim IDs in the scores and labels CSV files."
        )

    evaluation_df.drop(columns=["_merge"], inplace=True)

    if evaluation_df[LABEL_COLUMN].isna().any():
        raise ValueError(
            "Synthetic anomaly labels contain missing values after merging."
        )

    evaluation_df[LABEL_COLUMN] = (
        pd.to_numeric(
            evaluation_df[LABEL_COLUMN],
            errors="raise",
        )
        .astype(int)
    )

    invalid_labels = set(
        evaluation_df[LABEL_COLUMN].unique()
    ) - {0, 1}

    if invalid_labels:
        raise ValueError(
            f"Expected labels 0 and 1; found {invalid_labels}"
        )

    # --------------------------------------------------------
    # Overall dataset statistics
    # --------------------------------------------------------

    y_true = evaluation_df[LABEL_COLUMN]

    total_claims = len(evaluation_df)
    injected_anomalies = int(y_true.sum())
    normal_claims = total_claims - injected_anomalies
    anomaly_rate = (
        injected_anomalies / total_claims
        if total_claims
        else 0
    )

    print("\n" + "=" * 65)
    print("OVERALL DATASET")
    print("=" * 65)

    print(f"Claims evaluated: {total_claims:,}")
    print(f"Normal claims: {normal_claims:,}")
    print(f"Injected anomalies: {injected_anomalies:,}")
    print(f"Injected anomaly rate: {anomaly_rate:.2%}")


    # ========================================================
    # 4. PREPARE PREDICTIONS
    # ========================================================

    # Combined review queue.
    evaluation_df["combined_prediction"] = to_binary_flag(
        evaluation_df["flag_for_review"]
    )

    # Isolation Forest prediction.
    evaluation_df["ml_prediction"] = to_binary_flag(
        evaluation_df["ml_anomaly_flag"]
    )

    # Rule-based prediction uses ONLY actual Boolean rule columns.
    # This deliberately excludes rule_risk_points and rule_evidence.
    available_rule_columns = [
        column for column in RULE_COLUMNS
        if column in evaluation_df.columns
    ]

    if not available_rule_columns:
        raise ValueError(
            "No recognized Boolean rule columns were found in the fraud "
            "scores file. Expected at least one of: "
            f"{RULE_COLUMNS}\n"
            f"Available columns: {evaluation_df.columns.tolist()}"
        )

    print("\nRule columns used for rule-only evaluation:")
    for column in available_rule_columns:
        print(f"  - {column}")

    rule_flags = evaluation_df[available_rule_columns].copy()

    for column in available_rule_columns:
        rule_flags[column] = to_binary_flag(rule_flags[column])

    evaluation_df["rule_based_prediction"] = (
        rule_flags.any(axis=1).astype(int)
    )


    # ========================================================
    # 5. CALCULATE MODEL METRICS
    # ========================================================

    metric_rows = []

    model_definitions = [
        (
            "Rule + ML combined review queue",
            "combined_prediction",
        ),
        (
            "Isolation Forest only",
            "ml_prediction",
        ),
        (
            "Rule-based flags only",
            "rule_based_prediction",
        ),
    ]

    for model_name, prediction_column in model_definitions:
        metrics = calculate_metrics(
            y_true,
            evaluation_df[prediction_column],
            model_name,
        )
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows)

    print("\n" + "=" * 65)
    print("MODEL METRICS")
    print("=" * 65)

    print(metrics_df.to_string(index=False))


    # ========================================================
    # 6. COMBINED QUEUE CLASSIFICATION REPORT
    # ========================================================

    print("\n" + "=" * 65)
    print("COMBINED QUEUE CLASSIFICATION REPORT")
    print("=" * 65)

    print(
        classification_report(
            y_true,
            evaluation_df["combined_prediction"],
            labels=[0, 1],
            target_names=["Normal", "Synthetic anomaly"],
            zero_division=0,
            digits=3,
        )
    )


    # ========================================================
    # 7. SCENARIO-LEVEL EVALUATION
    # ========================================================

    print("\n" + "=" * 65)
    print("SCENARIO-LEVEL RESULTS")
    print("=" * 65)

    if "synthetic_scenario" in evaluation_df.columns:
        scenario_df = evaluation_df[
            evaluation_df[LABEL_COLUMN] == 1
        ].copy()

        scenario_df["combined_detected"] = (
            scenario_df["combined_prediction"]
        )

        scenario_df["ml_detected"] = (
            scenario_df["ml_prediction"]
        )

        scenario_df["rule_detected"] = (
            scenario_df["rule_based_prediction"]
        )

        scenario_results = (
            scenario_df
            .groupby("synthetic_scenario", dropna=False)
            .agg(
                injected_claims=(CLAIM_ID_COLUMN, "count"),
                detected_by_combined=("combined_detected", "sum"),
                detected_by_ml=("ml_detected", "sum"),
                detected_by_rules=("rule_detected", "sum"),
            )
            .reset_index()
        )

        scenario_results["combined_recall"] = (
            scenario_results["detected_by_combined"]
            / scenario_results["injected_claims"]
        )

        scenario_results["ml_recall"] = (
            scenario_results["detected_by_ml"]
            / scenario_results["injected_claims"]
        )

        scenario_results["rule_recall"] = (
            scenario_results["detected_by_rules"]
            / scenario_results["injected_claims"]
        )

        # Save recalls as decimal values in the CSV.
        # Convert to percentages only for console display.
        print_df = scenario_results.copy()

        for column in [
            "combined_recall",
            "ml_recall",
            "rule_recall",
        ]:
            print_df[column] = (
                print_df[column].map(
                    lambda value: f"{value:.2%}"
                )
            )

        print(print_df.to_string(index=False))

        scenario_results.to_csv(
            SCENARIO_FILE,
            index=False,
        )

    else:
        print(
            "WARNING: 'synthetic_scenario' was not found in synthetic "
            "labels. Scenario-level metrics were skipped."
        )

        scenario_results = pd.DataFrame()


    # ========================================================
    # 8. CLAIM-LEVEL EVALUATION OUTPUT
    # ========================================================

    evaluation_df["actual_label"] = y_true
    evaluation_df["actual_class"] = np.where(
        evaluation_df["actual_label"] == 1,
        "Synthetic anomaly",
        "Normal",
    )

    evaluation_df["combined_result"] = np.select(
        [
            (evaluation_df["actual_label"] == 1)
            & (evaluation_df["combined_prediction"] == 1),

            (evaluation_df["actual_label"] == 0)
            & (evaluation_df["combined_prediction"] == 1),

            (evaluation_df["actual_label"] == 1)
            & (evaluation_df["combined_prediction"] == 0),
        ],
        [
            "True Positive",
            "False Positive",
            "False Negative",
        ],
        default="True Negative",
    )

    evaluation_df.to_csv(
        CLAIM_EVALUATION_FILE,
        index=False,
    )


    # ========================================================
    # 9. SAVE MODEL METRICS
    # ========================================================

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )


    # ========================================================
    # 10. SUMMARY
    # ========================================================

    combined_metrics = metrics_df.loc[
        metrics_df["model"]
        == "Rule + ML combined review queue"
    ].iloc[0]

    flagged_count = int(
        evaluation_df["combined_prediction"].sum()
    )

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)

    print(f"Claims evaluated: {total_claims:,}")
    print(f"Injected anomalies: {injected_anomalies:,}")
    print(f"Claims in combined review queue: {flagged_count:,}")

    print(
        "Review queue rate: "
        f"{flagged_count / total_claims:.2%}"
        if total_claims
        else "Review queue rate: N/A"
    )

    print(
        f"Combined precision: {combined_metrics['precision']:.2%}"
    )
    print(
        f"Combined recall: {combined_metrics['recall']:.2%}"
    )
    print(
        f"Combined F1-score: {combined_metrics['f1_score']:.4f}"
    )

    print("\nFiles created:")
    print(f"  Metrics: {METRICS_FILE}")
    if not scenario_results.empty:
        print(f"  Scenario evaluation: {SCENARIO_FILE}")
    print(f"  Claim-level evaluation: {CLAIM_EVALUATION_FILE}")

    print("\nEvaluation completed successfully.")
    print(
        "Note: These metrics evaluate synthetic anomalies, "
        "not confirmed real-world fraud."
    )


if __name__ == "__main__":
    main()
