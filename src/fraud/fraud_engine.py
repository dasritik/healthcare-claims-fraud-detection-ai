
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_csv_datasets


OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initial POC thresholds; tune after reviewing synthetic scenarios.
HIGH_AMOUNT_RATIO = 3.0
HIGH_UNITS_MULTIPLIER = 1.0
MIN_PROVIDER_CLAIMS = 10
MIN_DUPLICATE_GROUP_SIZE = 2
CONTAMINATION = 0.03


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def safe_dates(series):
    return pd.to_datetime(series, errors="coerce")


def prepare_claim_features(datasets):
    """Join claim, procedure and provider data and build features."""
    claims = datasets["claims"].copy()
    procedures = datasets["procedures"].copy()
    providers = datasets["providers"].copy()
    line_items = datasets["claim_line_items"].copy()

    # Normalize IDs and key categorical columns.
    for df, columns in [
        (claims, ["claim_id", "member_id", "provider_id", "procedure_code"]),
        (procedures, ["procedure_code"]),
        (providers, ["provider_id"]),
        (line_items, ["claim_id", "line_procedure_code"]),
    ]:
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype("string").str.strip()

    # Convert numeric fields.
    numeric_claim_cols = [
        "units", "billed_amount", "allowed_amount", "paid_amount"
    ]
    for col in numeric_claim_cols:
        claims[col] = safe_numeric(claims[col])

    procedures["reference_amount"] = safe_numeric(
        procedures["reference_amount"]
    )
    procedures["maximum_units_guideline"] = safe_numeric(
        procedures["maximum_units_guideline"]
    )

    # Aggregate line items by claim for additional checks.
    line_items["line_units"] = safe_numeric(line_items["line_units"])
    line_items["line_billed_amount"] = safe_numeric(
        line_items["line_billed_amount"]
    )

    line_agg = (
        line_items.groupby("claim_id", dropna=False)
        .agg(
            line_item_count=("claim_id", "size"),
            total_line_units=("line_units", "sum"),
            total_line_billed=("line_billed_amount", "sum"),
        )
        .reset_index()
    )

    # Keep one procedure reference row per code to avoid join multiplication.
    procedures = procedures.drop_duplicates(
        subset=["procedure_code"], keep="first"
    )

    df = claims.merge(
        procedures[
            [
                "procedure_code",
                "description",
                "typical_specialty",
                "reference_amount",
                "maximum_units_guideline",
            ]
        ],
        on="procedure_code",
        how="left",
        validate="many_to_one",
    )

    df = df.merge(
        providers[
            [
                "provider_id",
                "specialty",
                "provider_type",
                "network_status",
                "provider_start_date",
            ]
        ].drop_duplicates(subset=["provider_id"]),
        on="provider_id",
        how="left",
        validate="many_to_one",
    )

    df = df.merge(
        line_agg,
        on="claim_id",
        how="left",
        validate="one_to_one",
    )

    df["service_date"] = safe_dates(df["service_date"])
    df["submission_date"] = safe_dates(df["submission_date"])
    df["provider_start_date"] = safe_dates(df["provider_start_date"])

    df["submission_lag_days"] = (
        df["submission_date"] - df["service_date"]
    ).dt.days

    df["amount_to_reference_ratio"] = (
        df["billed_amount"] / df["reference_amount"].replace(0, np.nan)
    )

    df["paid_to_billed_ratio"] = (
        df["paid_amount"] / df["billed_amount"].replace(0, np.nan)
    )

    df["allowed_to_billed_ratio"] = (
        df["allowed_amount"] / df["billed_amount"].replace(0, np.nan)
    )

    df["provider_tenure_days"] = (
        df["service_date"] - df["provider_start_date"]
    ).dt.days

    # Provider-level statistics provide context for unusual billing.
    provider_stats = (
        df.groupby("provider_id", dropna=False)
        .agg(
            provider_claim_count=("claim_id", "count"),
            provider_mean_billed=("billed_amount", "mean"),
            provider_median_billed=("billed_amount", "median"),
        )
        .reset_index()
    )

    df = df.merge(
        provider_stats,
        on="provider_id",
        how="left",
        validate="many_to_one",
    )

    df["provider_billing_ratio"] = (
        df["billed_amount"]
        / df["provider_median_billed"].replace(0, np.nan)
    )

    # A duplicate candidate uses the same member, provider, service date,
    # procedure and billed amount. It is a review signal, not proof.
    duplicate_keys = [
        "member_id",
        "provider_id",
        "service_date",
        "procedure_code",
        "billed_amount",
    ]

    df["duplicate_candidate"] = (
        df.groupby(duplicate_keys, dropna=False)["claim_id"]
        .transform("size")
        .ge(MIN_DUPLICATE_GROUP_SIZE)
    )

    # Basic consistency checks between claim header and line items.
    df["line_amount_mismatch"] = (
        df["total_line_billed"].notna()
        & df["billed_amount"].notna()
        & (
            (df["total_line_billed"] - df["billed_amount"]).abs()
            > 0.02
        )
    )

    return df


def apply_detection_rules(df):
    """Create interpretable rule flags and supporting evidence."""
    df = df.copy()

    df["rule_duplicate_claim"] = df["duplicate_candidate"].fillna(False)

    df["rule_high_amount"] = (
        df["amount_to_reference_ratio"] > HIGH_AMOUNT_RATIO
    ).fillna(False)

    df["rule_excessive_units"] = (
        df["maximum_units_guideline"].notna()
        & df["units"].notna()
        & (
            df["units"]
            > df["maximum_units_guideline"] * HIGH_UNITS_MULTIPLIER
        )
    ).fillna(False)

    df["rule_provider_billing_outlier"] = (
        (df["provider_claim_count"] >= MIN_PROVIDER_CLAIMS)
        & (df["provider_billing_ratio"] >= 2.0)
    ).fillna(False)

    df["rule_negative_submission_lag"] = (
        df["submission_lag_days"] < 0
    ).fillna(False)

    df["rule_line_amount_mismatch"] = (
        df["line_amount_mismatch"].fillna(False)
    )

    # Store evidence text to help investigators understand flags.
    evidence = []

    for row in df.itertuples(index=False):
        reasons = []

        if row.rule_duplicate_claim:
            reasons.append("Possible duplicate claim key")

        if row.rule_high_amount:
            reasons.append(
                f"Billed amount is {row.amount_to_reference_ratio:.1f}x "
                "the procedure reference amount"
            )

        if row.rule_excessive_units:
            reasons.append(
                f"Units ({row.units:g}) exceed the procedure guideline "
                f"({row.maximum_units_guideline:g})"
            )

        if row.rule_provider_billing_outlier:
            reasons.append(
                f"Billed amount is {row.provider_billing_ratio:.1f}x "
                "the provider's median billed amount"
            )

        if row.rule_negative_submission_lag:
            reasons.append("Submission date precedes service date")

        if row.rule_line_amount_mismatch:
            reasons.append(
                "Claim billed amount differs from summed line-item billed "
                "amount by more than 0.02"
            )

        evidence.append("; ".join(reasons))

    df["rule_evidence"] = evidence
    return df


def apply_isolation_forest(df):
    """Fit an unsupervised anomaly model without synthetic labels."""
    df = df.copy()

    feature_cols = [
        "billed_amount",
        "allowed_amount",
        "paid_amount",
        "units",
        "reference_amount",
        "maximum_units_guideline",
        "amount_to_reference_ratio",
        "submission_lag_days",
        "provider_claim_count",
        "provider_mean_billed",
        "provider_billing_ratio",
        "line_item_count",
        "total_line_units",
        "total_line_billed",
    ]

    # Explicitly exclude IDs, claim outcomes, and synthetic labels.
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=200,
                    contamination=CONTAMINATION,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    predictions = model.fit_predict(X)
    anomaly_strength = -model.named_steps["model"].score_samples(
        model.named_steps["scaler"].transform(
            model.named_steps["imputer"].transform(X)
        )
    )

    df["ml_anomaly_flag"] = predictions == -1
    df["ml_anomaly_strength"] = anomaly_strength

    # Convert anomaly strength into a percentile from 0 to 100.
    df["ml_anomaly_percentile"] = (
        df["ml_anomaly_strength"].rank(pct=True) * 100
    )

    return df


def create_risk_scores(df):
    """Combine rule evidence and ML anomaly signals into a 0-100 score."""
    df = df.copy()

    rule_cols = [
        "rule_duplicate_claim",
        "rule_high_amount",
        "rule_excessive_units",
        "rule_provider_billing_outlier",
        "rule_negative_submission_lag",
        "rule_line_amount_mismatch",
    ]

    # Transparent initial weights for this POC.
    weights = {
        "rule_duplicate_claim": 30,
        "rule_high_amount": 20,
        "rule_excessive_units": 20,
        "rule_provider_billing_outlier": 15,
        "rule_negative_submission_lag": 10,
        "rule_line_amount_mismatch": 10,
    }

    df["rule_risk_points"] = sum(
        df[col].astype(int) * weight
        for col, weight in weights.items()
    )

    df["rule_risk_points"] = df["rule_risk_points"].clip(upper=80)

    # ML contributes up to 20 points. A percentile is a ranking,
    # not a probability of fraud.
    df["ml_risk_points"] = (
        (df["ml_anomaly_percentile"] - 80).clip(lower=0) / 20 * 20
    )

    df["risk_score"] = (
        df["rule_risk_points"] + df["ml_risk_points"]
    ).clip(0, 100).round(1)

    df["risk_level"] = np.select(
        [
            df["risk_score"] >= 70,
            df["risk_score"] >= 40,
            df["risk_score"] >= 20,
        ],
        [
            "HIGH",
            "MEDIUM",
            "LOW",
        ],
        default="MINIMAL",
    )

    df["investigation_reasons"] = df["rule_evidence"].fillna("")

    df.loc[
        df["ml_anomaly_flag"]
        & df["investigation_reasons"].eq(""),
        "investigation_reasons",
    ] = "Unusual multivariate claim pattern identified by Isolation Forest"

    df.loc[
        df["ml_anomaly_flag"]
        & df["investigation_reasons"].ne("")
        & ~df["investigation_reasons"].str.contains(
            "Isolation Forest", regex=False
        ),
        "investigation_reasons",
    ] += "; Unusual multivariate pattern identified by Isolation Forest"

    df["flag_for_review"] = (
        (df["risk_score"] >= 40)
        | df["rule_duplicate_claim"]
        | df["rule_line_amount_mismatch"]
    )

    return df


def main():
    datasets, manifest = load_csv_datasets()

    required = [
        "claims",
        "claim_line_items",
        "procedures",
        "providers",
    ]
    missing = [name for name in required if name not in datasets]

    if missing:
        raise ValueError(
            "Required datasets are missing: " + ", ".join(missing)
        )

    print("\nPreparing claim features...")
    df = prepare_claim_features(datasets)

    print("Applying detection rules...")
    df = apply_detection_rules(df)

    print("Running Isolation Forest...")
    df = apply_isolation_forest(df)

    print("Calculating risk scores...")
    df = create_risk_scores(df)

    # Save full output and investigation queue.
    full_path = OUTPUT_DIR / "claim_fraud_scores.csv"
    review_path = OUTPUT_DIR / "fraud_investigation_queue.csv"
    summary_path = OUTPUT_DIR / "fraud_detection_summary.csv"

    # The synthetic labels are not used by any detection step.
    # Keep them out of the main scoring output; merge them only later
    # in a separate evaluation step.
    export_cols = [
        "claim_id",
        "member_id",
        "provider_id",
        "service_date",
        "submission_date",
        "procedure_code",
        "diagnosis_code",
        "units",
        "billed_amount",
        "allowed_amount",
        "paid_amount",
        "claim_status",
        "specialty",
        "network_status",
        "reference_amount",
        "amount_to_reference_ratio",
        "provider_claim_count",
        "provider_median_billed",
        "provider_billing_ratio",
        "line_item_count",
        "total_line_units",
        "total_line_billed",
        *[
            col for col in df.columns
            if col.startswith("rule_")
        ],
        "ml_anomaly_flag",
        "ml_anomaly_strength",
        "ml_anomaly_percentile",
        "rule_risk_points",
        "ml_risk_points",
        "risk_score",
        "risk_level",
        "flag_for_review",
        "investigation_reasons",
    ]

    export_cols = list(dict.fromkeys(
        col for col in export_cols if col in df.columns
    ))

    df[export_cols].to_csv(full_path, index=False)

    review = df[df["flag_for_review"]].sort_values(
        ["risk_score", "billed_amount"],
        ascending=[False, False],
    )
    review[export_cols].to_csv(review_path, index=False)

    summary = (
        df.groupby("risk_level", dropna=False)
        .agg(
            claim_count=("claim_id", "count"),
            average_risk_score=("risk_score", "mean"),
            total_billed_amount=("billed_amount", "sum"),
            claims_flagged=("flag_for_review", "sum"),
        )
        .reset_index()
    )
    summary["average_risk_score"] = summary[
        "average_risk_score"
    ].round(2)
    summary.to_csv(summary_path, index=False)

    print("\n=== FRAUD DETECTION RESULTS ===")
    print(f"Claims scored: {len(df):,}")
    print(f"Claims flagged for review: {int(df['flag_for_review'].sum()):,}")
    print(f"High-risk claims: {int((df['risk_level'] == 'HIGH').sum()):,}")
    print(f"ML anomalies: {int(df['ml_anomaly_flag'].sum()):,}")

    print("\nRisk level summary:")
    print(summary.to_string(index=False))

    print("\nFiles created:")
    print(f"  Full scores: {full_path}")
    print(f"  Investigation queue: {review_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
