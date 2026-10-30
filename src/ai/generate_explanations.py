
"""
Healthcare Insurance Claims Fraud Detection
Step 4: Fraud Explanation Generator

Reads existing claim fraud scores and generates:
- Plain-English fraud explanations
- Triggered rule summaries
- Recommended investigation actions

Does not modify the existing fraud detection engine or risk scores.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "outputs" / "claim_fraud_scores.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "claim_fraud_explanations.csv"


# ---------------------------------------------------------
# 2. FRAUD RULE DESCRIPTIONS AND ACTIONS
# ---------------------------------------------------------

RULE_DESCRIPTIONS = {
    "rule_duplicate_claim": {
        "explanation": (
            "Potential duplicate claim submission was flagged."
        ),
        "action": (
            "Compare this claim with the suspected matching claim. "
            "Verify member, provider, service date, procedure code, "
            "and submitted amount."
        ),
    },

    "rule_high_amount": {
        "explanation": (
            "The billed amount triggered the configured high-amount rule."
        ),
        "action": (
            "Compare the billed amount with the procedure reference amount "
            "and review the supporting documentation."
        ),
    },

    "rule_excessive_units": {
        "explanation": (
            "The units billed triggered the configured excessive-units rule."
        ),
        "action": (
            "Verify billed units against the procedure, clinical documentation, "
            "and applicable billing guidance."
        ),
    },

    "rule_provider_billing_outlier": {
        "explanation": (
            "The provider's billing pattern triggered the configured outlier rule."
        ),
        "action": (
            "Review the provider's claim history and compare similar procedures "
            "and billing amounts."
        ),
    },

    "rule_negative_submission_lag": {
        "explanation": (
            "The claim triggered the configured submission-timing rule."
        ),
        "action": (
            "Verify the service and submission dates and check applicable "
            "timely-filing requirements."
        ),
    },

    "rule_line_amount_mismatch": {
        "explanation": (
            "A claim line amount mismatch was flagged."
        ),
        "action": (
            "Reconcile the claim-level billed amount with its line items "
            "and supporting documentation."
        ),
    },
}

RULE_COLUMNS = list(RULE_DESCRIPTIONS.keys())


# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------

def as_bool(value):
    """Convert CSV boolean-like values safely into Python booleans."""

    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true", "1", "yes", "y"
    }


def safe_text(value, default="Not available"):
    """Return readable text for missing or empty values."""

    if pd.isna(value):
        return default

    value = str(value).strip()

    if not value:
        return default

    return value


# ---------------------------------------------------------
# 4. GENERATE EXPLANATION FOR ONE CLAIM
# ---------------------------------------------------------

def explain_claim(row):
    """
    Generate an explanation using existing rule flags,
    risk scores, and ML anomaly indicators.
    """

    triggered_rules = []

    for rule_column in RULE_COLUMNS:
        if rule_column in row.index:
            if as_bool(row[rule_column]):
                triggered_rules.append(rule_column)

    ml_flag = as_bool(row.get("ml_anomaly_flag", False))

    review_flag = as_bool(
        row.get("flag_for_review", False)
    )

    risk_level = safe_text(
        row.get("risk_level"),
        default="UNKNOWN"
    ).upper()

    risk_score = pd.to_numeric(
        row.get("risk_score"),
        errors="coerce"
    )

    # -----------------------------------------------------
    # Build explanation evidence
    # -----------------------------------------------------

    explanation_parts = []

    for rule_column in triggered_rules:
        description = RULE_DESCRIPTIONS[rule_column]["explanation"]
        explanation_parts.append(description)

    if ml_flag:
        explanation_parts.append(
            "The Isolation Forest model marked this claim as an anomaly "
            "based on its feature pattern. This signal does not identify "
            "a specific cause by itself."
        )

    # Include existing investigation reasons if available.
    existing_reasons = safe_text(
        row.get("investigation_reasons"),
        default=""
    )

    if existing_reasons and existing_reasons.lower() not in {
        "nan", "none"
    }:
        explanation_parts.append(
            f"Existing investigation reason: {existing_reasons}"
        )

    if not explanation_parts:
        if review_flag:
            explanation_parts.append(
                "This claim is in the review queue, but no individual "
                "rule flag or ML anomaly flag was identified in the "
                "available fields."
            )
        else:
            explanation_parts.append(
                "No configured rule flag or ML anomaly flag was identified "
                "in the available fields."
            )

    fraud_explanation = " ".join(explanation_parts)

    # -----------------------------------------------------
    # Build recommended investigation actions
    # -----------------------------------------------------

    action_parts = []

    for rule_column in triggered_rules:
        action = RULE_DESCRIPTIONS[rule_column]["action"]
        action_parts.append(action)

    if ml_flag:
        action_parts.append(
            "Inspect the underlying claim features and compare the claim "
            "with similar claims. The ML anomaly flag alone does not "
            "establish fraud."
        )

    if review_flag and not action_parts:
        action_parts.append(
            "Review the complete claim record and source documentation "
            "to establish why the claim entered the investigation queue."
        )

    if not action_parts:
        action_parts.append(
            "No specific investigation action is suggested by the "
            "available flags. Follow normal claim review procedures."
        )

    recommended_actions = " ".join(action_parts)

    # -----------------------------------------------------
    # Format risk score for display
    # -----------------------------------------------------

    if pd.notna(risk_score):
        formatted_risk_score = f"{risk_score:.1f}/100"
    else:
        formatted_risk_score = "Not available"

    # -----------------------------------------------------
    # Return explanation fields
    # -----------------------------------------------------

    return pd.Series({
        "explanation_risk_score": formatted_risk_score,
        "explanation_risk_level": risk_level,
        "is_flagged_for_review": review_flag,
        "triggered_rule_count": len(triggered_rules),
        "triggered_rules": (
            "; ".join(triggered_rules)
            if triggered_rules
            else "None"
        ),
        "explanation_ml_anomaly_flag": ml_flag,
        "fraud_explanation": fraud_explanation,
        "recommended_investigation_actions": recommended_actions,
        "review_disclaimer": (
            "This explanation is generated from synthetic POC data "
            "and existing detection outputs. It is not a fraud determination. "
            "Verify source records before taking action."
        ),
    })


# ---------------------------------------------------------
# 5. MAIN EXECUTION
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("HEALTHCARE CLAIMS FRAUD EXPLANATION GENERATOR")
    print("=" * 65)

    # Check that the input exists.
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_PATH}\n"
            "Ensure outputs/claim_fraud_scores.csv exists."
        )

    print(f"\nReading input file:\n{INPUT_PATH}")

    # Load the existing fraud scores.
    df = pd.read_csv(INPUT_PATH)

    print(f"Claims loaded: {len(df):,}")

    # Validate the minimum required schema.
    required_columns = {
        "claim_id",
        "risk_score",
        "risk_level",
        "flag_for_review",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Generate explanation fields.
    print("\nGenerating claim explanations...")

    explanation_df = df.apply(
        explain_claim,
        axis=1
    )

    # Append the new explanation fields to the existing data.
    # This writes a NEW file and does not overwrite the input.
    output_df = pd.concat(
        [df, explanation_df],
        axis=1
    )

    # Save results.
    output_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # Print summary.
    review_count = int(
        output_df["is_flagged_for_review"].sum()
    )

    ml_anomaly_count = int(
        output_df["explanation_ml_anomaly_flag"].sum()
    )

    print("\n" + "=" * 65)
    print("EXPLANATION GENERATION COMPLETED")
    print("=" * 65)

    print(f"Total claims processed: {len(output_df):,}")
    print(f"Claims flagged for review: {review_count:,}")
    print(f"Claims with ML anomaly flags: {ml_anomaly_count:,}")

    print(f"\nOutput file:\n{OUTPUT_PATH}")

    print("\nNew columns generated:")
    print("- explanation_risk_score")
    print("- explanation_risk_level")
    print("- is_flagged_for_review")
    print("- triggered_rule_count")
    print("- triggered_rules")
    print("- explanation_ml_anomaly_flag")
    print("- fraud_explanation")
    print("- recommended_investigation_actions")
    print("- review_disclaimer")

    print("\nNext: Open claim_fraud_explanations.csv in Excel.")
    print("=" * 65)


if __name__ == "__main__":
    main()
