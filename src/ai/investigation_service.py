
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SCORES_FILE = OUTPUT_DIR / "claim_fraud_scores.csv"
EXPLANATIONS_FILE = OUTPUT_DIR / "claim_fraud_explanations.csv"
QUEUE_FILE = OUTPUT_DIR / "fraud_investigation_queue.csv"


def load_csv(path: Path) -> pd.DataFrame:
    """Load a required CSV and fail with a useful message if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run the existing fraud scoring and explanation scripts first."
        )
    return pd.read_csv(path)


def find_claim(claim_id: str) -> dict:
    """Find a claim and combine its fraud score with explanation fields."""
    scores = load_csv(SCORES_FILE)

    if "claim_id" not in scores.columns:
        raise ValueError("claim_fraud_scores.csv has no claim_id column.")

    matched = scores[
        scores["claim_id"].astype(str).str.strip()
        == str(claim_id).strip()
    ]

    if matched.empty:
        return {
            "found": False,
            "claim_id": claim_id,
            "message": "Claim ID not found in the fraud scores file."
        }

    result = matched.iloc[0].to_dict()

    if EXPLANATIONS_FILE.exists():
        explanations = pd.read_csv(EXPLANATIONS_FILE)

        if "claim_id" in explanations.columns:
            explanation_match = explanations[
                explanations["claim_id"].astype(str).str.strip()
                == str(claim_id).strip()
            ]

            if not explanation_match.empty:
                explanation = explanation_match.iloc[0].to_dict()
                result.update(explanation)

    result["found"] = True
    result["claim_id"] = claim_id
    result["disclaimer"] = (
        "A risk flag indicates a need for review. It is not proof of fraud."
    )

    return result


def get_investigation_queue(limit: int = 20) -> pd.DataFrame:
    """Return the highest-priority claims from the existing review queue."""
    queue = load_csv(QUEUE_FILE)

    if queue.empty:
        return queue

    # Use existing risk scores when available.
    possible_score_columns = [
        "explanation_risk_score",
        "risk_score",
        "fraud_risk_score",
    ]

    score_column = next(
        (col for col in possible_score_columns if col in queue.columns),
        None
    )

    if score_column:
        queue = queue.sort_values(score_column, ascending=False)

    return queue.head(max(1, limit))


def build_investigation_context(claim_id: str) -> str:
    """Format claim evidence into text suitable for a future RAG prompt."""
    claim = find_claim(claim_id)

    if not claim.get("found"):
        return claim["message"]

    preferred_fields = [
        "claim_id",
        "risk_level",
        "explanation_risk_level",
        "explanation_risk_score",
        "is_flagged_for_review",
        "triggered_rule_count",
        "triggered_rules",
        "explanation_ml_anomaly_flag",
        "fraud_explanation",
        "recommended_investigation_actions",
    ]

    lines = [
        f"{field}: {claim[field]}"
        for field in preferred_fields
        if field in claim and pd.notna(claim[field])
    ]

    lines.append(
        "Important: These are risk indicators for investigation, "
        "not a confirmed finding of fraud."
    )

    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 65)
    print("Healthcare Claims - Investigation Service Test")
    print("=" * 65)

    try:
        queue = get_investigation_queue(limit=5)
        print("\nTop claims in the review queue:")
        print(queue.to_string(index=False))

        scores = load_csv(SCORES_FILE)
        if not scores.empty and "claim_id" in scores.columns:
            sample_claim_id = str(scores.iloc[0]["claim_id"])

            print(f"\nSample claim lookup: {sample_claim_id}")
            print(build_investigation_context(sample_claim_id))

    except Exception as exc:
        print(f"\nERROR: {exc}")
