from src.ai.ai_investigator import generate_claim_explanation
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

from src.ai.investigation_service import (
    find_claim,
    get_investigation_queue,
)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SCORES_FILE = OUTPUT_DIR / "claim_fraud_scores.csv"
EXPLANATIONS_FILE = OUTPUT_DIR / "claim_fraud_explanations.csv"
METRICS_FILE = OUTPUT_DIR / "fraud_model_metrics.csv"


st.set_page_config(
    page_title="Healthcare Claims AI Investigator",
    page_icon="🏥",
    layout="wide",
)

st.title("Healthcare Claims Fraud Detection")
st.caption(
    "AI-assisted investigation workspace | "
    "Risk indicators support review and do not establish fraud."
)


@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


scores = load_csv(str(SCORES_FILE))
explanations = load_csv(str(EXPLANATIONS_FILE))
metrics = load_csv(str(METRICS_FILE))

if scores.empty:
    st.error(
        "Fraud scores were not found. Run your existing fraud scoring "
        "pipeline before launching the dashboard."
    )
    st.stop()


# Merge explanations into scores for consistent dashboard display.
if (
    not explanations.empty
    and "claim_id" in scores.columns
    and "claim_id" in explanations.columns
):
    explanation_columns = [
        col for col in explanations.columns
        if col == "claim_id" or col not in scores.columns
    ]
    scores = scores.merge(
        explanations[explanation_columns],
        on="claim_id",
        how="left",
    )


def first_existing_column(frame, candidates):
    for column in candidates:
        if column in frame.columns:
            return column
    return None


risk_column = first_existing_column(
    scores,
    ["explanation_risk_level", "risk_level"],
)
flag_column = first_existing_column(
    scores,
    ["is_flagged_for_review", "flag_for_review"],
)
score_column = first_existing_column(
    scores,
    ["explanation_risk_score", "risk_score"],
)

if flag_column:
    flagged = (
        scores[flag_column]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )
else:
    flagged = pd.Series(False, index=scores.index)

if risk_column:
    high_risk = (
        scores[risk_column]
        .astype(str)
        .str.upper()
        .eq("HIGH")
    )
else:
    high_risk = pd.Series(False, index=scores.index)


# -------------------------------------------------------
# Overview metrics
# -------------------------------------------------------
st.header("Portfolio overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Claims scored", f"{len(scores):,}")
c2.metric("Flagged for review", f"{int(flagged.sum()):,}")
c3.metric("High-risk claims", f"{int(high_risk.sum()):,}")
c4.metric(
    "Review rate",
    f"{flagged.mean() * 100:.2f}%" if len(scores) else "0%",
)

left, right = st.columns(2)

with left:
    st.subheader("Risk-level distribution")
    if risk_column:
        risk_counts = (
            scores[risk_column]
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
            .rename_axis("Risk level")
            .reset_index(name="Claims")
        )
        fig = px.bar(
            risk_counts,
            x="Risk level",
            y="Claims",
            title="Claims by risk level",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No recognized risk-level column was found.")

with right:
    st.subheader("Flagged versus unflagged")
    flag_counts = pd.DataFrame({
        "Status": ["Flagged for review", "Not flagged"],
        "Claims": [
            int(flagged.sum()),
            int((~flagged).sum()),
        ],
    })
    fig = px.pie(
        flag_counts,
        names="Status",
        values="Claims",
        title="Review queue composition",
        hole=0.4,
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------
# Investigation queue
# -------------------------------------------------------
st.divider()
st.header("Investigation queue")

queue = scores.loc[flagged].copy()

if risk_column:
    risk_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2,
        "MINIMAL": 3,
    }
    queue["_risk_order"] = (
        queue[risk_column].astype(str).str.upper()
        .map(risk_order)
        .fillna(4)
    )
    queue = queue.sort_values("_risk_order").drop(
        columns=["_risk_order"]
    )

search_text = st.text_input(
    "Filter by claim ID, member ID or provider ID",
    placeholder="e.g. CLM00013845",
)

if search_text:
    searchable = [
        col for col in ["claim_id", "member_id", "provider_id"]
        if col in queue.columns
    ]
    if searchable:
        mask = pd.Series(False, index=queue.index)
        for column in searchable:
            mask |= (
                queue[column].astype(str)
                .str.contains(search_text, case=False, na=False)
            )
        queue = queue[mask]

display_columns = [
    col for col in [
        "claim_id",
        "member_id",
        "provider_id",
        "billed_amount",
        "amount_to_reference_ratio",
        "risk_level",
        "explanation_risk_level",
        "risk_score",
        "explanation_risk_score",
        "triggered_rules",
        "fraud_explanation",
        "investigation_reasons",
    ]
    if col in queue.columns
]

st.dataframe(
    queue[display_columns].head(200),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"Showing up to 200 matching claims out of {len(queue):,}."
)


# -------------------------------------------------------
# Individual claim investigation
# -------------------------------------------------------

# -------------------------------------------------------
# Individual claim investigation with AI explanation
# -------------------------------------------------------
st.divider()
st.header("AI-powered claim investigation")

st.write(
    "Enter a claim ID to retrieve its fraud evidence and generate "
    "an AI explanation using the claim data and indexed policy documents."
)

claim_id = st.text_input(
    "Claim ID",
    placeholder="e.g. CLM00013845",
    key="investigation_claim_id",
)

if st.button("Retrieve claim evidence", type="primary"):
    # Clear previous results so a different claim cannot show stale output.
    st.session_state.pop("selected_claim", None)
    st.session_state.pop("ai_claim_explanation", None)
    st.session_state.pop("ai_claim_error", None)

    normalized_claim_id = claim_id.strip()

    if not normalized_claim_id:
        st.session_state["ai_claim_error"] = (
            "Please check the claim ID again. Claim not found in the list."
        )
    else:
        claim = find_claim(normalized_claim_id)

        if not claim.get("found"):
            st.session_state["ai_claim_error"] = (
                "Please check the claim ID again. "
                "Claim not found in the list."
            )
        else:
            st.session_state["selected_claim"] = claim

            try:
                with st.spinner(
                    "Analyzing claim evidence and retrieving policy documents..."
                ):
                    answer = generate_claim_explanation(claim)

                st.session_state["ai_claim_explanation"] = answer

            except Exception as exc:
                # Show an operational error instead of inventing an AI answer.
                st.session_state["ai_claim_error"] = (
                    "The claim was found, but the AI explanation could not "
                    "be generated. Check your NVIDIA API configuration, "
                    "model access, network connection and ChromaDB index. "
                    f"Technical details: {exc}"
                )


# Display errors from the latest request only.
if st.session_state.get("ai_claim_error"):
    st.error(st.session_state["ai_claim_error"])


# Display the generated AI explanation.
claim = st.session_state.get("selected_claim")
ai_answer = st.session_state.get("ai_claim_explanation")

if claim:
    st.subheader(f"Claim: {claim.get('claim_id', 'Unknown')}")

    risk_level = claim.get(
        "explanation_risk_level",
        claim.get("risk_level", "Unknown"),
    )
    risk_score = claim.get(
        "explanation_risk_score",
        claim.get("risk_score", "Unavailable"),
    )
    flagged_value = claim.get(
        "is_flagged_for_review",
        claim.get("flag_for_review", "Unknown"),
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk level", str(risk_level))
    m2.metric("Risk score", str(risk_score))
    m3.metric("Flagged for review", str(flagged_value))

    if ai_answer:
        st.subheader("AI investigation explanation")
        st.markdown(ai_answer)

    with st.expander("View original claim evidence"):
        details_columns = [
            col for col in [
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
                "reference_amount",
                "amount_to_reference_ratio",
                "provider_median_billed",
                "provider_billing_ratio",
                "rule_evidence",
                "triggered_rules",
                "fraud_explanation",
                "recommended_investigation_actions",
                "investigation_reasons",
            ]
            if col in claim
        ]

        details = pd.DataFrame({
            "Field": details_columns,
            "Value": [str(claim.get(col, "")) for col in details_columns],
        })

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "AI-generated analysis is decision support only. "
        "Risk indicators are not proof of fraud. Verify findings "
        "against source records and applicable policy."
    )


# -------------------------------------------------------
# Model evaluation
# -------------------------------------------------------
st.divider()
st.header("Model evaluation")

if not metrics.empty:
    st.dataframe(metrics, use_container_width=True, hide_index=True)
else:
    st.info(
        "Model metrics CSV was not found. Run the existing evaluation "
        "script to display precision, recall, F1 and related metrics."
    )

st.caption(
    "This dashboard reads existing project outputs. "
    "It does not retrain models or change fraud-scoring thresholds."
)
