# Healthcare Claims AI — Generated Synthetic Data Bundle

## Contents
- `data/raw/`: copies of the original uploaded CSV files, preserved unchanged.
- `data/processed/claims_extended.csv`: original claims plus 10,000 additional synthetic claims.
- `data/processed/claim_line_items_extended.csv`: original line items plus generated line items; 8% of new claims have a second line.
- `data/processed/synthetic_labels_extended.csv`: original labels plus labels for generated claims.
- `data/processed/claim_denial_reasons.csv`: one record per claim; reasons are populated for Denied and Partially Paid statuses.
- `data/processed/claim_status_history.csv`: synthetic submission and status/settlement events.
- `data/processed/claim_payments.csv`: synthetic payment records for claims with positive paid amounts.
- `data/processed/denial_reason_catalog.csv`: reason codes and suggested next steps.
- `documents/`: generated synthetic business documents and copies of the uploaded reference PDFs.

## Important limitations
- All generated records and documents are fictional and for software testing only.
- New claims use IDs prefixed `CLM26-` and service dates throughout calendar year 2026; submission dates may extend into 2027 due to long submission lags.
- Synthetic denial reasons are illustrative and are not authoritative adjudication decisions.
- Status history and payment dates are generated approximations, not recovered operational events.
- Synthetic anomaly labels are intended only for post-scoring evaluation and must be excluded from model features and Text-to-SQL production metrics unless the question explicitly asks for evaluation labels.
- The existing source files in `data/raw/` are preserved unchanged.
