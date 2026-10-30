
from pathlib import Path
import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Actual location of the generated dataset bundle
DATA_BUNDLE_DIR = (
    PROJECT_ROOT / "data" / "healthcare_claims_ai_generated"
)

RAW_DIR = DATA_BUNDLE_DIR / "raw"
PROCESSED_DIR = DATA_BUNDLE_DIR / "processed"


# Prefer extended datasets; otherwise use original raw files.
FILE_CANDIDATES = {
    "claims": [
        PROCESSED_DIR / "claims_extended.csv",
        RAW_DIR / "claims.csv",
    ],
    "claim_line_items": [
        PROCESSED_DIR / "claim_line_items_extended.csv",
        RAW_DIR / "claim_line_items.csv",
    ],
    "members": [
        PROCESSED_DIR / "members.csv",
        RAW_DIR / "members.csv",
    ],
    "providers": [
        PROCESSED_DIR / "providers.csv",
        RAW_DIR / "providers.csv",
    ],
    "procedures": [
        PROCESSED_DIR / "procedures.csv",
        RAW_DIR / "procedures.csv",
    ],
    "synthetic_labels": [
        PROCESSED_DIR / "synthetic_labels_extended.csv",
        RAW_DIR / "synthetic_labels.csv",
    ],
    "claim_denial_reasons": [
        PROCESSED_DIR / "claim_denial_reasons.csv",
        RAW_DIR / "claim_denial_reasons.csv",
    ],
    "denial_reason_catalog": [
        PROCESSED_DIR / "denial_reason_catalog.csv",
        RAW_DIR / "denial_reason_catalog.csv",
    ],
    "claim_status_history": [
        PROCESSED_DIR / "claim_status_history.csv",
        RAW_DIR / "claim_status_history.csv",
    ],
    "claim_payments": [
        PROCESSED_DIR / "claim_payments.csv",
        RAW_DIR / "claim_payments.csv",
    ],
}


def find_file(candidates):
    """Find the first available file."""
    for path in candidates:
        if path.exists():
            return path
    return None


def load_csv_datasets():
    """Load available CSV datasets into Pandas DataFrames."""
    data = {}
    manifest_rows = []

    for dataset_name, candidates in FILE_CANDIDATES.items():
        file_path = find_file(candidates)

        if file_path is None:
            manifest_rows.append({
                "dataset": dataset_name,
                "file_path": "",
                "status": "MISSING",
                "rows": 0,
                "columns": 0,
                "error": "No candidate file found",
            })
            print(f"[MISSING] {dataset_name}")
            continue

        try:
            df = pd.read_csv(file_path, low_memory=False)
            df.columns = df.columns.str.strip()

            data[dataset_name] = df

            manifest_rows.append({
                "dataset": dataset_name,
                "file_path": str(file_path.relative_to(PROJECT_ROOT)),
                "status": "LOADED",
                "rows": len(df),
                "columns": len(df.columns),
                "error": "",
            })

            print(
                f"[LOADED] {dataset_name}: "
                f"{len(df):,} rows, {len(df.columns)} columns"
            )

        except Exception as exc:
            manifest_rows.append({
                "dataset": dataset_name,
                "file_path": str(file_path),
                "status": "ERROR",
                "rows": 0,
                "columns": 0,
                "error": str(exc),
            })
            print(f"[ERROR] {dataset_name}: {exc}")

    manifest = pd.DataFrame(manifest_rows)
    return data, manifest


if __name__ == "__main__":
    datasets, manifest = load_csv_datasets()

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "data_loading_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("\n=== DATA LOADING SUMMARY ===")
    print(manifest.to_string(index=False))
    print(f"\nManifest saved to: {manifest_path}")
