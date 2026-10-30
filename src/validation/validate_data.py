
import sys
from pathlib import Path

import pandas as pd

# Add project root to Python import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_csv_datasets


OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Common identifier names, matched case-insensitively.
ID_CANDIDATES = {
    "claim": [
        "claim_id", "claimid", "claim_number", "claim_no"
    ],
    "member": [
        "member_id", "memberid", "patient_id", "beneficiary_id"
    ],
    "provider": [
        "provider_id", "providerid", "npi"
    ],
    "procedure": [
        "procedure_id", "procedureid", "procedure_code",
        "cpt_code", "hcpcs_code"
    ],
}


def find_column(df, candidates):
    """Find a column using case-insensitive matching."""
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def add_check(rows, dataset, check, status, details):
    rows.append({
        "dataset": dataset,
        "check": check,
        "status": status,
        "details": details,
    })


def validate_datasets(datasets):
    results = []

    # 1. Basic file-level validation
    for name, df in datasets.items():
        add_check(
            results, name, "row_count",
            "PASS" if len(df) > 0 else "WARNING",
            f"{len(df):,} rows"
        )

        add_check(
            results, name, "column_count",
            "PASS" if len(df.columns) > 0 else "FAIL",
            f"{len(df.columns):,} columns"
        )

        empty_columns = [
            col for col in df.columns
            if df[col].isna().all()
        ]
        add_check(
            results, name, "fully_empty_columns",
            "WARNING" if empty_columns else "PASS",
            ", ".join(map(str, empty_columns))
            if empty_columns else "No fully empty columns"
        )

        missing_cells = int(df.isna().sum().sum())
        add_check(
            results, name, "missing_cells",
            "WARNING" if missing_cells else "PASS",
            f"{missing_cells:,} missing cells across all columns"
        )

        duplicate_rows = int(df.duplicated().sum())
        add_check(
            results, name, "duplicate_rows",
            "WARNING" if duplicate_rows else "PASS",
            f"{duplicate_rows:,} fully duplicated rows"
        )

        # 2. Check likely identifier columns for duplicates.
        for entity, candidates in ID_CANDIDATES.items():
            col = find_column(df, candidates)
            if col is None:
                continue

            non_null = df[col].dropna()
            duplicate_ids = int(non_null.duplicated().sum())

            # Claims, members, providers and procedures are normally
            # entity tables, but line-item and history tables can repeat IDs.
            is_entity_table = (
                (entity == "claim" and name == "claims")
                or (entity == "member" and name == "members")
                or (entity == "provider" and name == "providers")
                or (entity == "procedure" and name == "procedures")
            )

            if is_entity_table:
                status = "FAIL" if duplicate_ids else "PASS"
                details = (
                    f"Column '{col}': {duplicate_ids:,} duplicate IDs "
                    f"among {len(non_null):,} non-null values"
                )
            else:
                status = "INFO"
                details = (
                    f"Column '{col}': {duplicate_ids:,} repeated values; "
                    "may be valid for detail/history tables"
                )

            add_check(
                results, name, f"{entity}_identifier_duplicates",
                status, details
            )

    # 3. Referential integrity checks where matching columns exist.
    claims = datasets.get("claims")
    members = datasets.get("members")
    providers = datasets.get("providers")
    line_items = datasets.get("claim_line_items")

    if claims is not None:
        claim_id_col = find_column(
            claims, ID_CANDIDATES["claim"]
        )

        if claim_id_col:
            claim_ids = set(claims[claim_id_col].dropna())

            if line_items is not None:
                line_claim_col = find_column(
                    line_items, ID_CANDIDATES["claim"]
                )

                if line_claim_col:
                    line_ids = line_items[line_claim_col].dropna()
                    orphan_count = int(
                        (~line_ids.isin(claim_ids)).sum()
                    )
                    add_check(
                        results,
                        "claim_line_items",
                        "claim_reference_integrity",
                        "FAIL" if orphan_count else "PASS",
                        f"{orphan_count:,} line-item rows reference "
                        "claim IDs not found in claims"
                    )

        member_claim_col = find_column(
            claims, ID_CANDIDATES["member"]
        )
        if members is not None and member_claim_col:
            member_id_col = find_column(
                members, ID_CANDIDATES["member"]
            )

            if member_id_col:
                member_ids = set(members[member_id_col].dropna())
                references = claims[member_claim_col].dropna()
                orphan_count = int(
                    (~references.isin(member_ids)).sum()
                )
                add_check(
                    results,
                    "claims",
                    "member_reference_integrity",
                    "FAIL" if orphan_count else "PASS",
                    f"{orphan_count:,} claims reference member IDs "
                    "not found in members"
                )

        provider_claim_col = find_column(
            claims, ID_CANDIDATES["provider"]
        )
        if providers is not None and provider_claim_col:
            provider_id_col = find_column(
                providers, ID_CANDIDATES["provider"]
            )

            if provider_id_col:
                provider_ids = set(
                    providers[provider_id_col].dropna()
                )
                references = claims[provider_claim_col].dropna()
                orphan_count = int(
                    (~references.isin(provider_ids)).sum()
                )
                add_check(
                    results,
                    "claims",
                    "provider_reference_integrity",
                    "FAIL" if orphan_count else "PASS",
                    f"{orphan_count:,} claims reference provider IDs "
                    "not found in providers"
                )

    return pd.DataFrame(results)


def main():
    datasets, manifest = load_csv_datasets()

    manifest.to_csv(
        OUTPUT_DIR / "data_loading_manifest.csv",
        index=False
    )

    report = validate_datasets(datasets)
    report_path = OUTPUT_DIR / "data_validation_report.csv"
    report.to_csv(report_path, index=False)

    print("\n=== DATA LOADING SUMMARY ===")
    if not manifest.empty:
        print(manifest.to_string(index=False))

    print("\n=== VALIDATION SUMMARY ===")
    if not report.empty:
        print(
            report.groupby(["status"]).size()
            .rename("checks")
            .to_string()
        )

        problems = report[
            report["status"].isin(["FAIL", "WARNING"])
        ]

        if not problems.empty:
            print("\nChecks requiring review:")
            print(problems.to_string(index=False))

    print(f"\nValidation report: {report_path}")

    if (manifest["status"] == "ERROR").any():
        raise SystemExit(
            "One or more CSV files could not be loaded. "
            "Review the loading manifest."
        )


if __name__ == "__main__":
    main()
