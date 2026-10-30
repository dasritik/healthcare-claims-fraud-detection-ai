
from src.data.load_data import load_csv_datasets


def main():
    datasets, manifest = load_csv_datasets()

    print("\n" + "=" * 70)
    print("DATASET SCHEMA INSPECTION")
    print("=" * 70)

    for name, df in datasets.items():
        print(f"\nDataset: {name}")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")

        print("\nColumn names:")
        for col in df.columns:
            print(f"  - {col}")

        print("\nSample records:")
        print(df.head(3).to_string(index=False))

    print("\n" + "=" * 70)
    print("Inspection completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
