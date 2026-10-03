from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_FOLDER = PROJECT_ROOT / "data" / "cleaned"

EXPECTED_COLUMNS = [
    "product_id",
    "platform",
    "category",
    "product_name",
    "brand",
    "price",
    "rating",
    "features",
    "description",
    "image_url",
    "product_url",
]

files = sorted(CLEANED_FOLDER.glob("*_cleaned.csv"))

print(f"Found {len(files)} cleaned datasets.\n")

all_ids = []
all_rows = []

for file in files:
    print("=" * 70)
    print(file.name)

    df = pd.read_csv(file)

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    extra_columns = [
        column for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    duplicate_ids = df["product_id"].duplicated().sum()

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Duplicate product IDs:", duplicate_ids)

    if missing_columns:
        print("Missing columns:", missing_columns)
    else:
        print("Schema: OK")

    if extra_columns:
        print("Extra columns:", extra_columns)

    print("\nMissing values:")
    print(df[EXPECTED_COLUMNS].isna().sum().to_string())

    all_ids.extend(df["product_id"].astype(str).tolist())
    all_rows.append(df)

print("\n" + "=" * 70)
print("OVERALL CHECK")
print("=" * 70)

combined = pd.concat(all_rows, ignore_index=True)

print("Total rows:", len(combined))
print("Unique product IDs:", combined["product_id"].astype(str).nunique())
print(
    "Duplicate IDs across all datasets:",
    combined["product_id"].astype(str).duplicated().sum()
)

print("\nRows by platform:")
print(combined["platform"].value_counts().to_string())

print("\nRows by category:")
print(combined["category"].value_counts().to_string())

print("\nRows by platform and category:")
print(
    pd.crosstab(
        combined["platform"],
        combined["category"]
    ).to_string()
)

print("\nQuality check completed.")