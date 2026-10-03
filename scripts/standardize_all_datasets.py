from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"


# ============================================================
# STANDARD SCHEMA
# ============================================================

STANDARD_COLUMNS = [
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


# ============================================================
# DATA TYPES
# ============================================================

TEXT_COLUMNS = [
    "product_id",
    "platform",
    "category",
    "product_name",
    "brand",
    "features",
    "description",
    "image_url",
    "product_url",
]

NUMERIC_COLUMNS = [
    "price",
    "rating",
]


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    return (
        str(value)
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


# ============================================================
# CLEAN NUMERIC
# ============================================================

def clean_numeric(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    if value == "":
        return pd.NA

    # Remove common currency symbols and commas
    value = (
        value
        .replace("₹", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace(",", "")
        .strip()
    )

    return pd.to_numeric(value, errors="coerce")


# ============================================================
# STANDARDIZE ONE DATASET
# ============================================================

def standardize_dataset(input_file):
    print("=" * 70)
    print(f"Processing: {input_file.name}")

    try:
        df = pd.read_csv(input_file)
    except Exception as error:
        print(f"ERROR reading file: {error}")
        return None

    original_rows = len(df)

    print(f"Original rows: {original_rows}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in STANDARD_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("Missing columns:")
        for column in missing_columns:
            print(f"  - {column}")

        print("Skipping this dataset.")
        return None

    # --------------------------------------------------------
    # Keep only standard columns
    # --------------------------------------------------------

    df = df[STANDARD_COLUMNS].copy()

    # --------------------------------------------------------
    # Clean text columns
    # --------------------------------------------------------

    for column in TEXT_COLUMNS:
        df[column] = df[column].apply(clean_text)

    # --------------------------------------------------------
    # Clean numeric columns
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:
        df[column] = df[column].apply(clean_numeric)

    # --------------------------------------------------------
    # Remove rows without product ID
    # --------------------------------------------------------

    before_product_id = len(df)

    df = df[
        df["product_id"].notna()
        & (df["product_id"].astype(str).str.strip() != "")
    ].copy()

    removed_no_id = before_product_id - len(df)

    # --------------------------------------------------------
    # Remove duplicate product IDs
    # --------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["product_id"],
        keep="first"
    ).copy()

    removed_duplicates = before_duplicates - len(df)

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    df = df.reset_index(drop=True)

    # --------------------------------------------------------
    # Save standardized dataset
    # --------------------------------------------------------

    df.to_csv(
        input_file,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(f"Final rows: {len(df)}")
    print(f"Removed missing product IDs: {removed_no_id}")
    print(f"Removed duplicate product IDs: {removed_duplicates}")

    print("\nMissing values:")

    missing = df.isna().sum()

    for column in STANDARD_COLUMNS:
        print(f"  {column}: {missing[column]}")

    print("\nSchema:")
    print(list(df.columns))

    print(f"\nSaved: {input_file.name}")

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    if not CLEANED_DIR.exists():
        print("ERROR: Cleaned directory does not exist.")
        print(CLEANED_DIR)
        return

    files = sorted(
        CLEANED_DIR.glob("*.csv")
    )

    if not files:
        print("No cleaned CSV files found.")
        return

    print("=" * 70)
    print("STANDARDIZING ALL CLEANED DATASETS")
    print("=" * 70)

    print(f"\nFound {len(files)} CSV files.\n")

    successful = 0
    failed = 0

    for file in files:

        result = standardize_dataset(file)

        if result is None:
            failed += 1
        else:
            successful += 1

        print()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("=" * 70)
    print("STANDARDIZATION COMPLETED")
    print("=" * 70)

    print(f"Datasets processed successfully: {successful}")
    print(f"Datasets skipped/failed: {failed}")

    print("\nAll datasets should now have this schema:")

    for column in STANDARD_COLUMNS:
        print(f"  - {column}")


if __name__ == "__main__":
    main()