from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "master_products.csv"


# ============================================================
# EXPECTED COMMON SCHEMA
# ============================================================

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
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING MASTER DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Check cleaned directory
    # --------------------------------------------------------

    if not CLEANED_DIR.exists():
        print("\nERROR: Cleaned data directory does not exist:")
        print(CLEANED_DIR)
        return

    # --------------------------------------------------------
    # Find all cleaned CSV files
    # --------------------------------------------------------

    files = sorted(CLEANED_DIR.glob("*_cleaned.csv"))

    if not files:
        print("\nERROR: No cleaned CSV files found.")
        print(f"Expected files inside: {CLEANED_DIR}")
        return

    print(f"\nFound {len(files)} cleaned datasets.\n")

    # --------------------------------------------------------
    # Load every dataset
    # --------------------------------------------------------

    dataframes = []

    total_before = 0

    for file in files:

        print(f"Loading: {file.name}")

        try:
            df = pd.read_csv(
                file,
                low_memory=False
            )

        except Exception as e:
            print(f"ERROR loading {file.name}: {e}")
            continue

        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")

        # ----------------------------------------------------
        # Check schema
        # ----------------------------------------------------

        missing_columns = [
            column
            for column in EXPECTED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            print(
                f"  WARNING: Missing columns: "
                f"{missing_columns}"
            )
            continue

        # ----------------------------------------------------
        # Keep only common columns
        # ----------------------------------------------------

        df = df[EXPECTED_COLUMNS].copy()

        # ----------------------------------------------------
        # Clean text columns
        # ----------------------------------------------------

        for column in TEXT_COLUMNS:

            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        # ----------------------------------------------------
        # Convert numeric columns
        # ----------------------------------------------------

        for column in NUMERIC_COLUMNS:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # ----------------------------------------------------
        # Remove rows with empty product IDs
        # ----------------------------------------------------

        before_empty_id = len(df)

        df = df[
            df["product_id"].str.strip() != ""
        ].copy()

        removed_empty_id = (
            before_empty_id - len(df)
        )

        if removed_empty_id > 0:
            print(
                f"  Removed empty product IDs: "
                f"{removed_empty_id:,}"
            )

        # ----------------------------------------------------
        # Remove duplicate IDs inside this dataset
        # ----------------------------------------------------

        before_duplicates = len(df)

        df = df.drop_duplicates(
            subset=["product_id"],
            keep="first"
        ).copy()

        removed_duplicates = (
            before_duplicates - len(df)
        )

        if removed_duplicates > 0:
            print(
                f"  Removed duplicate IDs: "
                f"{removed_duplicates:,}"
            )

        print(f"  Final rows: {len(df):,}")

        dataframes.append(df)

        total_before += len(df)

    # --------------------------------------------------------
    # Make sure datasets were loaded
    # --------------------------------------------------------

    if not dataframes:
        print("\nERROR: No valid datasets could be loaded.")
        return

    # --------------------------------------------------------
    # Combine all datasets
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMBINING DATASETS")
    print("=" * 70)

    master_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    print(
        f"\nRows before global duplicate removal: "
        f"{len(master_df):,}"
    )

    # --------------------------------------------------------
    # Check duplicate product IDs across ALL platforms
    # --------------------------------------------------------

    duplicate_mask = master_df.duplicated(
        subset=["product_id"],
        keep=False
    )

    duplicate_count = duplicate_mask.sum()

    print(
        f"Duplicate product-ID rows found globally: "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:

        print(
            "\nRemoving duplicate product IDs "
            "across all datasets..."
        )

        master_df = master_df.drop_duplicates(
            subset=["product_id"],
            keep="first"
        ).copy()

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    master_df.reset_index(
        drop=True,
        inplace=True
    )

    # --------------------------------------------------------
    # Ensure final column order
    # --------------------------------------------------------

    master_df = master_df[
        EXPECTED_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Create processed directory
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save master dataset
    # --------------------------------------------------------

    master_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("MASTER DATASET CREATED")
    print("=" * 70)

    print(
        f"\nOutput file:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nTotal products: "
        f"{len(master_df):,}"
    )

    print(
        f"Total columns: "
        f"{len(master_df.columns)}"
    )

    # --------------------------------------------------------
    # Verify duplicate IDs
    # --------------------------------------------------------

    duplicate_ids = master_df[
        master_df["product_id"].duplicated(keep=False)
    ]

    print(
        f"\nDuplicate product IDs remaining: "
        f"{len(duplicate_ids):,}"
    )

    # --------------------------------------------------------
    # Platform summary
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("PRODUCTS BY PLATFORM")
    print("-" * 70)

    platform_counts = (
        master_df["platform"]
        .value_counts()
    )

    print(platform_counts.to_string())

    # --------------------------------------------------------
    # Category summary
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("PRODUCTS BY CATEGORY")
    print("-" * 70)

    category_counts = (
        master_df["category"]
        .value_counts()
    )

    print(category_counts.to_string())

    # --------------------------------------------------------
    # Platform + category summary
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("PRODUCTS BY PLATFORM AND CATEGORY")
    print("-" * 70)

    platform_category = pd.crosstab(
        master_df["platform"],
        master_df["category"]
    )

    print(platform_category.to_string())

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("MISSING VALUES")
    print("-" * 70)

    missing_values = master_df.isna().sum()

    missing_values = missing_values[
        missing_values > 0
    ]

    if len(missing_values) == 0:
        print("No missing values.")
    else:
        print(missing_values.to_string())

    # --------------------------------------------------------
    # Empty text fields
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("EMPTY TEXT FIELDS")
    print("-" * 70)

    for column in TEXT_COLUMNS:

        empty_count = (
            master_df[column]
            .eq("")
            .sum()
        )

        if empty_count > 0:

            percentage = (
                empty_count
                / len(master_df)
                * 100
            )

            print(
                f"{column:<15} "
                f"{empty_count:>10,} "
                f"({percentage:.2f}%)"
            )

    # --------------------------------------------------------
    # Numeric statistics
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("NUMERIC DATA SUMMARY")
    print("-" * 70)

    print("\nPrice:")

    print(
        master_df["price"]
        .describe()
        .to_string()
    )

    print("\nRating:")

    print(
        master_df["rating"]
        .describe()
        .to_string()
    )

    # --------------------------------------------------------
    # Final schema
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("FINAL SCHEMA")
    print("-" * 70)

    for index, column in enumerate(
        master_df.columns,
        start=1
    ):
        print(
            f"{index:>2}. "
            f"{column:<15} "
            f"{master_df[column].dtype}"
        )

    print("\n" + "=" * 70)
    print("MASTER DATASET BUILD COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()