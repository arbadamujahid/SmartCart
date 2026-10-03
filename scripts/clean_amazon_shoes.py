from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = next(
    (PROJECT_ROOT / "data" / "raw" / "amazon_shoes").glob("*.xlsx")
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data" / "cleaned"
    / "amazon_shoes_cleaned.csv"
)

print("Loading Amazon shoes dataset...")

df = pd.read_excel(INPUT_FILE)

print(f"Original rows: {len(df)}")

# Remove duplicate products
df = df.drop_duplicates(subset="ASIN")
df = df.dropna(subset=["ASIN", "Title"])

print(f"After removing duplicates: {len(df)}")


def clean_price(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).replace(",", "").replace("$", "").strip()
    return pd.to_numeric(value, errors="coerce")


def get_first_image(value):
    if pd.isna(value):
        return ""

    images = str(value).split(";")

    return images[0].strip() if images else ""


# Clean basic fields
df["Title"] = (
    df["Title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["Brand"] = (
    df["Brand"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["ASIN"] = (
    df["ASIN"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Clean price
df["price"] = df["Buy Box: Current"].apply(clean_price)

# Get the first product image
df["image_url"] = df["Image"].apply(get_first_image)

# Create useful recommendation features
feature_columns = [
    "Color",
    "Size",
    "Material",
    "Style",
    "Pattern",
    "Target Audience",
    "Recommended Uses",
    "Specific Uses",
    "Product Benefit",
]

df["features"] = ""

for column in feature_columns:
    if column in df.columns:
        values = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df["features"] = (
            df["features"]
            + column.replace("_", " ")
            + ": "
            + values
            + ", "
        )

df["features"] = (
    df["features"]
    .str.rstrip(", ")
    .str.strip()
)

# Create common project columns
df["product_id"] = df["ASIN"]
df["platform"] = "Amazon"
df["category"] = "Shoes"
df["product_name"] = df["Title"]
df["brand"] = df["Brand"]
df["rating"] = pd.to_numeric(
    df["Reviews: Rating"],
    errors="coerce"
)

df["description"] = (
    df["Product Benefit"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["product_url"] = (
    "https://www.amazon.com/dp/"
    + df["ASIN"]
)

cleaned_df = df[
    [
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
].copy()

# Clean final text columns
text_columns = [
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

for column in text_columns:
    cleaned_df[column] = (
        cleaned_df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

cleaned_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\nAmazon shoes cleaning completed!")
print(f"Final rows: {len(cleaned_df)}")
print(f"Final columns: {len(cleaned_df.columns)}")

print("\nColumns:")
print(cleaned_df.columns.tolist())

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nSample:")
print(
    cleaned_df[
        ["product_id", "product_name", "brand", "price", "rating"]
    ].head(5).to_string(index=False)
)