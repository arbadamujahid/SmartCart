from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = next(
    (PROJECT_ROOT / "data" / "raw" / "amazon_mobiles").glob("*.csv")
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data" / "cleaned"
    / "amazon_mobiles_cleaned.csv"
)

print("Loading Amazon mobile dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original rows: {len(df)}")

# Remove exact duplicate rows
df = df.drop_duplicates().copy()

print(f"After removing duplicates: {len(df)}")


def extract_brand(name):
    if not name:
        return ""

    known_brands = [
        "Samsung",
        "Redmi",
        "Xiaomi",
        "POCO",
        "iQOO",
        "OnePlus",
        "Realme",
        "OPPO",
        "Vivo",
        "Motorola",
        "Nothing",
        "Google",
        "Apple",
        "Nokia",
        "ASUS",
        "Oppo",
        "Vivo",
        "Tecno",
        "Infinix",
        "Lava",
        "Honor",
        "Huawei",
        "Sony",
        "OnePlus",
    ]

    for brand in known_brands:
        if name.lower().startswith(brand.lower()):
            return brand

    return name.split()[0]


def clean_price(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).replace(",", "").replace("₹", "").strip()
    return pd.to_numeric(value, errors="coerce")


df["Product Name"] = (
    df["Product Name"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["Product Rating"] = pd.to_numeric(
    df["Product Rating"],
    errors="coerce"
)

df["Product Review"] = pd.to_numeric(
    df["Product Review"],
    errors="coerce"
)

df["Product Price"] = df["Product Price"].apply(clean_price)

df["RAM_GB"] = pd.to_numeric(
    df["RAM_GB"],
    errors="coerce"
)

df["Storage_GB"] = pd.to_numeric(
    df["Storage_GB"],
    errors="coerce"
)

df["Key_Features"] = (
    df["Key_Features"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Extract brand from the product name
df["brand"] = df["Product Name"].apply(extract_brand)

# Build recommendation features
df["features"] = (
    "RAM: " + df["RAM_GB"].fillna(0).astype(str) + " GB, "
    + "Storage: " + df["Storage_GB"].fillna(0).astype(str) + " GB, "
    + df["Key_Features"]
)

# Create common project columns
df["product_id"] = [
    f"AM{index:04d}"
    for index in range(1, len(df) + 1)
]

df["platform"] = "Amazon"
df["category"] = "Mobiles"
df["product_name"] = df["Product Name"]
df["price"] = df["Product Price"]
df["rating"] = df["Product Rating"]

# Amazon product URLs and images are not available in this dataset
df["image_url"] = ""
df["product_url"] = ""
df["description"] = ""

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

print("\nAmazon mobile cleaning completed!")
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