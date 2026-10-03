from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "myntra"
    / "products catalog.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaned"

print("Loading Myntra dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original rows: {len(df)}")


def clean_price(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).replace(",", "").replace("₹", "").strip()

    return pd.to_numeric(value, errors="coerce")


def is_shoe_product(description):
    if pd.isna(description):
        return False

    text = str(description).lower()

    shoe_patterns = [
        r"\bshoe\b",
        r"\bshoes\b",
        r"\bsneaker\b",
        r"\bsneakers\b",
        r"\bsandal\b",
        r"\bsandals\b",
        r"\bslipper\b",
        r"\bslippers\b",
        r"\bflip[- ]flop\b",
        r"\bflip[- ]flops\b",
        r"\bheels?\b",
        r"\bloafers?\b",
        r"\bboots?\b",
        r"\bbellies\b",
        r"\bflats?\b",
        r"\bfootwear\b",
        r"\bmules?\b",
        r"\bderbys?\b",
        r"\bcasual shoes\b",
        r"\bsports shoes\b",
        r"\brunning shoes\b",
        r"\bformal shoes\b",
        r"\bparty shoes\b",
        r"\bopen toe\b",
        r"\bpeep toe\b",
        r"\bround toe\b",
        r"\bpointed toe\b",
        r"\btoe shoes\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in shoe_patterns
    )


# Clean basic fields
df["ProductID"] = (
    df["ProductID"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["Description"] = (
    df["Description"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["Gender"] = (
    df["Gender"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Remove rows without a product ID
df = df[df["ProductID"] != ""]

# Remove duplicate product IDs
df = df.drop_duplicates(subset="ProductID")


# Identify shoe products
shoe_mask = df["Description"].apply(is_shoe_product)


# -------------------------
# Myntra Fashion
# -------------------------

fashion = df[~shoe_mask].copy()

fashion["product_id"] = fashion["ProductID"]
fashion["platform"] = "Myntra"
fashion["category"] = "Fashion"

fashion["product_name"] = fashion["Description"]

fashion["brand"] = ""

fashion["price"] = fashion["Price (INR)"].apply(clean_price)

fashion["rating"] = pd.NA

fashion["features"] = (
    fashion["Gender"]
    + " | Primary Color: "
    + fashion["PrimaryColor"].fillna("").astype(str)
)


fashion["image_url"] = ""

fashion["product_url"] = ""


fashion_cleaned = fashion[
    [
        "product_id",
        "platform",
        "category",
        "product_name",
        "brand",
        "price",
        "rating",
        "features",
        "Description",
        "image_url",
        "product_url",
    ]
].copy()

fashion_cleaned = fashion_cleaned.rename(
    columns={
        "Description": "description"
    }
)


# -------------------------
# Myntra Shoes
# -------------------------

shoes = df[shoe_mask].copy()

shoes["product_id"] = shoes["ProductID"]
shoes["platform"] = "Myntra"
shoes["category"] = "Shoes"

shoes["product_name"] = shoes["Description"]

shoes["brand"] = ""

shoes["price"] = shoes["Price (INR)"].apply(clean_price)

shoes["rating"] = pd.NA

shoes["features"] = (
    shoes["Gender"]
    + " | Primary Color: "
    + shoes["PrimaryColor"].fillna("").astype(str)
)


shoes["image_url"] = ""

shoes["product_url"] = ""


shoes_cleaned = shoes[
    [
        "product_id",
        "platform",
        "category",
        "product_name",
        "brand",
        "price",
        "rating",
        "features",
        "Description",
        "image_url",
        "product_url",
    ]
].copy()

shoes_cleaned = shoes_cleaned.rename(
    columns={
        "Description": "description"
    }
)


# Clean text columns
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
    fashion_cleaned[column] = (
        fashion_cleaned[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    shoes_cleaned[column] = (
        shoes_cleaned[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# Save Fashion dataset
fashion_output = (
    OUTPUT_DIR / "myntra_fashion_cleaned.csv"
)

fashion_cleaned.to_csv(
    fashion_output,
    index=False,
    encoding="utf-8"
)


# Save Shoes dataset
shoes_output = (
    OUTPUT_DIR / "myntra_shoes_cleaned.csv"
)

shoes_cleaned.to_csv(
    shoes_output,
    index=False,
    encoding="utf-8"
)


print(
    f"Fashion: {len(fashion_cleaned)} products "
    f"→ {fashion_output.name}"
)

print(
    f"Shoes: {len(shoes_cleaned)} products "
    f"→ {shoes_output.name}"
)

print("\nMyntra cleaning completed.")