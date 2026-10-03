import pandas as pd
import re
import ast
import os

# ============================================================
# FLIPKART NEW MOBILE DATA CLEANING + MERGING
# ============================================================

NEW_FILE = "data/raw/flipkart_new/flipkart_mobile_data.csv"
OLD_FILE = "data/cleaned/flipkart_mobiles_cleaned.csv"
OUTPUT_FILE = "data/cleaned/flipkart_mobiles_cleaned.csv"


# ------------------------------------------------------------
# 1. Load new dataset
# ------------------------------------------------------------

print("=" * 70)
print("CLEANING NEW FLIPKART MOBILE DATA")
print("=" * 70)

print("\nLoading new Flipkart mobile dataset...")

new_df = pd.read_csv(NEW_FILE)

print(f"Original new rows: {len(new_df)}")
print(f"Original columns: {new_df.columns.tolist()}")


# ------------------------------------------------------------
# 2. Rename columns to project standard
# ------------------------------------------------------------

new_df = new_df.rename(columns={
    "Title": "product_name",
    "Product Link": "product_url",
    "Image URL": "image_url",
    "Rating": "rating",
    "Price": "price",
    "Original Price": "original_price",
    "Discount": "discount",
    "Key Features": "features"
})


# ------------------------------------------------------------
# 3. Create standard columns
# ------------------------------------------------------------

new_df["platform"] = "Flipkart"
new_df["category"] = "Mobiles"


# ------------------------------------------------------------
# 4. Create product IDs
# ------------------------------------------------------------

def extract_product_id(url):
    if pd.isna(url):
        return None

    match = re.search(r"pid=([A-Z0-9]+)", str(url))

    if match:
        return match.group(1)

    # fallback: use URL
    return str(url).strip()


new_df["product_id"] = new_df["product_url"].apply(extract_product_id)


# ------------------------------------------------------------
# 5. Extract brand from product name
# ------------------------------------------------------------

def extract_brand(name):
    if pd.isna(name):
        return "Unknown"

    name = str(name).strip()

    # Common multi-word brands
    multi_word_brands = [
        "Nothing Phone",
        "OnePlus",
        "POCO",
        "Realme",
        "Redmi",
        "Motorola",
        "Samsung",
        "Google",
        "iQOO",
        "Infinix",
        "Tecno",
        "Lava",
        "Nokia",
        "Oppo",
        "Vivo",
        "Xiaomi",
        "Apple",
        "Honor",
        "Asus",
        "Huawei",
        "Itel",
        "Micromax",
        "Nothing",
        "CMF"
    ]

    for brand in multi_word_brands:
        if name.lower().startswith(brand.lower()):
            return brand

    # fallback: first word
    return name.split()[0]


new_df["brand"] = new_df["product_name"].apply(extract_brand)


# ------------------------------------------------------------
# 6. Clean price
# ------------------------------------------------------------

def clean_price(value):
    if pd.isna(value):
        return None

    value = str(value)

    value = value.replace("₹", "")
    value = value.replace(",", "")
    value = value.strip()

    match = re.search(r"\d+(?:\.\d+)?", value)

    if match:
        return float(match.group())

    return None


new_df["price"] = new_df["price"].apply(clean_price)


# ------------------------------------------------------------
# 7. Clean rating
# ------------------------------------------------------------

def clean_rating(value):
    if pd.isna(value):
        return None

    match = re.search(r"\d+(?:\.\d+)?", str(value))

    if match:
        rating = float(match.group())

        if 0 <= rating <= 5:
            return rating

    return None


new_df["rating"] = new_df["rating"].apply(clean_rating)


# ------------------------------------------------------------
# 8. Clean features
# ------------------------------------------------------------

def clean_features(value):
    if pd.isna(value):
        return ""

    # Convert list-like strings into readable text
    if isinstance(value, str):
        value = value.strip()

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                return " | ".join(str(x).strip() for x in parsed)

        except:
            pass

        return value

    return str(value)


new_df["features"] = new_df["features"].apply(clean_features)


# ------------------------------------------------------------
# 9. Create description
# ------------------------------------------------------------

new_df["description"] = (
    new_df["features"].fillna("").astype(str)
)

# Add original price and discount where available
new_df["description"] = (
    new_df["description"]
    + " | Original Price: "
    + new_df["original_price"].fillna("").astype(str)
    + " | Discount: "
    + new_df["discount"].fillna("").astype(str)
)


# ------------------------------------------------------------
# 10. Clean text columns
# ------------------------------------------------------------

text_columns = [
    "product_id",
    "platform",
    "category",
    "product_name",
    "brand",
    "features",
    "description",
    "image_url",
    "product_url"
]

for column in text_columns:
    new_df[column] = (
        new_df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# ------------------------------------------------------------
# 11. Remove missing product IDs
# ------------------------------------------------------------

before = len(new_df)

new_df = new_df[
    new_df["product_id"].notna()
    & (new_df["product_id"] != "")
]

print(f"\nRemoved missing product IDs: {before - len(new_df)}")


# ------------------------------------------------------------
# 12. Remove duplicate product IDs in new data
# ------------------------------------------------------------

before = len(new_df)

new_df = new_df.drop_duplicates(
    subset=["product_id"],
    keep="first"
)

print(f"Removed duplicate new product IDs: {before - len(new_df)}")


# ------------------------------------------------------------
# 13. Select exact project columns
# ------------------------------------------------------------

standard_columns = [
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
    "product_url"
]

new_df = new_df[standard_columns]


# ------------------------------------------------------------
# 14. Load old cleaned Flipkart mobile data
# ------------------------------------------------------------

print("\nLoading existing Flipkart mobile dataset...")

old_df = pd.read_csv(OLD_FILE)

print(f"Existing rows: {len(old_df)}")


# ------------------------------------------------------------
# 15. Make sure old data has same columns
# ------------------------------------------------------------

old_df = old_df[standard_columns]


# ------------------------------------------------------------
# 16. Combine old + new
# ------------------------------------------------------------

print("\nCombining datasets...")

combined_df = pd.concat(
    [old_df, new_df],
    ignore_index=True
)

print(f"Rows before final duplicate removal: {len(combined_df)}")


# ------------------------------------------------------------
# 17. Remove duplicates across old + new
# ------------------------------------------------------------

before = len(combined_df)

combined_df = combined_df.drop_duplicates(
    subset=["product_id"],
    keep="first"
)

print(f"Duplicates removed: {before - len(combined_df)}")


# ------------------------------------------------------------
# 18. Final sorting
# ------------------------------------------------------------

combined_df = combined_df.reset_index(drop=True)


# ------------------------------------------------------------
# 19. Save final dataset
# ------------------------------------------------------------

combined_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 20. Final report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FLIPKART MOBILE CLEANING COMPLETED")
print("=" * 70)

print(f"Old dataset rows: {len(old_df)}")
print(f"New dataset rows: {len(new_df)}")
print(f"Final combined rows: {len(combined_df)}")

print("\nFinal columns:")
print(combined_df.columns.tolist())

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\nPlatform:")
print(combined_df["platform"].value_counts())

print("\nCategory:")
print(combined_df["category"].value_counts())

print("\nSample:")
print(combined_df.head(5).to_string())