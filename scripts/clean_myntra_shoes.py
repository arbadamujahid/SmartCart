from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "myntra"
    / "products catalog.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "myntra_shoes_cleaned.csv"
)

print("Loading Myntra dataset...")

df = pd.read_csv(INPUT_FILE)

text = (
    df["Description"]
    .fillna("")
    .astype(str)
    .str.lower()
)

shoe_patterns = [
    r"a pair of .*shoes",
    r"a pair of .*sneakers",
    r"a pair of .*sandals",
    r"a pair of .*heels",
    r"a pair of .*boots",
    r"a pair of .*loafers",
    r"a pair of .*flats",
    r"a pair of .*flip flops",
    r"a pair of .*juttis",
    r"a pair of .*mules",
    r"a pair of .*pumps",
    r"a pair of .*wedges",
    r"running sports shoes",
    r"sports shoes",
]

mask = text.apply(
    lambda value: any(
        re.search(pattern, value, flags=re.IGNORECASE)
        for pattern in shoe_patterns
    )
)

df = df[mask].copy()

df = df.drop_duplicates(subset="ProductID")


def get_product_name(description):
    description = str(description).strip()

    match = re.search(
        r"(?:a pair of|design details:)\s*(.*?)(?:, has|has regular|synthetic upper|mesh upper|leather upper|rubber upper)",
        description,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return description[:150].strip()


cleaned = pd.DataFrame()

cleaned["product_id"] = df["ProductID"].astype(str)
cleaned["platform"] = "Myntra"
cleaned["category"] = "Shoes"
cleaned["product_name"] = df["Description"].apply(get_product_name)
cleaned["brand"] = ""
cleaned["price"] = pd.to_numeric(df["Price (INR)"], errors="coerce")
cleaned["rating"] = pd.NA

cleaned["features"] = (
    "Gender: " + df["Gender"].fillna("").astype(str)
    + ", Color: " + df["PrimaryColor"].fillna("").astype(str)
    + ", Images: " + df["NumImages"].fillna("").astype(str)
)

cleaned["description"] = df["Description"].fillna("").astype(str)
cleaned["image_url"] = ""

cleaned["product_url"] = (
    "https://www.myntra.com/"
    + df["ProductID"].astype(str)
)

cleaned.to_csv(OUTPUT_FILE, index=False)

print("\nMyntra Shoes cleaning completed!")
print("Final rows:", len(cleaned))
print("Final columns:", len(cleaned.columns))

print("\nColumns:")
print(cleaned.columns.tolist())

print("\nSample:")
print(
    cleaned[
        ["product_id", "product_name", "price"]
    ].head().to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)