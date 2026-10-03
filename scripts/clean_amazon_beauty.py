from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "amazon_beauty"
    / "train-00000-of-00001.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "amazon_beauty_cleaned.csv"
)

print("Loading Amazon Beauty dataset...")

df = pd.read_parquet(INPUT_FILE)


def list_to_text(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        return ", ".join(str(x) for x in value)

    if hasattr(value, "size"):
        if value.size == 0:
            return ""

        try:
            return ", ".join(str(x) for x in value.tolist())
        except Exception:
            return str(value)

    if isinstance(value, float) and pd.isna(value):
        return ""

    return str(value)


df = df.drop_duplicates(subset="asin")

cleaned = pd.DataFrame()

cleaned["product_id"] = df["asin"].astype(str)
cleaned["platform"] = "Amazon"
cleaned["category"] = "Beauty"
cleaned["product_name"] = df["title"].fillna("").astype(str)
cleaned["brand"] = df["brand"].fillna("").astype(str)

cleaned["price"] = pd.NA
cleaned["rating"] = pd.NA

cleaned["features"] = ""

cleaned["description"] = (
    df["description"]
    .apply(list_to_text)
)

cleaned["image_url"] = df.apply(
    lambda row: (
        row["imageURLHighRes"][0]
        if isinstance(row["imageURLHighRes"], list)
        and len(row["imageURLHighRes"]) > 0
        else (
            row["imageURL"][0]
            if isinstance(row["imageURL"], list)
            and len(row["imageURL"]) > 0
            else ""
        )
    ),
    axis=1
)

cleaned["product_url"] = (
    "https://www.amazon.com/dp/"
    + df["asin"].astype(str)
)

cleaned.to_csv(OUTPUT_FILE, index=False)

print("\nAmazon Beauty cleaning completed!")
print("Final rows:", len(cleaned))
print("Final columns:", len(cleaned.columns))

print("\nColumns:")
print(cleaned.columns.tolist())

print("\nSample:")
print(
    cleaned[
        ["product_id", "product_name", "brand", "price", "rating"]
    ].head().to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)