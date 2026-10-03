from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "amazon_laptops"
    / "full-00000-of-00001.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "amazon_laptops_cleaned.csv"
)

print("Loading Amazon Laptops dataset...")

df = pd.read_parquet(INPUT_FILE)

df = df.drop_duplicates(subset="parent_asin")


def list_to_text(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        return ", ".join(str(x) for x in value)

    if isinstance(value, dict):
        return ", ".join(
            f"{key}: {value}"
            for key, value in value.items()
        )

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


cleaned = pd.DataFrame()

cleaned["product_id"] = df["parent_asin"].astype(str)
cleaned["platform"] = "Amazon"
cleaned["category"] = "Laptops"
cleaned["product_name"] = df["title"].fillna("").astype(str)
cleaned["brand"] = df["store"].fillna("").astype(str)

cleaned["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

cleaned["rating"] = pd.to_numeric(
    df["average_rating"],
    errors="coerce"
)

cleaned["features"] = df["features"].apply(list_to_text)

cleaned["description"] = df.apply(
    lambda row: (
        list_to_text(row["description"])
        + (
            " Specs: " + list_to_text(row["details"])
            if list_to_text(row["details"])
            else ""
        )
    ),
    axis=1
)

def get_image(value):
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                if item.get("hi_res"):
                    return item["hi_res"]
                if item.get("large"):
                    return item["large"]

    return ""


cleaned["image_url"] = df["images"].apply(get_image)

cleaned["product_url"] = (
    "https://www.amazon.com/dp/"
    + df["parent_asin"].astype(str)
)

cleaned.to_csv(OUTPUT_FILE, index=False)

print("\nAmazon Laptops cleaning completed!")
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