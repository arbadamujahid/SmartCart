from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nykaa"
    / "Nykaa Cosmetics Data.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "nykaa_beauty_cleaned.csv"
)

print("Loading Nykaa dataset...")

df = pd.read_csv(INPUT_FILE)

df = df.drop_duplicates(subset="Product_ID")


cleaned = pd.DataFrame()

cleaned["product_id"] = df["Product_ID"].astype(str)
cleaned["platform"] = "Nykaa"
cleaned["category"] = "Beauty"
cleaned["product_name"] = df["product_title"].fillna("").astype(str)
cleaned["brand"] = df["Brand_Name"].fillna("").astype(str)

cleaned["price"] = pd.to_numeric(
    df["Price"],
    errors="coerce"
)

cleaned["rating"] = pd.to_numeric(
    df["rating"],
    errors="coerce"
)

cleaned["features"] = (
    "MRP: ₹"
    + df["MRP"].astype(str)
    + ", In Stock: "
    + df["in_stock"].astype(str)
)

cleaned["description"] = (
    df["tags"]
    .fillna("")
    .astype(str)
)

cleaned["image_url"] = ""

cleaned["product_url"] = (
    "https://www.nykaa.com/"
    + df["Product_ID"].astype(str)
)

cleaned.to_csv(OUTPUT_FILE, index=False)

print("\nNykaa Beauty cleaning completed!")
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