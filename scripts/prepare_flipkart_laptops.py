
from pathlib import Path
import pandas as pd
import re

# Find the main project folder automatically
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Input: original raw laptop data
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "flipkart" / "Flipkart_data.txt"

# Output: a NEW cleaned dataset (existing files stay untouched)
OUTPUT_DIR = PROJECT_DIR / "data" / "cleaned_datasets"
OUTPUT_FILE = OUTPUT_DIR / "flipkart_laptops_cleaned_new.csv"


def get_brand(product_name):
    """Extract a likely brand from the product name."""
    name = str(product_name).strip()
    brands = [
        "Acer", "Apple", "ASUS", "Dell", "HP", "Lenovo",
        "Samsung", "MSI", "Microsoft", "Infinix", "Realme",
        "Xiaomi", "Avita", "Fujitsu", "LG"
    ]

    for brand in brands:
        if re.search(r"\b" + re.escape(brand) + r"\b", name, re.I):
            return brand

    return "Unknown"


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found:\n{INPUT_FILE}")
        print("Check that Flipkart_data.txt is inside data/raw/flipkart/")
        return

    # Read the raw file; try common text encodings
    try:
        df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding="latin-1")

    # Normalize column names
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")
        for col in df.columns
    ]

    required = ["product_name", "ram", "rom", "display_in_cm", "price"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        print("ERROR: Some expected columns were not found.")
        print("Columns detected:", df.columns.tolist())
        print("Missing:", missing)
        print("No output dataset was created.")
        return

    # Clean product names and remove empty rows
    df["product_name"] = df["product_name"].astype("string").str.strip()
    df = df.dropna(subset=["product_name"])
    df = df[df["product_name"] != ""]

    # Convert price values such as '17,990' into numbers
    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace(",", "", regex=False)
                         .str.replace("₹", "", regex=False)
                         .str.replace("Rs.", "", regex=False)
                         .str.strip(),
        errors="coerce"
    )

    # Build features from the available laptop specifications
    df["brand"] = df["product_name"].apply(get_brand)
    df["platform"] = "Unknown - verify source"
    df["category"] = "Laptops"
    df["rating"] = pd.NA
    df["features"] = (
        "RAM: " + df["ram"].fillna("").astype(str)
        + "; Storage: " + df["rom"].fillna("").astype(str)
        + "; Display: " + df["display_in_cm"].fillna("").astype(str) + " cm"
    )
    df["description"] = df["product_name"] + " | " + df["features"]
    df["product_id"] = [
        f"laptop_{i:04d}" for i in range(1, len(df) + 1)
    ]
    df["image_url"] = ""
    df["product_url"] = ""

    # Match the existing recommender's expected column names
    cleaned = df[
        [
            "product_id", "platform", "category", "product_name",
            "brand", "price", "rating", "features", "description",
            "image_url", "product_url"
        ]
    ].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Never overwrite the existing cleaned dataset
    cleaned.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nDataset preparation completed!")
    print(f"Input rows read: {len(df)}")
    print(f"Output rows saved: {len(cleaned)}")
    print(f"Output file:\n{OUTPUT_FILE}")
    print("\nFirst 5 products:")
    print(cleaned[["product_name", "brand", "price"]].head().to_string(index=False))
    print("\nNote: Platform/source is unverified; no ratings or product links were invented.")


if __name__ == "__main__":
    main()