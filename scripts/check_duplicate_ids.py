from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_FOLDER = PROJECT_ROOT / "data" / "cleaned"

files = sorted(CLEANED_FOLDER.glob("*_cleaned.csv"))

all_data = []

for file in files:
    df = pd.read_csv(file, usecols=["product_id", "platform", "category"])
    all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)

duplicates = combined[
    combined["product_id"].astype(str).duplicated(keep=False)
].copy()

duplicates = duplicates.sort_values(
    "product_id",
    key=lambda x: x.astype(str)
)

print("Total duplicate rows:", len(duplicates))
print(
    "Unique duplicated product IDs:",
    duplicates["product_id"].astype(str).nunique()
)

print("\nDuplicate IDs by platform:")
print(
    duplicates.groupby("platform")["product_id"]
    .nunique()
    .sort_values(ascending=False)
    .to_string()
)

print("\nFlipkart duplicate IDs:")

flipkart_duplicates = duplicates[
    duplicates["platform"] == "Flipkart"
]

print(
    flipkart_duplicates.to_string(index=False)
)