from pathlib import Path
import ast
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "flipkart"
    / "flipkart_com-ecommerce_sample.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaned"

print("Loading Flipkart dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original rows: {len(df)}")


def clean_price(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).replace(",", "").replace("₹", "").strip()
    return pd.to_numeric(value, errors="coerce")


def clean_rating(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    if "no rating" in value.lower():
        return pd.NA

    return pd.to_numeric(value, errors="coerce")


def get_first_image(value):
    if pd.isna(value):
        return ""

    try:
        images = ast.literal_eval(str(value))

        if isinstance(images, list) and images:
            return str(images[0]).strip()

    except Exception:
        pass

    return ""


def prepare_category(data, category_name, category_keywords):
    category_text = (
        data["product_category_tree"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    mask = category_text.apply(
        lambda value: any(
            keyword in value
            for keyword in category_keywords
        )
    )

    # Keep laptop products but exclude laptop accessories.
    if category_name == "Laptops":
        mask &= category_text.str.contains(
            "laptop",
            na=False
        )

        mask &= ~category_text.str.contains(
            "laptop accessories|notebook accessories",
            na=False
        )

    # Keep only products under the mobile/accessories category.
    if category_name == "Mobiles":
        mask &= category_text.str.contains(
            "mobiles & accessories",
            na=False
        )

    # Keep shoes only when the category tree identifies footwear.
    if category_name == "Shoes":
        mask &= category_text.str.contains(
            "footwear",
            na=False
        )

    # Prevent mobile and laptop products from entering Fashion.
    if category_name == "Fashion":
        mask &= ~category_text.str.contains(
            "mobiles & accessories|computers >> laptops|laptop",
            na=False
        )

    # Prevent mobile products from entering Beauty.
    if category_name == "Beauty":
        mask &= ~category_text.str.contains(
            "mobiles & accessories",
            na=False
        )

    result = data[mask].copy()

    result = result.drop_duplicates(subset="pid")
    result = result.dropna(subset=["pid", "product_name"])

    result["product_id"] = result["pid"]
    result["platform"] = "Flipkart"
    result["category"] = category_name

    result["product_name"] = (
        result["product_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["brand"] = (
        result["brand"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["price"] = result["discounted_price"].apply(clean_price)

    result["rating"] = result["product_rating"].apply(clean_rating)

    result["features"] = (
        result["product_category_tree"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["description"] = (
        result["description"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["image_url"] = result["image"].apply(get_first_image)

    cleaned = result[
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
        cleaned[column] = (
            cleaned[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    output_file = (
        OUTPUT_DIR
        / f"flipkart_{category_name.lower()}_cleaned.csv"
    )

    cleaned.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print(
        f"{category_name}: "
        f"{len(cleaned)} products → {output_file.name}"
    )


categories = {
    "Fashion": [
        "clothing",
        "apparel",
        "ethnic wear",
        "western wear",
        "lingerie",
        "innerwear",
        "saree",
        "jeans",
        "trousers",
        "shorts",
        "t-shirt",
        "skirts",
        "bras",
        "pyjamas",
        "tops",
    ],

    "Beauty": [
        "beauty and personal care",
        "makeup",
        "cosmetics",
        "skin care",
        "skincare",
        "hair care",
        "haircare",
        "fragrance",
        "perfume",
        "lipstick",
        "shampoo",
        "conditioner",
        "face care",
    ],

    "Laptops": [
        "laptop",
        "laptops",
        "notebook",
        "ultrabook",
        "chromebook",
    ],

    "Mobiles": [
        "mobiles & accessories",
        "smartphone",
        "smart phone",
        "cell phone",
        "mobile phone",
        "tablet accessories",
        "cases & covers",
    ],

    "Shoes": [
        "footwear",
        "shoes",
        "shoe",
        "sandal",
        "sandals",
        "sneaker",
        "sneakers",
        "boots",
        "bellies",
        "loafers",
        "flip flops",
        "sports shoes",
    ],
}


for category_name, category_keywords in categories.items():
    prepare_category(
        df,
        category_name,
        category_keywords
    )

print("\nFlipkart cleaning completed.")