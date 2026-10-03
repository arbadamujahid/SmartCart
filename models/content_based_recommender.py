
from pathlib import Path
import re
import html

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cleaned_datasets"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TOP_N = 5

CATEGORY_OPTIONS = {
    "1": ("Fashion", "fashion"),
    "2": ("Laptops", "laptops"),
    "3": ("Makeup", "beauty"),
    "4": ("Mobiles", "mobiles"),
    "5": ("Shoes", "shoes"),
}

PLATFORM_NAMES = {
    "amazon": "Amazon",
    "flipkart": "Flipkart",
    "myntra": "Myntra",
    "nykaa": "Nykaa",
}

# Product types are filtered using product names.
PRODUCT_TYPES = {
    "fashion": {
        "1": ("Shirt", ["shirt"]),
        "2": ("T-shirt", ["t-shirt", "t shirt", "tee"]),
        "3": ("Jeans", ["jeans", "denim jeans"]),
        "4": ("Dress", ["dress", "gown"]),
        "5": ("All Fashion Products", []),
    },
    "laptops": {
        "1": ("Gaming Laptop", ["gaming laptop", "gaming notebook"]),
        "2": ("Business Laptop", ["business laptop"]),
        "3": ("All Laptops", []),
    },
    "beauty": {
        "1": ("Lipstick", ["lipstick"]),
        "2": ("Foundation", ["foundation"]),
        "3": ("Mascara", ["mascara"]),
        "4": ("Eyeliner", ["eyeliner"]),
        "5": ("All Makeup Products", []),
    },
    "mobiles": {
        "1": ("Smartphone", ["smartphone", "mobile phone", "cell phone"]),
        "2": ("Samsung", ["samsung"]),
        "3": ("iPhone", ["iphone"]),
        "4": ("All Mobile Products", []),
    },
    "shoes": {
        "1": ("Sneakers", ["sneaker"]),
        "2": ("Sports Shoes", ["sports shoe", "running shoe"]),
        "3": ("Sandals", ["sandal"]),
        "4": ("Heels", ["heel"]),
        "5": ("All Shoes", []),
    },
}

COLUMN_ALIASES = {
    "product_name": ["product_name", "name", "title", "product"],
    "brand": ["brand", "manufacturer", "brand_name"],
    "price": ["price", "selling_price", "discounted_price"],
    "rating": ["rating", "ratings", "average_rating"],
    "description": [
        "description", "about_product", "product_description"
    ],
    "features": ["features", "specifications", "details"],
    "product_url": ["product_url", "url", "link"],
}


# --------------------------------------------------
# 2. STANDARDIZE DATASET COLUMNS
# --------------------------------------------------

def standardize_columns(df):
    df = df.copy()

    df.columns = [
        re.sub(
            r"[^a-z0-9]+",
            "_",
            str(column).strip().lower()
        ).strip("_")
        for column in df.columns
    ]

    for standard_name, aliases in COLUMN_ALIASES.items():
        if standard_name not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df[standard_name] = df[alias]
                    break

    return df


# --------------------------------------------------
# 3. LOAD THE EXISTING CLEANED DATASETS
# --------------------------------------------------

def load_products():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}"
        )

    files = sorted(DATA_DIR.glob("*_cleaned.csv"))

    if not files:
        raise FileNotFoundError(
            f"No cleaned CSV files found in: {DATA_DIR}"
        )

    frames = []

    for file in files:
        print(f"Loading {file.name}...")

        df = pd.read_csv(file, low_memory=False)
        df = standardize_columns(df)

        if "product_name" not in df.columns:
            print("Skipped: product name column missing.")
            continue

        # Determine category and platform from the filename.
        # This avoids trusting an incorrect row-level category.
        filename = file.stem.lower()

        category_key = next(
            (
                category
                for category in [
                    "beauty", "fashion", "laptops",
                    "mobiles", "shoes"
                ]
                if f"_{category}_" in f"_{filename}_"
            ),
            None
        )

        platform_key = next(
            (
                platform
                for platform in PLATFORM_NAMES
                if filename.startswith(platform + "_")
            ),
            None
        )

        if category_key is None or platform_key is None:
            print("Skipped: category/platform not identified.")
            continue

        df["category_key"] = category_key
        df["category"] = category_key
        df["platform"] = PLATFORM_NAMES[platform_key]
        df["source_file"] = file.name

        for column in [
            "brand", "description", "features",
            "product_url", "price", "rating"
        ]:
            if column not in df.columns:
                df[column] = ""

        for column in [
            "product_name", "brand",
            "description", "features", "product_url"
        ]:
            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .replace({"nan": "", "None": ""})
                .map(html.unescape)
                .str.strip()
            )

        for column in ["price", "rating"]:
            cleaned = (
                df[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
            )
            df[column] = pd.to_numeric(
                cleaned, errors="coerce"
            )

        df = df[df["product_name"].ne("")].copy()
        frames.append(df)

    if not frames:
        raise ValueError(
            "No usable cleaned datasets were found."
        )

    products = pd.concat(
        frames, ignore_index=True, sort=False
    )

    products.drop_duplicates(
        subset=["source_file", "product_name", "product_url"],
        keep="first",
        inplace=True
    )

    products.reset_index(drop=True, inplace=True)

    return products


# --------------------------------------------------
# 4. CATEGORY SELECTION
# --------------------------------------------------

def choose_category(products):
    print("\n========== SELECT CATEGORY ==========")

    for number, (name, key) in CATEGORY_OPTIONS.items():
        count = int(products["category_key"].eq(key).sum())
        print(f"{number}. {name} ({count:,} products)")

    while True:
        choice = input("\nEnter category number (1-5): ").strip()

        if choice not in CATEGORY_OPTIONS:
            print("Please enter a number from 1 to 5.")
            continue

        name, key = CATEGORY_OPTIONS[choice]

        selected = products[
            products["category_key"].eq(key)
        ].copy()

        if selected.empty:
            print(f"No products available in {name}.")
            continue

        print(f"\nSelected category: {name}")
        return selected, name, key


# --------------------------------------------------
# 5. PLATFORM SELECTION
# --------------------------------------------------

def choose_platform(products):
    print("\n========== SELECT PLATFORM ==========")

    available = sorted(
        products["platform"].dropna().unique().tolist()
    )

    if not available:
        raise ValueError(
            "No platforms found in this category."
        )

    for number, platform in enumerate(available, start=1):
        count = int(products["platform"].eq(platform).sum())
        print(f"{number}. {platform} ({count:,} products)")

    while True:
        choice = input(
            "\nEnter platform number: "
        ).strip()

        if not choice.isdigit():
            print("Enter a valid number.")
            continue

        index = int(choice) - 1

        if not 0 <= index < len(available):
            print("Choose a number from the list.")
            continue

        platform = available[index]

        selected = products[
            products["platform"].eq(platform)
        ].copy()

        print(f"\nSelected platform: {platform}")
        return selected, platform


# --------------------------------------------------
# 6. PRODUCT-TYPE SELECTION
# --------------------------------------------------

def choose_product_type(products, category_key):
    print("\n========== SELECT PRODUCT TYPE ==========")

    options = PRODUCT_TYPES[category_key]

    for number, (name, keywords) in options.items():
        if not keywords:
            count = len(products)
        else:
            pattern = "|".join(
                re.escape(word) for word in keywords
            )
            count = int(
                products["product_name"].str.contains(
                    pattern, case=False, na=False, regex=True
                ).sum()
            )

        print(f"{number}. {name} ({count:,} matching products)")

    while True:
        choice = input(
            "\nEnter product type number: "
        ).strip()

        if choice not in options:
            print("Choose a valid product type.")
            continue

        type_name, keywords = options[choice]

        if not keywords:
            selected = products.copy()
        else:
            pattern = "|".join(
                re.escape(word) for word in keywords
            )

            selected = products[
                products["product_name"].str.contains(
                    pattern, case=False, na=False, regex=True
                )
            ].copy()

        if selected.empty:
            print(
                f"No products found for '{type_name}' "
                "on this platform."
            )
            print("Choose another product type.")
            continue

        selected.reset_index(drop=True, inplace=True)

        print(f"\nSelected product type: {type_name}")
        print(f"Matching products: {len(selected):,}")

        return selected, type_name


# --------------------------------------------------
# 7. PREPARE TEXT FOR CONTENT-BASED RECOMMENDATIONS
# --------------------------------------------------

def prepare_product_text(products):
    products = products.copy()

    products["recommendation_text"] = (
        (products["product_name"] + " ") * 3
        + (products["brand"] + " ") * 2
        + products["features"] + " "
        + products["description"]
    )

    products["recommendation_text"] = (
        products["recommendation_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return products


# --------------------------------------------------
# 8. SELECT A PRODUCT
# --------------------------------------------------

def find_product(products):
    """Find a product by name, excluding mobile accessories from phone searches."""
    # Use a clean positional index so the returned index always matches .iloc[].
    products = products.copy().reset_index(drop=True)
    names = products["product_name"].fillna("").astype(str)

    is_mobile_category = (
        "category_key" in products.columns
        and products["category_key"].astype(str).str.lower().eq("mobiles").any()
    )

    if is_mobile_category:
        # Accessory terms are checked in the name AND metadata. This avoids
        # treating a phone model mentioned in an accessory title as a phone.
        accessory_pattern = (
            r"\b(pouch|case|cover|back panel|body panel|panel|charger|"
            r"charging cable|usb cable|data cable|otg|earphones?|headphones?|"
            r"tempered glass|screen guard|screen protector|screen protection|"
            r"sleeves?|holders?|socket|replacement parts?|battery|adapter|"
            r"mount|stand|skin|decal|flip cover|mobile accessories|"
            r"tablet accessories|stylus|memory card|power bank)\b"
        )
        metadata_columns = [
            column for column in ("features", "description")
            if column in products.columns
        ]
        metadata = (
            products[metadata_columns].fillna("").astype(str).agg(" ".join, axis=1)
            if metadata_columns else pd.Series("", index=products.index)
        )

        # Category paths in this dataset explicitly label these rows as
        # Mobile Accessories or Tablet Accessories. Exclude those rows even
        # if the accessory title contains words like Samsung or iPhone.
        accessory_category = metadata.str.contains(
            r"mobile accessories|tablet accessories|mobile pouches|"
            r"mobile chargers|mobile body panels|screen guards|"
            r"mobile accessories >>|tablet accessories >>",
            case=False, na=False, regex=True,
        )
        accessory_name = names.str.contains(
            accessory_pattern, case=False, na=False, regex=True
        )

        # A real phone title generally identifies a phone/device, but brand
        # names alone are not enough because accessories mention those brands.
        phone_pattern = (
            r"\b(smartphone|smart phone|mobile phone|cell phone|"
            r"mobile handset|iphone|galaxy [aszmf]?[0-9]|redmi (?!.*(?:case|cover|"
            r"pouch|glass|protector|charger))|poco [a-z0-9]+|oneplus [a-z0-9]+|"
            r"realme [a-z0-9]+|oppo [a-z0-9]+|vivo [a-z0-9]+|pixel [0-9]+|"
            r"motorola [a-z0-9]+|moto [a-z0-9]+|nokia [a-z0-9]+|"
            r"iqoo [a-z0-9]+|infinix [a-z0-9]+|tecno [a-z0-9]+)\b"
        )
        looks_like_phone = names.str.contains(
            phone_pattern, case=False, na=False, regex=True
        )
        valid_products = ~(accessory_category | accessory_name) & looks_like_phone

        if not valid_products.any():
            print(
                "\nNo smartphone listings are available in this selection. "
                "The selected dataset appears to contain mobile/tablet accessories, "
                "so it cannot provide phone recommendations. Please add a dataset "
                "containing actual smartphones."
            )
            return None
    else:
        valid_products = pd.Series(True, index=products.index)

    eligible = products.loc[valid_products].copy()
    eligible_names = eligible["product_name"].fillna("").astype(str)

    while True:
        keyword = input("\nSearch for a product (or type 'back'): ").strip()

        if keyword.lower() == "back":
            return None
        if not keyword:
            print("Enter a product name or keyword.")
            continue

        # First match the complete phrase; if none, require every search word.
        matched_mask = eligible_names.str.contains(
            keyword, case=False, na=False, regex=False
        )
        matches = eligible.loc[matched_mask].head(10)
        if matches.empty:
            words = [word.casefold() for word in keyword.split()]
            if words:
                all_words_mask = eligible_names.apply(
                    lambda name: all(word in name.casefold() for word in words)
                )
                matches = eligible.loc[all_words_mask].head(10)

        if matches.empty:
            print("No matching products found. Try another keyword.")
            continue

        print("\n========== MATCHING PRODUCTS ==========")
        for number, (_, row) in enumerate(matches.iterrows(), start=1):
            brand = row.get("brand", "N/A")
            if pd.isna(brand) or not str(brand).strip():
                brand = "N/A"
            platform = row.get("platform", "N/A")
            print(
                f"{number}. {row['product_name']} "
                f"| Brand: {brand} | Platform: {platform}"
            )

        choice = input("\nChoose product number (0 to search again): ").strip()
        if choice == "0":
            continue
        if not choice.isdigit():
            print("Enter a valid number.")
            continue

        choice_number = int(choice)
        if not 1 <= choice_number <= len(matches):
            print("Choose a number shown in the list.")
            continue

        # products has a RangeIndex, so this position is safe for .iloc[].
        return int(matches.index[choice_number - 1])


# --------------------------------------------------
# 9. GENERATE TOP-N RECOMMENDATIONS
# --------------------------------------------------

def recommend_products(products, product_index, top_n=TOP_N):
    products = products.reset_index(drop=True)

    if not 0 <= product_index < len(products):
        raise IndexError("Selected product index is invalid.")

    selected = products.iloc[product_index]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    try:
        matrix = vectorizer.fit_transform(
            products["recommendation_text"]
        )
    except ValueError:
        print("Not enough text to build recommendations.")
        return products.iloc[0:0].copy()

    scores = cosine_similarity(
        matrix[product_index], matrix
    ).ravel()

    scores[product_index] = -1

    ranked_indices = scores.argsort()[::-1]
    chosen_indices = []
    chosen_names = []

    def name_words(name):
        return set(
            re.findall(r"[a-z0-9]+", str(name).lower())
        )

    def overlap(name_a, name_b):
        words_a = name_words(name_a)
        words_b = name_words(name_b)
        union = words_a | words_b

        if not union:
            return 0

        return len(words_a & words_b) / len(union)

    for index in ranked_indices:
        if scores[index] <= 0:
            continue

        candidate_name = products.iloc[index]["product_name"]

        # Skip the selected product and near-identical variants.
        if overlap(
            selected["product_name"], candidate_name
        ) >= 0.85:
            continue

        if any(
            overlap(candidate_name, previous) >= 0.85
            for previous in chosen_names
        ):
            continue

        chosen_indices.append(index)
        chosen_names.append(candidate_name)

        if len(chosen_indices) == top_n:
            break

    recommendations = products.iloc[chosen_indices].copy()

    if not recommendations.empty:
        recommendations["similarity_score"] = [
            round(float(scores[index]), 3)
            for index in chosen_indices
        ]

    return recommendations


# --------------------------------------------------
# 10. DISPLAY AND SAVE RECOMMENDATIONS
# --------------------------------------------------

def display_recommendations(recommendations):
    if recommendations.empty:
        print(
            "\nNo sufficiently similar recommendations found."
        )
        return

    columns = [
        "product_name", "brand", "price",
        "rating", "platform", "similarity_score"
    ]

    print("\n========== TOP RECOMMENDATIONS ==========")
    print(
        recommendations[columns].to_string(index=False)
    )


def save_recommendations(recommendations):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "sample_recommendations.csv"

    recommendations.to_csv(output_file, index=False)

    print(f"\nResults saved to:\n{output_file}")


# --------------------------------------------------
# 11. MAIN PROGRAM
# --------------------------------------------------

def main():
    print("=" * 55)
    print("      PRODUCT RECOMMENDATION SYSTEM")
    print("       CONTENT-BASED FILTERING")
    print("=" * 55)

    try:
        products = load_products()
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\nCould not load datasets: {error}")
        return

    print(f"\nTotal products loaded: {len(products):,}")

    while True:
        # Step 1: Category
        category_products, category_name, category_key = (
            choose_category(products)
        )

        # Step 2: Platform
        platform_products, platform_name = choose_platform(
            category_products
        )

        # Step 3: Product type
        type_products, type_name = choose_product_type(
            platform_products, category_key
        )

        # Step 4: Prepare the model only for this selection
        type_products = prepare_product_text(type_products)
        type_products.reset_index(drop=True, inplace=True)

        # Step 5: Select the product to get recommendations for
        product_index = find_product(type_products)

        if product_index is None:
            print("\nReturning to category selection...")
            continue

        selected = type_products.iloc[product_index]

        print("\n========== YOUR SELECTION ==========")
        print(f"Category: {category_name}")
        print(f"Platform: {platform_name}")
        print(f"Product type: {type_name}")
        print(f"Product: {selected['product_name']}")

        # Step 6: Recommend similar products
        recommendations = recommend_products(
            type_products, product_index
        )

        display_recommendations(recommendations)

        if not recommendations.empty:
            save_recommendations(recommendations)

        again = input(
            "\nSearch another product? (y/n): "
        ).strip().lower()

        if again != "y":
            print("\nThank you for using the system!")
            break


if __name__ == "__main__":
    main()