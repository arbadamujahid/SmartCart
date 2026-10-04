from pathlib import Path
import re
import html
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PRODUCT RECOMMENDATION SYSTEM - CONTENT-BASED FILTERING
# Works whether this file is in the project root or in scripts/
# ============================================================

def find_project_root():
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent]
    for candidate in candidates:
        if (candidate / "data" / "cleaned_datasets").exists():
            return candidate
    # Default to the parent when this script is inside scripts/
    return here.parent if here.name.lower() == "scripts" else here


PROJECT_ROOT = find_project_root()
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

PRODUCT_TYPES = {
    "fashion": {
        "1": ("Men's Fashion", ["men", "mens", "male"]),
        "2": ("Women's Fashion", ["women", "womens", "female", "ladies"]),
        "3": ("T-shirts", ["t-shirt", "tshirt", "tee"]),
        "4": ("Shirts", ["shirt"]),
        "5": ("Jeans", ["jeans", "denim"]),
        "6": ("All Fashion Products", []),
    },
    "laptops": {
        "1": ("Gaming Laptops", ["gaming"]),
        "2": ("HP Laptops", ["hp"]),
        "3": ("Dell Laptops", ["dell"]),
        "4": ("Lenovo Laptops", ["lenovo"]),
        "5": ("All Laptops", []),
    },
    "beauty": {
        "1": ("Lipstick", ["lipstick"]),
        "2": ("Foundation", ["foundation"]),
        "3": ("Mascara", ["mascara"]),
        "4": ("Eyeliner", ["eyeliner"]),
        "5": ("All Makeup Products", []),
    },
    "mobiles": {
        "1": ("Smartphones", ["phone", "smartphone", "mobile", "galaxy", "iphone",
                              "redmi", "poco", "oneplus", "realme", "oppo", "vivo",
                              "pixel", "motorola", "nokia", "iqoo", "infinix", "tecno"]),
        "2": ("Samsung", ["samsung"]),
        "3": ("iPhone", ["iphone"]),
        "4": ("OnePlus", ["oneplus"]),
        "5": ("All Mobile Products", []),
    },
    "shoes": {
        "1": ("Sneakers", ["sneaker"]),
        "2": ("Sports Shoes", ["sports shoe", "running shoe", "sports shoes"]),
        "3": ("Sandals", ["sandal"]),
        "4": ("Heels", ["heel"]),
        "5": ("All Shoes", []),
    },
}

COLUMN_ALIASES = {
    "product_name": ["product_name", "name", "title", "product"],
    "brand": ["brand", "manufacturer", "brand_name"],
    "price": ["price", "selling_price", "discounted_price", "final_price"],
    "rating": ["rating", "ratings", "average_rating", "stars"],
    "description": ["description", "about_product", "product_description"],
    "features": ["features", "specifications", "details"],
    "product_url": ["product_url", "url", "link"],
}


def standardize_columns(df):
    """Normalize column labels and map common dataset column names."""
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        for column in df.columns
    ]

    for standard_name, aliases in COLUMN_ALIASES.items():
        if standard_name not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df[standard_name] = df[alias]
                    break
    return df


def category_from_filename(filename):
    name = filename.lower()
    for category in ("beauty", "fashion", "laptops", "mobiles", "shoes"):
        if re.search(rf"(?:^|_){category}(?:_|$)", name):
            return category
    return None


def platform_from_filename(filename):
    name = filename.lower()
    for platform in PLATFORM_NAMES:
        if name.startswith(platform + "_"):
            return platform
    return None


def load_products():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATA_DIR}\n\n"
            "Check that this script is in the project root or scripts folder."
        )

    # Accept the cleaned filenames used in the current project.
    files = sorted(
        file for file in DATA_DIR.glob("*.csv")
        if "cleaned" in file.stem.lower()
    )
    if not files:
        raise FileNotFoundError(
            f"No cleaned CSV files found in:\n{DATA_DIR}\n"
            "Expected CSV filenames containing 'cleaned', such as "
            "'amazon_fashion_cleaned.csv'."
        )

    frames = []
    print(f"\nReading datasets from: {DATA_DIR}")

    for file in files:
        try:
            df = pd.read_csv(file, low_memory=False)
        except Exception as error:
            print(f"Skipped {file.name}: could not read CSV ({error})")
            continue

        df = standardize_columns(df)
        if "product_name" not in df.columns:
            print(f"Skipped {file.name}: no product name/title column found.")
            continue

        category_key = category_from_filename(file.stem)
        platform_key = platform_from_filename(file.stem)
        if category_key is None or platform_key is None:
            print(f"Skipped {file.name}: could not identify category/platform from filename.")
            continue

        df["category_key"] = category_key
        df["category"] = category_key
        df["platform"] = PLATFORM_NAMES[platform_key]
        df["source_file"] = file.name

        for column in ("brand", "description", "features", "product_url", "price", "rating"):
            if column not in df.columns:
                df[column] = ""

        for column in ("product_name", "brand", "description", "features", "product_url"):
            df[column] = (
                df[column].fillna("").astype(str)
                .replace({"nan": "", "None": ""})
                .map(html.unescape)
                .str.strip()
            )

        for column in ("price", "rating"):
            cleaned = (
                df[column].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("Rs.", "", regex=False)
                .str.strip()
            )
            df[column] = pd.to_numeric(cleaned, errors="coerce")

        df = df[df["product_name"].ne("")].copy()
        if not df.empty:
            frames.append(df)
            print(f"Loaded {len(df):,} products from {file.name}")

    if not frames:
        raise ValueError(
            "No usable product datasets were loaded. Check the CSV column names "
            "and filenames in data/cleaned_datasets."
        )

    products = pd.concat(frames, ignore_index=True, sort=False)
    products.drop_duplicates(
        subset=["source_file", "product_name", "product_url"],
        keep="first",
        inplace=True,
    )
    products.reset_index(drop=True, inplace=True)
    return products


def choose_from_options(prompt, options):
    while True:
        choice = input(prompt).strip()
        if choice in options:
            return choice
        print("Please choose one of the numbers shown.")


def choose_category(products):
    print("\n========== SELECT CATEGORY ==========")
    available_options = {}
    for number, (name, key) in CATEGORY_OPTIONS.items():
        count = int(products["category_key"].eq(key).sum())
        print(f"{number}. {name} ({count:,} products)")
        if count > 0:
            available_options[number] = (name, key)

    if not available_options:
        raise ValueError("No products are available for any supported category.")

    while True:
        choice = input("\nEnter category number: ").strip()
        if choice not in CATEGORY_OPTIONS:
            print("Enter a number from 1 to 5.")
            continue
        name, key = CATEGORY_OPTIONS[choice]
        selected = products[products["category_key"].eq(key)].copy()
        if selected.empty:
            print(f"No products are available in {name}. Choose another category.")
            continue
        print(f"\nSelected category: {name}")
        return selected, name, key


def choose_platform(products):
    print("\n========== SELECT PLATFORM ==========")
    available = sorted(products["platform"].dropna().unique().tolist())
    for number, platform in enumerate(available, start=1):
        count = int(products["platform"].eq(platform).sum())
        print(f"{number}. {platform} ({count:,} products)")

    while True:
        choice = input("\nEnter platform number: ").strip()
        if not choice.isdigit() or not 1 <= int(choice) <= len(available):
            print("Choose a valid number from the list.")
            continue
        platform = available[int(choice) - 1]
        selected = products[products["platform"].eq(platform)].copy()
        print(f"\nSelected platform: {platform}")
        return selected, platform


def choose_product_type(products, category_key):
    print("\n========== SELECT PRODUCT TYPE ==========")
    options = PRODUCT_TYPES[category_key]
    for number, (name, keywords) in options.items():
        if not keywords:
            count = len(products)
        else:
            pattern = "|".join(re.escape(word) for word in keywords)
            count = int(products["product_name"].str.contains(
                pattern, case=False, na=False, regex=True
            ).sum())
        print(f"{number}. {name} ({count:,} matching products)")

    while True:
        choice = input("\nEnter product type number: ").strip()
        if choice not in options:
            print("Choose a valid product type number.")
            continue
        type_name, keywords = options[choice]
        if not keywords:
            selected = products.copy()
        else:
            pattern = "|".join(re.escape(word) for word in keywords)
            selected = products[products["product_name"].str.contains(
                pattern, case=False, na=False, regex=True
            )].copy()

        if selected.empty:
            print(f"No products found for '{type_name}' on this platform.")
            print("Choose another product type.")
            continue

        selected.reset_index(drop=True, inplace=True)
        print(f"\nSelected product type: {type_name}")
        print(f"Matching products: {len(selected):,}")
        return selected, type_name


def prepare_product_text(products):
    products = products.copy()
    products["recommendation_text"] = (
        (products["product_name"].fillna("").astype(str) + " ") * 3
        + (products["brand"].fillna("").astype(str) + " ") * 2
        + products["features"].fillna("").astype(str) + " "
        + products["description"].fillna("").astype(str)
    )
    products["recommendation_text"] = (
        products["recommendation_text"].str.replace(r"\s+", " ", regex=True).str.strip()
    )
    # Avoid empty documents causing TF-IDF errors.
    empty_text = products["recommendation_text"].eq("")
    products.loc[empty_text, "recommendation_text"] = products.loc[empty_text, "product_name"]
    return products


def find_product(products):
    """Search product names and let the user choose one."""
    while True:
        keyword = input("\nSearch for a product (or type 'back'): ").strip()
        if keyword.lower() == "back":
            return None
        if not keyword:
            print("Enter a product name or keyword.")
            continue

        names = products["product_name"].fillna("").astype(str)
        mask = names.str.contains(keyword, case=False, na=False, regex=False)
        matches = products.loc[mask].head(10)

        if matches.empty:
            words = [word.casefold() for word in keyword.split()]
            mask = names.apply(lambda name: all(word in name.casefold() for word in words))
            matches = products.loc[mask].head(10)

        if matches.empty:
            print("No matching products found. Try another keyword.")
            continue

        print("\n========== MATCHING PRODUCTS ==========")
        for number, (_, row) in enumerate(matches.iterrows(), start=1):
            brand = row.get("brand", "")
            brand = "N/A" if pd.isna(brand) or not str(brand).strip() else str(brand)
            print(f"{number}. {row['product_name']} | Brand: {brand} | Platform: {row['platform']}")

        choice = input("\nChoose product number (0 to search again): ").strip()
        if choice == "0":
            continue
        if not choice.isdigit() or not 1 <= int(choice) <= len(matches):
            print("Choose a number shown in the list.")
            continue

        chosen_index = matches.index[int(choice) - 1]
        return products.index.get_loc(chosen_index)


def recommend_products(products, product_index, top_n=TOP_N):
    products = products.reset_index(drop=True)
    if not 0 <= product_index < len(products):
        raise IndexError("Selected product index is invalid.")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(products["recommendation_text"].fillna(""))
    except ValueError:
        print("Not enough usable text to build recommendations.")
        return products.iloc[0:0].copy()

    scores = cosine_similarity(matrix[product_index], matrix).ravel()
    scores[product_index] = -1

    selected_name = str(products.iloc[product_index]["product_name"])

    def word_overlap(name_a, name_b):
        words_a = set(re.findall(r"[a-z0-9]+", name_a.lower()))
        words_b = set(re.findall(r"[a-z0-9]+", name_b.lower()))
        union = words_a | words_b
        return len(words_a & words_b) / len(union) if union else 0

    ranked_indices = scores.argsort()[::-1]
    chosen_indices, chosen_names = [], []

    for index in ranked_indices:
        if scores[index] <= 0:
            continue
        candidate_name = str(products.iloc[index]["product_name"])
        if word_overlap(selected_name, candidate_name) >= 0.85:
            continue
        if any(word_overlap(candidate_name, previous) >= 0.85 for previous in chosen_names):
            continue
        chosen_indices.append(index)
        chosen_names.append(candidate_name)
        if len(chosen_indices) >= top_n:
            break

    recommendations = products.iloc[chosen_indices].copy()
    if not recommendations.empty:
        recommendations["similarity_score"] = [
            round(float(scores[index]), 3) for index in chosen_indices
        ]
    return recommendations


def display_recommendations(recommendations):
    if recommendations.empty:
        print("\nNo sufficiently similar recommendations found.")
        return
    columns = ["product_name", "brand", "price", "rating", "platform", "similarity_score"]
    for column in columns:
        if column not in recommendations.columns:
            recommendations[column] = ""
    print("\n========== TOP RECOMMENDATIONS ==========")
    print(recommendations[columns].to_string(index=False))


def save_recommendations(recommendations):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "sample_recommendations.csv"
    recommendations.to_csv(output_file, index=False)
    print(f"\nResults saved to:\n{output_file}")


def main():
    print("=" * 55)
    print("       PRODUCT RECOMMENDATION SYSTEM")
    print("        CONTENT-BASED FILTERING")
    print("=" * 55)
    print(f"Project folder: {PROJECT_ROOT}")

    try:
        products = load_products()
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\nCould not load datasets:\n{error}")
        return

    print(f"\nTotal products loaded: {len(products):,}")

    while True:
        category_products, category_name, category_key = choose_category(products)
        platform_products, platform_name = choose_platform(category_products)
        type_products, type_name = choose_product_type(platform_products, category_key)
        type_products = prepare_product_text(type_products)
        type_products.reset_index(drop=True, inplace=True)

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

        recommendations = recommend_products(type_products, product_index)
        display_recommendations(recommendations)
        if not recommendations.empty:
            save_recommendations(recommendations)

        again = input("\nSearch another product? (y/n): ").strip().lower()
        if again != "y":
            print("\nThank you for using the system!")
            break


if __name__ == "__main__":
    main()
