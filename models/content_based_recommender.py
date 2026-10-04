
from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cleaned_datasets"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "sample_recommendations.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOP_N = 5


CATEGORY_OPTIONS = {
    "1": ("Fashion", "fashion"),
    "2": ("Laptops", "laptops"),
    "3": ("Makeup", "beauty"),
    "4": ("Mobiles", "mobiles"),
    "5": ("Shoes", "shoes"),
}


PRODUCT_TYPES = {
    "fashion": {
        "1": ("Men's Fashion", ["men", "mens", "male"]),
        "2": ("Women's Fashion", ["women", "womens", "female", "ladies"]),
        "3": ("T-shirts", ["t-shirt", "tshirt", "tee"]),
        "4": ("Shirts", ["shirt"]),
        "5": ("Jeans", ["jeans", "denim"]),
        "6": ("All Fashion", []),
    },
    "laptops": {
        "1": ("Gaming Laptops", ["gaming"]),
        "2": ("HP Laptops", ["hp"]),
        "3": ("Dell Laptops", ["dell"]),
        "4": ("Lenovo Laptops", ["lenovo"]),
        "5": ("Apple MacBooks", ["macbook", "mac book"]),
        "6": ("All Laptops", []),
    },
    "beauty": {
        "1": ("Lipstick", ["lipstick", "lip stick"]),
        "2": ("Foundation", ["foundation"]),
        "3": ("Mascara", ["mascara"]),
        "4": ("Face Makeup", ["face", "concealer", "blush"]),
        "5": ("Skincare", ["skincare", "moisturizer", "serum", "cleanser"]),
        "6": ("All Makeup and Beauty", []),
    },
    "mobiles": {
        "1": (
            "Smartphones",
            [
                "phone", "smartphone", "iphone", "samsung", "galaxy",
                "redmi", "poco", "realme", "oneplus", "oppo", "vivo",
                "motorola", "pixel", "iqoo", "infinix", "tecno",
                "nothing", "nokia", "honor", "cmf", "jio",
            ],
        ),
        "2": ("Samsung Phones", ["samsung", "galaxy"]),
        "3": ("iPhones", ["iphone"]),
        "4": ("All Mobiles", []),
    },
    "shoes": {
        "1": ("Sports Shoes", ["sports", "running", "athletic"]),
        "2": ("Sneakers", ["sneaker"]),
        "3": ("Formal Shoes", ["formal"]),
        "4": ("Heels", ["heels", "heel"]),
        "5": ("Sandals and Slippers", ["sandal", "slipper", "flip-flop"]),
        "6": ("All Shoes", []),
    },
}


# ============================================================
# MOBILE ACCESSORY FILTER
# ============================================================

MOBILE_ACCESSORY_PATTERN = (
    r"\b(?:"
    r"phone cases?|mobile cases?|back covers?|flip covers?|"
    r"protective cases?|silicone cases?|phone covers?|mobile covers?|"
    r"screen protectors?|tempered glass|camera protectors?|"
    r"chargers?|charging cables?|usb cables?|data cables?|"
    r"adapters?|power banks?|phone holders?|mobile holders?|"
    r"socket holders?|phone stands?|mobile stands?|car mounts?|"
    r"earphones?|headphones?|earbuds?|"
    r"replacement screens?|replacement batteries?|"
    r"phone skins?|mobile skins?|charging docks?|"
    r"phone pouch(?:es)?|mobile pouch(?:es)?|pouch(?:es)?|"
    r"phone rings?|ring holders?|pop sockets?|"
    r"camera lenses?|phone lenses?|mobile phone lenses?|"
    r"camera kits?|screen expanders?|selfie sticks?|tripods?|"
    r"memory cards?|sim ejector pins?|phone grips?|mobile grips?|"
    r"phone straps?|mobile straps?|stylus pens?|replacement parts?|"
    r"tablet cases?|book covers?|body panels?|"
    r"front and back panels?|sleeves?|otg cables?|"
    r"mobile batter(?:y|ies)|phone batter(?:y|ies)|"
    r"battery\s*[-:]|[^|]{0,60}\bbatter(?:y|ies)\b|"
    r"tablet covers?|tab covers?"
    r")\b"
)


def remove_mobile_accessories(products, category_key):
    """Exclude common mobile accessories from mobile results."""

    products = products.copy()

    if category_key != "mobiles" or products.empty:
        return products

    accessory_mask = products["product_name"].fillna("").astype(str).str.contains(
        MOBILE_ACCESSORY_PATTERN,
        case=False,
        na=False,
        regex=True,
    )

    return products.loc[~accessory_mask].copy()


# ============================================================
# DATA HELPERS
# ============================================================

def find_column(df, possible_names):
    """Find a column regardless of capitalization."""

    columns = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in df.columns
    }

    for name in possible_names:
        key = name.strip().lower().replace(" ", "_")
        if key in columns:
            return columns[key]

    return None


def load_category_data(category_key):
    """Load and combine the cleaned datasets for a category."""

    if category_key == "beauty":
        files = sorted(
            set(DATA_DIR.glob("*_beauty_cleaned.csv"))
            | set(DATA_DIR.glob("*_makeup_cleaned.csv"))
        )
    else:
        files = sorted(DATA_DIR.glob(f"*_{category_key}_cleaned.csv"))

    if not files:
        print(f"\nNo dataset files found for: {category_key}")
        print(f"Expected folder: {DATA_DIR}")
        return pd.DataFrame()

    frames = []

    for file_path in files:
        try:
            df = pd.read_csv(file_path, low_memory=False)

            name_col = find_column(
                df,
                ["product_name", "name", "title", "product", "product_title"],
            )

            if name_col is None:
                print(f"Skipping {file_path.name}: product name column missing.")
                continue

            description_col = find_column(
                df, ["description", "product_description", "details"]
            )
            brand_col = find_column(
                df, ["brand", "manufacturer", "seller"]
            )
            price_col = find_column(
                df, ["price", "selling_price", "discounted_price", "amount"]
            )
            url_col = find_column(
                df, ["url", "product_url", "link"]
            )
            category_col = find_column(
                df, ["category", "product_category", "subcategory"]
            )

            result = pd.DataFrame()

            result["product_name"] = df[name_col].fillna("").astype(str)

            result["description"] = (
                df[description_col].fillna("").astype(str)
                if description_col else ""
            )

            result["brand"] = (
                df[brand_col].fillna("").astype(str)
                if brand_col else ""
            )

            result["price"] = (
                df[price_col] if price_col else ""
            )

            result["url"] = (
                df[url_col].fillna("").astype(str)
                if url_col else ""
            )

            result["category"] = (
                df[category_col].fillna("").astype(str)
                if category_col else category_key
            )

            # The dataset filename identifies the platform.
            result["source"] = file_path.name.split("_")[0].title()

            result = result[
                result["product_name"].str.strip().ne("")
            ].copy()

            frames.append(result)
            print(f"Loaded {len(result):,} products from {file_path.name}")

        except Exception as error:
            print(f"Could not read {file_path.name}: {error}")

    if not frames:
        return pd.DataFrame()

    products = pd.concat(frames, ignore_index=True)

    products = products.drop_duplicates(
        subset=["product_name", "source"],
        keep="first",
    )

    products = remove_mobile_accessories(products, category_key)

    return products.reset_index(drop=True)


# ============================================================
# PRODUCT TYPE SELECTION
# ============================================================

def choose_product_type(products, category_key):
    """Filter products by the selected product type."""

    options = PRODUCT_TYPES[category_key]

    print("\n========== PRODUCT TYPES ==========")

    for number, (label, keywords) in options.items():
        if keywords:
            pattern = "|".join(re.escape(word) for word in keywords)
            count = products["product_name"].str.contains(
                pattern, case=False, na=False, regex=True
            ).sum()
        else:
            count = len(products)

        print(f"{number}. {label} ({count:,} products)")

    choice = input("\nChoose a product type: ").strip()

    if choice not in options:
        print("Invalid choice.")
        return pd.DataFrame()

    label, keywords = options[choice]

    if keywords:
        pattern = "|".join(re.escape(word) for word in keywords)
        selected = products.loc[
            products["product_name"].str.contains(
                pattern, case=False, na=False, regex=True
            )
        ].copy()
    else:
        selected = products.copy()

    selected = remove_mobile_accessories(selected, category_key)

    print(f"\nProducts available for {label}: {len(selected):,}")

    return selected.reset_index(drop=True)


# ============================================================
# PLATFORM SELECTION
# ============================================================

def choose_platform(products):
    """Allow users to choose a platform or search all platforms."""

    if products.empty:
        return products

    available_platforms = sorted(
        products["source"].dropna().astype(str).unique(),
        key=str.lower,
    )

    print("\n========== PLATFORM SELECTION ==========")
    print("1. All Platforms")

    for number, platform in enumerate(available_platforms, start=2):
        print(f"{number}. {platform}")

    choice = input("\nChoose a platform: ").strip()

    if choice == "1":
        selected = products.copy()
        print("\nShowing products from all available platforms.")
        return selected.reset_index(drop=True)

    platform_index = int(choice) - 2 if choice.isdigit() else -1

    if platform_index < 0 or platform_index >= len(available_platforms):
        print("Invalid platform selection.")
        return pd.DataFrame()

    chosen_platform = available_platforms[platform_index]

    selected = products.loc[
        products["source"].str.lower() == chosen_platform.lower()
    ].copy()

    print(f"\nPlatform selected: {chosen_platform}")
    print(f"Products available: {len(selected):,}")

    return selected.reset_index(drop=True)


# ============================================================
# PRODUCT SEARCH
# ============================================================

def find_product(products, category_key="other"):
    """Search products by name or brand."""

    products = remove_mobile_accessories(products, category_key)

    if products.empty:
        print("\nNo products are available for this selection.")
        return None

    print("\n========== FIND A PRODUCT ==========")

    keyword = input(
        "Enter a product name or keyword (e.g. Samsung): "
    ).strip()

    if not keyword:
        print("Please enter a search keyword.")
        return None

    searchable = (
        products["product_name"].fillna("").astype(str)
        + " "
        + products["brand"].fillna("").astype(str)
    )

    matches = products.loc[
        searchable.str.contains(
            re.escape(keyword),
            case=False,
            na=False,
            regex=True,
        )
    ].copy()

    if matches.empty:
        print("\nNo matching products found. Try another keyword.")
        print("\nSome available products:")

        for name in products["product_name"].head(10):
            print(f"- {name}")

        return None

    matches = matches.head(50).reset_index(drop=True)

    print("\n========== MATCHING PRODUCTS ==========")

    for index, row in matches.iterrows():
        price = str(row["price"]).strip()
        price_text = (
            f" | Price: {price}"
            if price and price.lower() != "nan"
            else ""
        )

        print(
            f"{index + 1}. {row['product_name']} "
            f"| {row['source']}{price_text}"
        )

    selection = input("\nSelect a product number: ").strip()

    if not selection.isdigit():
        print("Please enter a valid number.")
        return None

    index = int(selection) - 1

    if index < 0 or index >= len(matches):
        print("Invalid product number.")
        return None

    return matches.iloc[index]


# ============================================================
# CONTENT-BASED RECOMMENDATION ENGINE
# ============================================================

def build_product_text(products):
    """Combine product information into text for TF-IDF."""

    columns = [
        column
        for column in ["product_name", "brand", "category", "description"]
        if column in products.columns
    ]

    return products[columns].fillna("").astype(str).agg(" ".join, axis=1)


def recommend_products(
    products,
    selected_product,
    top_n=TOP_N,
    category_key="other",
):
    """Recommend similar products using TF-IDF and cosine similarity."""

    products = remove_mobile_accessories(products, category_key)
    products = products.reset_index(drop=True)

    if products.empty:
        print("\nNo eligible products are available.")
        return pd.DataFrame()

    same_product = (
        products["product_name"].eq(selected_product["product_name"])
        & products["source"].eq(selected_product["source"])
    )

    selected_rows = products.loc[same_product]

    if selected_rows.empty:
        print("\nSelected product is not present in the recommendation data.")
        return pd.DataFrame()

    selected_index = selected_rows.index[0]

    if len(products) < 2:
        print("\nNot enough products to generate recommendations.")
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
    )

    try:
        matrix = vectorizer.fit_transform(build_product_text(products))
    except ValueError:
        print("\nNot enough searchable product text.")
        return pd.DataFrame()

    scores = cosine_similarity(
        matrix[selected_index],
        matrix,
    ).flatten()

    scores[selected_index] = -1

    ranked_indices = [
        index
        for index in scores.argsort()[::-1]
        if scores[index] >= 0
    ][:top_n]

    if not ranked_indices:
        print("\nNo similar products could be found.")
        return pd.DataFrame()

    recommendations = products.iloc[ranked_indices].copy()
    recommendations["similarity_score"] = [
        round(float(scores[index]), 4)
        for index in ranked_indices
    ]

    return recommendations.reset_index(drop=True)


# ============================================================
# DISPLAY AND CSV OUTPUT
# ============================================================

def display_recommendations(selected_product, recommendations):
    print("\n========== SELECTED PRODUCT ==========")
    print(f"Name: {selected_product['product_name']}")
    print(f"Brand: {selected_product['brand']}")
    print(f"Price: {selected_product['price']}")
    print(f"Platform: {selected_product['source']}")

    print("\n========== RECOMMENDED PRODUCTS ==========")

    if recommendations.empty:
        print("No recommendations to display.")
        return

    for index, row in recommendations.iterrows():
        print(f"\n{index + 1}. {row['product_name']}")
        print(f"   Brand: {row['brand']}")
        print(f"   Price: {row['price']}")
        print(f"   Platform: {row['source']}")
        print(f"   Similarity: {row['similarity_score']:.4f}")

        if row["url"]:
            print(f"   URL: {row['url']}")


def save_recommendations(selected_product, recommendations):
    """Save recommendations to CSV."""

    if recommendations.empty:
        print("\nNo recommendations to save.")
        return

    output = recommendations.copy()
    output.insert(0, "selected_product", selected_product["product_name"])
    output.insert(1, "selected_platform", selected_product["source"])

    output.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nRecommendations saved to:\n{OUTPUT_FILE}")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    print("=" * 55)
    print("       CONTENT-BASED PRODUCT RECOMMENDER")
    print("=" * 55)

    if not DATA_DIR.exists():
        print(f"\nDataset folder not found: {DATA_DIR}")
        return

    print("\n========== CATEGORIES ==========")

    for number, (label, _) in CATEGORY_OPTIONS.items():
        print(f"{number}. {label}")

    category_choice = input("\nChoose a category: ").strip()

    if category_choice not in CATEGORY_OPTIONS:
        print("Invalid category selection.")
        return

    category_name, category_key = CATEGORY_OPTIONS[category_choice]

    print(f"\nLoading {category_name} datasets...")
    products = load_category_data(category_key)

    if products.empty:
        print("No products could be loaded.")
        return

    print(f"\nTotal eligible products: {len(products):,}")

    # Step 1: Select product type.
    type_products = choose_product_type(products, category_key)

    if type_products.empty:
        print("\nNo products match this product type.")
        return

    # Step 2: Select platform.
    platform_products = choose_platform(type_products)

    if platform_products.empty:
        print("\nNo products match this platform selection.")
        return

    # Step 3: Search for a product.
    selected_product = find_product(platform_products, category_key)

    if selected_product is None:
        return

    # Step 4: Generate recommendations within the chosen platform scope.
    recommendations = recommend_products(
        platform_products,
        selected_product,
        top_n=TOP_N,
        category_key=category_key,
    )

    # Step 5: Display and save results.
    display_recommendations(selected_product, recommendations)
    save_recommendations(selected_product, recommendations)


if __name__ == "__main__":
    main()