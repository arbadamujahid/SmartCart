
import pandas as pd
import re


def load_products(file_path):
    """Load product data from CSV."""
    return pd.read_csv(file_path)


def parse_query(query):
    """Convert a natural-language shopping query into filters."""
    query = query.lower()

    filters = {
        "max_price": None,
        "min_rating": None,
        "brand": None,
        "category": None,
        "ram": None
    }

    # Price / budget
    price_match = re.search(
        r"\b(?:under|below|less than|upto|up to|within)\s*₹?\s*([\d,]+)",
        query
    )
    if price_match:
        filters["max_price"] = float(
            price_match.group(1).replace(",", "")
        )

    # Rating
    rating_match = re.search(
        r"\b(?:rating|rated)\s*(?:of|above|over|at least)?\s*(\d+(?:\.\d+)?)",
        query
    )
    if rating_match:
        filters["min_rating"] = float(rating_match.group(1))

    # RAM
    ram_match = re.search(r"\b(\d+)\s*gb\s*ram\b", query)
    if ram_match:
        filters["ram"] = ram_match.group(1)

    # Category
    categories = [
        "laptops", "laptop",
        "mobiles", "mobile",
        "phones", "phone",
        "shoes", "fashion", "beauty"
    ]

    for category in categories:
        if re.search(r"\b" + re.escape(category) + r"\b", query):
            if category in ("laptops", "laptop"):
                filters["category"] = "laptop"
            elif category in ("mobiles", "mobile", "phones", "phone"):
                filters["category"] = "mobile"
            else:
                filters["category"] = category
            break

    # Brand
    brands = [
        "oneplus", "samsung", "lenovo", "adidas",
        "apple", "asus", "acer", "dell", "nike", "hp"
    ]

    for brand in brands:
        if re.search(r"\b" + re.escape(brand) + r"\b", query):
            filters["brand"] = brand
            break

    return filters


def filter_products(df, filters):
    """Filter products according to user requirements."""
    result = df.copy()

    if filters["max_price"] is not None and "price" in result.columns:
        prices = pd.to_numeric(result["price"], errors="coerce")
        result = result[prices <= filters["max_price"]]

    if filters["min_rating"] is not None and "rating" in result.columns:
        ratings = pd.to_numeric(result["rating"], errors="coerce")
        result = result[ratings >= filters["min_rating"]]

    if filters["category"] is not None and "category" in result.columns:
        category = filters["category"]
        category_pattern = {
            "laptop": r"\blaptops?\b",
            "mobile": r"\b(?:mobiles?|phones?)\b",
            "shoes": r"\bshoes?\b",
            "fashion": r"\bfashion\b",
            "beauty": r"\bbeauty\b"
        }.get(category, re.escape(category))

        result = result[
            result["category"].astype(str).str.lower().str.contains(
                category_pattern, regex=True, na=False
            )
        ]

    if filters["brand"] is not None and "brand" in result.columns:
        result = result[
            result["brand"].astype(str).str.lower().str.strip()
            == filters["brand"]
        ]

    if filters["ram"] is not None:
        searchable_columns = [
            column for column in ["features", "product_name", "description"]
            if column in result.columns
        ]

        if searchable_columns:
            text = result[searchable_columns].fillna("").astype(str).agg(
                " ".join, axis=1
            )

            ram_pattern = (
                rf"(?<!\d){re.escape(filters['ram'])}\s*gb\s*(?:ram|ddr\d)"
            )

            result = result[
                text.str.contains(ram_pattern, case=False, regex=True, na=False)
            ]

    return result


def shopping_assistant(query, file_path):
    """Find products matching the shopping query."""
    df = load_products(file_path)
    filters = parse_query(query)
    results = filter_products(df, filters)

    if results.empty:
        return pd.DataFrame(
            columns=[
                "product_name", "brand", "category", "price",
                "rating", "platform", "image_url", "product_url"
            ]
        )

    # Prioritize products with image URLs.
    if "image_url" in results.columns:
        results = results.copy()
        image_urls = results["image_url"].fillna("").astype(str).str.strip()

        valid_urls = image_urls.str.match(r"^https?://", case=False)
        results = pd.concat(
            [results[valid_urls], results[~valid_urls]],
            ignore_index=True
        )

    display_columns = [
        "product_name",
        "brand",
        "category",
        "price",
        "rating",
        "platform",
        "image_url",
        "product_url"
    ]

    available_columns = [
        column for column in display_columns
        if column in results.columns
    ]

    return results[available_columns].head(10)


if __name__ == "__main__":
    file_path = "data/processed/master_products.csv"
    user_query = input("What product are you looking for? ")

    results = shopping_assistant(user_query, file_path)

    print("\nRecommended Products:\n")
    print(results.to_string(index=False))

