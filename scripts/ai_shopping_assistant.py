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
        r"(?:under|below|less than|upto|up to|within)\s*₹?\s*([\d,]+)",
        query
    )

    if price_match:
        filters["max_price"] = float(
            price_match.group(1).replace(",", "")
        )

    # Rating
    rating_match = re.search(
        r"(?:rating|rated)\s*(?:of|above|over|at least)?\s*(\d+(?:\.\d+)?)",
        query
    )

    if rating_match:
        filters["min_rating"] = float(rating_match.group(1))

    # RAM
    ram_match = re.search(r"(\d+)\s*gb\s*ram", query)

    if ram_match:
        filters["ram"] = ram_match.group(1)

    # Category
    categories = [
        "laptop",
        "laptops",
        "mobile",
        "mobiles",
        "phone",
        "phones",
        "shoes",
        "fashion",
        "beauty"
    ]

    for category in categories:
        if category in query:
            filters["category"] = category.rstrip("s")
            break

    # Brand
    brands = [
        "hp",
        "dell",
        "lenovo",
        "asus",
        "acer",
        "apple",
        "samsung",
        "oneplus",
        "nike",
        "adidas"
    ]

    for brand in brands:
        if brand in query:
            filters["brand"] = brand
            break

    return filters


def filter_products(df, filters):
    """Filter products according to user requirements."""

    result = df.copy()

    # Price filter
    if filters["max_price"] is not None:
        result = result[
            pd.to_numeric(result["price"], errors="coerce")
            <= filters["max_price"]
        ]

    # Rating filter
    if filters["min_rating"] is not None:
        result = result[
            pd.to_numeric(result["rating"], errors="coerce")
            >= filters["min_rating"]
        ]

    # Category filter
    if filters["category"] is not None:
        result = result[
            result["category"]
            .astype(str)
            .str.lower()
            .str.contains(filters["category"], na=False)
        ]

    # Brand filter
    if filters["brand"] is not None:
        result = result[
            result["brand"]
            .astype(str)
            .str.lower()
            .eq(filters["brand"].lower())
        ]

    # RAM / feature filter
    if filters["ram"] is not None:
        ram_value = filters["ram"]

        result = result[
            result["features"]
            .astype(str)
            .str.lower()
            .str.contains(
                rf"(?<!\d){ram_value}\s*gb\s*(?:ram|ddr\d)",
                regex=True,
                na=False
            )
        ]

    return result


def shopping_assistant(query, file_path):
    """Process a natural-language shopping query."""

    df = load_products(file_path)

    filters = parse_query(query)

    results = filter_products(df, filters)

    if results.empty:
        return "No products found matching your requirements."

    return results[
        [
            "product_name",
            "brand",
            "category",
            "price",
            "rating",
            "platform",
            "product_url"
        ]
    ].head(10)


if __name__ == "__main__":

    file_path = "data/cleaned/amazon_laptops_cleaned.csv"

    user_query = input(
        "What product are you looking for? "
    )

    results = shopping_assistant(
        user_query,
        file_path
    )

    print("\nRecommended Products:\n")
    print(results)