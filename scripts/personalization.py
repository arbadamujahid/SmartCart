import json
import os


PROFILE_FILE = "user_profile.json"


def load_profile():
    """Load the user's preference profile."""
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    return {
        "preferred_categories": {},
        "preferred_brands": {},
        "price_ranges": []
    }


def update_profile(category=None, brand=None, price=None):
    """Update user preferences based on their activity."""

    profile = load_profile()

    if category:
        category = category.lower()
        profile["preferred_categories"][category] = (
            profile["preferred_categories"].get(category, 0) + 1
        )

    if brand:
        brand = brand.lower()
        profile["preferred_brands"][brand] = (
            profile["preferred_brands"].get(brand, 0) + 1
        )

    if price is not None:
        profile["price_ranges"].append(float(price))

    with open(PROFILE_FILE, "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=4)

    return profile


def get_preferences():
    """Return the user's current preferences."""
    return load_profile()


def personalize_results(results):
    """Rank products based on user's preferences."""

    profile = load_profile()

    preferred_categories = profile["preferred_categories"]
    preferred_brands = profile["preferred_brands"]
    preferred_prices = profile["price_ranges"]

    results = results.copy()

    def calculate_score(row):
        score = 0

        category = str(row.get("category", "")).lower()
        brand = str(row.get("brand", "")).lower()
        price = row.get("price", 0)

        if category in preferred_categories:
            score += preferred_categories[category] * 2

        if brand in preferred_brands:
            score += preferred_brands[brand] * 3

        if preferred_prices:
            average_price = sum(preferred_prices) / len(preferred_prices)

            if abs(float(price) - average_price) <= 5000:
                score += 2

        return score

    results["personalization_score"] = results.apply(
        calculate_score,
        axis=1
    )

    results = results.sort_values(
        by="personalization_score",
        ascending=False
    )

    return results


if __name__ == "__main__":

    print("Updating user profile...")

    update_profile(
        category="laptop",
        brand="HP",
        price=65000
    )

    print("\nCurrent User Preferences:")
    print(get_preferences())