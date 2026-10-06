from ai_shopping_assistant import shopping_assistant , parse_query
from personalization import update_profile, get_preferences , personalize_results
from user_history import record_search


def run_member3_assistant(query, file_path):
    """Run the complete Member 3 shopping assistant."""

    # 1. Convert user query into filters
    filters = parse_query(query)

    # 2. Record the user's search
    record_search(query)

    # 3. Update user preferences
    update_profile(
        category=filters["category"],
        brand=filters["brand"],
        price=filters["max_price"]
    )

    # 4. Get products based on the user's query
    results = shopping_assistant(query, file_path)
    if not isinstance(results, str):
        results = personalize_results(results)

    # 5. Get updated preferences
    profile = get_preferences()

    return results, profile


if __name__ == "__main__":

    file_path = "data/processed/master_products.csv"

    query = input(
        "What product are you looking for? "
    )

    results, profile = run_member3_assistant(
        query,
        file_path
    )

    print("\nRecommended Products:\n")
    print(results)

    print("\nYour Current Preferences:\n")
    print(profile)