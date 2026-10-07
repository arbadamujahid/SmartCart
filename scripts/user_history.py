import json
import os
from datetime import datetime


HISTORY_FILE = "user_history.json"


def load_history():
    """Load previous user activity."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    return {
        "search_history": [],
        "click_history": []
    }


def record_search(query):
    """Record a user's search query."""

    history = load_history()

    history["search_history"].append({
        "query": query,
        "timestamp": datetime.now().isoformat()
    })

    save_history(history)


def record_click(product_id, product_name):
    """Record a product clicked/viewed by the user."""

    history = load_history()

    history["click_history"].append({
        "product_id": str(product_id),
        "product_name": product_name,
        "timestamp": datetime.now().isoformat()
    })

    save_history(history)


def save_history(history):
    """Save user activity to a JSON file."""

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def get_history():
    """Return the complete user activity history."""
    return load_history()


if __name__ == "__main__":

    record_search(
        "gaming laptop under 70000 with 16GB RAM"
    )

    record_click(
        "B0955H5CJ4R",
        "HP 15.6 FHD Laptop"
    )

    print("User History:")
    print(get_history())