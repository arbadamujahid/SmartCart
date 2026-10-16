import os
import pandas as pd

from image_search import download_image, find_similar_products


METADATA_PATH = "outputs/image_recommendation/product_metadata.csv"

TEST_CATEGORIES = [
    "Shoes",
    "Fashion",
    "Mobiles",
    "Laptops",
]


metadata = pd.read_csv(METADATA_PATH)

print("=" * 70)
print("MULTI-CATEGORY IMAGE RECOMMENDATION TEST")
print("=" * 70)


for category in TEST_CATEGORIES:

    category_products = metadata[
        metadata["category"].astype(str).str.strip().str.lower()
        == category.lower()
    ]

    if category_products.empty:
        print(f"\n{category}: No products available.")
        continue

    product = category_products.iloc[0]

    print("\n" + "-" * 70)
    print(f"TEST CATEGORY: {category}")
    print(f"Product used: {product['product_name']}")
    print("-" * 70)

    try:
        image = download_image(product["image_url"])

        recommendations, predicted_category = find_similar_products(
            image,
            top_k=5
        )

        print(f"\nPredicted category: {predicted_category}")

        print("\nTop recommendations:")

        for position, result in enumerate(
            recommendations,
            start=1
        ):
            print(
                f"{position}. "
                f"{result['product_name'][:70]}"
            )

            print(
                f"   Category: {result['category']} | "
                f"Similarity: "
                f"{result['image_similarity']:.3f}"
            )

    except Exception as e:

        print(f"Test failed for {category}")
        print(f"Error: {e}")


print("\n" + "=" * 70)
print("MULTI-CATEGORY TEST COMPLETED")
print("=" * 70)