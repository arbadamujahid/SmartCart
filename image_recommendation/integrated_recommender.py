
import pandas as pd

from image_recommendation.image_search import find_similar_products


def recommend_from_image(image, top_k=10):
    recommendations, predicted_category = find_similar_products(
        image,
        top_k=top_k
    )

    if not recommendations:
        return pd.DataFrame(), predicted_category

    df = pd.DataFrame(recommendations)

    # Rank products by image similarity only.
    if "image_similarity" in df.columns:
        df["similarity_score"] = pd.to_numeric(
            df["image_similarity"],
            errors="coerce"
        ).fillna(0)

        df = df.sort_values(
            by="similarity_score",
            ascending=False
        ).reset_index(drop=True)

    return df, predicted_category
