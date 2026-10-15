import pandas as pd

from image_recommendation.image_search import find_similar_products
from models.hybrid_recommender import hybrid_rank
from scripts.personalization import personalize_results


def recommend_from_image(image, top_k=10):
    recommendations, predicted_category = find_similar_products(
        image,
        top_k=top_k
    )

    if not recommendations:
        return pd.DataFrame(), predicted_category

    df = pd.DataFrame(recommendations)

    df["similarity_score"] = pd.to_numeric(
        df["image_similarity"],
        errors="coerce"
    ).fillna(0)

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    df = hybrid_rank(
        df,
        content_weight=0.75,
        rating_weight=0.25
    )

    df = personalize_results(df)

    return df, predicted_category