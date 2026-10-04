
"""
Hybrid Product Recommendation Module

Combines content similarity with available product ratings.

Note:
Product ratings are a quality signal, not collaborative filtering.
Collaborative filtering requires user-product interaction data.
"""

import pandas as pd


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def normalize_scores(values):
    """Normalize numeric values to a range between 0 and 1."""

    values = pd.to_numeric(values, errors="coerce")
    minimum = values.min()
    maximum = values.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(0.5, index=values.index)

    if maximum == minimum:
        return pd.Series(0.5, index=values.index)

    return (values - minimum) / (maximum - minimum)


# ============================================================
# HYBRID RECOMMENDATION RANKING
# ============================================================

def hybrid_rank(
    recommendations,
    content_weight=0.75,
    rating_weight=0.25,
):
    """
    Rank products using content similarity and available ratings.

    Required column:
        similarity_score

    Optional column:
        rating

    Products without ratings use content similarity alone.
    """

    if recommendations.empty:
        return recommendations.copy()

    if content_weight < 0 or rating_weight < 0:
        raise ValueError("Weights cannot be negative.")

    if content_weight + rating_weight <= 0:
        raise ValueError("At least one weight must be positive.")

    if "similarity_score" not in recommendations.columns:
        raise ValueError(
            "Recommendations must contain a similarity_score column."
        )

    results = recommendations.copy()

    # Normalize content similarity scores.
    content_scores = normalize_scores(
        results["similarity_score"]
    )

    # Read ratings if available.
    if "rating" in results.columns:
        ratings = pd.to_numeric(
            results["rating"],
            errors="coerce",
        )
    else:
        ratings = pd.Series(
            float("nan"),
            index=results.index,
        )

    has_rating = ratings.notna()

    # Start with content similarity for every product.
    results["hybrid_score"] = content_scores
    results["ranking_method"] = "content_only"

    # Combine both signals only for products with ratings.
    if has_rating.any():
        rating_scores = normalize_scores(ratings)

        results.loc[has_rating, "hybrid_score"] = (
            content_weight * content_scores[has_rating]
            + rating_weight * rating_scores[has_rating]
        )

        results.loc[has_rating, "ranking_method"] = (
            "content_plus_rating"
        )

    # Highest hybrid score appears first.
    results = results.sort_values(
        by="hybrid_score",
        ascending=False,
    ).reset_index(drop=True)

    return results


# ============================================================
# TEST DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    sample_products = pd.DataFrame({
        "product_name": [
            "Product A",
            "Product B",
            "Product C",
        ],
        "similarity_score": [
            0.90,
            0.80,
            0.70,
        ],
        "rating": [
            3.0,
            5.0,
            None,
        ],
    })

    print("=" * 60)
    print("       HYBRID RECOMMENDATION DEMONSTRATION")
    print("=" * 60)

    ranked_products = hybrid_rank(sample_products)

    print(
        ranked_products[
            [
                "product_name",
                "similarity_score",
                "rating",
                "hybrid_score",
                "ranking_method",
            ]
        ].to_string(index=False)
    )

    print("\nTest completed successfully.")