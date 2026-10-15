import io
import os

import numpy as np
import pandas as pd
import requests
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


EMBEDDINGS_PATH = "outputs/image_recommendation/product_embeddings.npy"
METADATA_PATH = "outputs/image_recommendation/product_metadata.csv"

IMAGE_WEIGHT = 0.80
CATEGORY_WEIGHT = 0.10
RATING_WEIGHT = 0.10


# Load model
weights = MobileNet_V3_Small_Weights.DEFAULT

model = mobilenet_v3_small(weights=weights)
model.classifier = nn.Identity()
model.eval()

preprocess = weights.transforms()


# Load embeddings and product data
if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(f"Embeddings not found: {EMBEDDINGS_PATH}")

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

embeddings = np.load(EMBEDDINGS_PATH)
metadata = pd.read_csv(METADATA_PATH)

print(f"Loaded {len(embeddings)} product embeddings.")
print(f"Loaded {len(metadata)} product records.")


def get_embedding(image):
    """Convert an image into a normalized embedding."""

    image = image.convert("RGB")

    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        feature = model(tensor)

    feature = feature.squeeze(0).numpy()

    norm = np.linalg.norm(feature)

    if norm > 0:
        feature = feature / norm

    return feature


def predict_category(similarities):
    """Predict category using the closest products."""

    top_indices = np.argsort(similarities)[::-1][:10]

    categories = (
        metadata.iloc[top_indices]["category"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    if categories.empty:
        return None

    return categories.mode().iloc[0]


def rating_score(rating):
    """Convert rating into a 0-1 score."""

    if pd.isna(rating):
        return 0.5

    try:
        rating = float(rating)
        rating = max(0, min(5, rating))
        return rating / 5

    except (ValueError, TypeError):
        return 0.5


def product_key(product):
    """Create a key for removing duplicate products."""

    name = str(product.get("product_name", "")).lower()
    brand = str(product.get("brand", "")).lower()

    return brand + "|" + " ".join(name.split()[:8])


def remove_duplicates(results, top_k):
    """Remove duplicate product variants."""

    selected = []
    seen = set()

    for product in results:
        key = product_key(product)

        if key in seen:
            continue

        seen.add(key)
        selected.append(product)

        if len(selected) == top_k:
            break

    return selected


def find_similar_products(image, top_k=5):
    """Find products similar to the uploaded image."""

    query_embedding = get_embedding(image)

    similarities = np.dot(
        embeddings,
        query_embedding
    )

    predicted_category = predict_category(
        similarities
    )

    category_scores = np.array([
        1.0
        if (
            predicted_category is not None
            and pd.notna(category)
            and str(category).strip().lower()
            == str(predicted_category).strip().lower()
        )
        else 0.0
        for category in metadata["category"]
    ])

    rating_scores = np.array([
        rating_score(rating)
        for rating in metadata["rating"]
    ])

    final_scores = (
        IMAGE_WEIGHT * similarities
        + CATEGORY_WEIGHT * category_scores
        + RATING_WEIGHT * rating_scores
    )

    candidate_count = min(
        len(metadata),
        max(top_k * 10, 50)
    )

    indices = np.argsort(
        final_scores
    )[::-1][:candidate_count]

    results = []

    for index in indices:

        product = metadata.iloc[index]

        results.append({
            "product_id": product.get("product_id", ""),
            "platform": product.get("platform", ""),
            "category": product.get("category", ""),
            "product_name": product.get("product_name", ""),
            "brand": product.get("brand", ""),
            "price": product.get("price", ""),
            "rating": product.get("rating", ""),
            "image_url": product.get("image_url", ""),
            "product_url": product.get("product_url", ""),

            "image_similarity": float(
                similarities[index]
            ),

            "category_score": float(
                category_scores[index]
            ),

            "rating_score": float(
                rating_scores[index]
            ),

            "final_score": float(
                final_scores[index]
            )
        })

    results = remove_duplicates(
        results,
        top_k
    )

    return results, predicted_category