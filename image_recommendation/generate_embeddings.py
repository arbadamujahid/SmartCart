import io
import os

import numpy as np
import pandas as pd
import requests
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


CSV_PATH = "data/processed/master_products.csv"
OUTPUT_DIR = "outputs/image_recommendation"

TOTAL_PRODUCTS = 10000
PRODUCTS_PER_CATEGORY = 2000


os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 60)
print("IMAGE RECOMMENDATION - EMBEDDING GENERATION")
print("=" * 60)

print("\nLoading product dataset...")

df = pd.read_csv(
    CSV_PATH,
    usecols=[
        "product_id",
        "platform",
        "category",
        "product_name",
        "brand",
        "price",
        "rating",
        "image_url",
        "product_url",
    ],
)

df = df[df["image_url"].notna()].copy()

df["category"] = df["category"].astype(str).str.strip()

categories = [
    "Shoes",
    "Fashion",
    "Mobiles",
    "Laptops",
    "Makeup",
]

selected_products = []

for category in categories:
    category_df = df[
        df["category"].str.lower() == category.lower()
    ]

    if len(category_df) == 0:
        print(f"No products found for {category}")
        continue

    sample_size = min(
        PRODUCTS_PER_CATEGORY,
        len(category_df)
    )

    selected = category_df.sample(
        n=sample_size,
        random_state=42
    )

    selected_products.append(selected)

df = pd.concat(
    selected_products,
    ignore_index=True
)

df = df.drop_duplicates(
    subset="product_id"
).reset_index(drop=True)

print(f"\nProducts selected: {len(df)}")

print("\nProducts by category:")
print(df["category"].value_counts())


print("\nLoading MobileNetV3 Small...")

weights = MobileNet_V3_Small_Weights.DEFAULT
preprocess = weights.transforms()

model = mobilenet_v3_small(weights=weights)
model.classifier = nn.Identity()

device = torch.device("cpu")

model = model.to(device)
model.eval()

print("Model loaded successfully.")


def download_image(url):
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        response.raise_for_status()

        return Image.open(
            io.BytesIO(response.content)
        ).convert("RGB")

    except Exception:
        return None


embeddings = []
valid_rows = []

print("\nGenerating image embeddings...\n")

with torch.inference_mode():

    for index, row in df.iterrows():

        print(
            f"[{index + 1}/{len(df)}] "
            f"{row['product_name'][:60]}"
        )

        image = download_image(row["image_url"])

        if image is None:
            print("   Skipped.")
            continue

        try:
            image_tensor = preprocess(image)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device)

            feature = model(image_tensor)

            feature = feature / feature.norm(
                dim=1,
                keepdim=True
            ).clamp_min(1e-12)

            embedding = (
                feature
                .squeeze(0)
                .cpu()
                .numpy()
            )

            embeddings.append(embedding)
            valid_rows.append(row)

            print("   ✓ Embedding generated")

        except Exception as e:
            print(f"   ✗ Processing failed: {e}")


if len(embeddings) == 0:

    print("\nNo embeddings were generated.")
    print("Please check the image URLs.")

else:

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    metadata = pd.DataFrame(valid_rows)

    embedding_path = os.path.join(
        OUTPUT_DIR,
        "product_embeddings.npy"
    )

    metadata_path = os.path.join(
        OUTPUT_DIR,
        "product_metadata.csv"
    )

    np.save(
        embedding_path,
        embeddings
    )

    metadata.to_csv(
        metadata_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)

    print(f"Successful images : {len(embeddings)}")
    print(f"Embedding shape    : {embeddings.shape}")

    print("\nSaved:")
    print(f"  {embedding_path}")
    print(f"  {metadata_path}")