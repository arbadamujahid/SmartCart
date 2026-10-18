# Member 2 — Content-Based Recommendation Engine

## 1\. Module Name

**Content-Based Product Recommendation Engine**

## 2\. Overview

This module recommends products similar to a user-selected product by analyzing product names, brands, features, and descriptions. It helps users discover relevant products within supported categories and shopping platforms.

## 3\. Key Features

- Content-based product filtering.
- TF-IDF text feature extraction.
- Cosine similarity scoring.
- Selection by category, platform, and product type.
- Top-5 recommendations by default.
- CSV export of recommendation results.

## 4\. Implementation Details

The module uses Python and Pandas for product data processing, Scikit-learn's `TfidfVectorizer` to convert product text into numerical features, and cosine similarity to rank similar products. Regular expressions and HTML text decoding are used for data cleaning.

## 5\. Files and Folder Structure

- `models/content_based_recommender.py` — Loads product datasets, filters products, calculates similarity scores, displays recommendations, and saves results.
- `data/cleaned_datasets/` — Contains the cleaned product CSV files expected by the recommender.
- `data/processed/sample_recommendations.csv` — Stores generated recommendations.

## 6\. Input and Output

**Input:** Cleaned product datasets containing product names and available attributes such as brands, features, and descriptions.

**Output:** A ranked list of similar products with product details and similarity scores, saved to a CSV file.

## 7\. Integration with SmartCart

The module uses cleaned product data prepared by the data-processing component and provides recommendation results for use by the main SmartCart application. Final application integration should match the implementation in the merged branch.

## 8\. How to Run

Install dependencies:

``` bash
pip install pandas scikit-learn
```

Run from the project root:

``` bash
python models/content_based_recommender.py
```

Ensure the required cleaned CSV files are available in `data/cleaned_datasets/`.


## 9\. Current Status and Limitations

**Implemented:** Content-based filtering, TF-IDF feature extraction, cosine similarity ranking, product selection, Top-5 recommendations, and CSV export.

**Limitations:** Recommendation quality depends on product data completeness and text similarity. Collaborative filtering and user-interaction-based learning are not implemented in this module. Live prices and stock availability are not guaranteed.
