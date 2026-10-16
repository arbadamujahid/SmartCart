from pathlib import Path
from io import BytesIO
import ast
import re
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps, UnidentifiedImageError
from sympy import python
from image_recommendation.integrated_recommender import recommend_from_image
from scripts.ai_shopping_assistant import shopping_assistant

# -------------------- PAGE SETTINGS --------------------

st.set_page_config(
    page_title="SmartCart",
    page_icon="🛒",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "master_products.csv"
IMAGE_SIZE = (320, 250)
CATEGORY_IMAGE_SIZE = (800, 450)
PRODUCT_TYPE_PLACEHOLDER = "Choose product type"

# -------------------- CATEGORY INFORMATION --------------------
CATEGORY_ICONS = {
    "Fashion": "👗",
    "Beauty": "💄",
    "Mobiles": "📱",
    "Laptops": "💻",
    "Shoes": "👟"
}
CATEGORY_IMAGES = {
    # Each category has its own distinct image URL.
    "Fashion": (
        "https://images.unsplash.com/photo-1445205170230-053b83016050"
        "?auto=format&fit=crop&w=900&q=85&v=smartcart-fashion-3"
    ),
    "Beauty": (
        "https://images.unsplash.com/photo-1611930022073-b7a4ba5fcccd"
        "?auto=format&fit=crop&w=900&q=85&v=smartcart-beauty-3"
    ),
    "Mobiles": (
        "https://images.unsplash.com/photo-1598327105666-5b89351aff97"
        "?auto=format&fit=crop&w=900&q=85&v=smartcart-mobiles-3"
    ),
    "Laptops": (
        "https://images.unsplash.com/photo-1496181133206-80ce9b88a853"
        "?auto=format&fit=crop&w=900&q=85&v=smartcart-laptops-3"
    ),
    "Shoes": (
        "https://images.unsplash.com/photo-1600185365483-26d7a4cc7519"
        "?auto=format&fit=crop&w=900&q=85&v=smartcart-shoes-3"
    ),
}

# Separate images for the three home-page discovery cards, so they do not
# reuse the category photos shown farther down the page.
HOME_DISCOVERY_IMAGES = {
    "browse": (
        "https://images.openai.com/static-rsc-4/Qcosp6Orrg8uwcm5-6NalT-vGJkZIieQ3WPT56QuJt5TouvrLwHYRXuSsz_VxpVQJlyBXqprNrNEAcYYhL3z-oIRRIW35QaUjxzG_eNqb5cEaMA0DcEoawtCuvmv-VuC4onXRnipr4XwsRAckdBwu2tlBJII6rGENIvpKky1PWE?purpose=inline"
    ),
    "image_search": (
        "https://images.openai.com/static-rsc-4/rz9op0FulvAC8_T-CGF7F1lDBVa88TRv9WgDyUT1yvAzzf8FOIpLvmd_p0S7Vpdq19X9ov1x7f4TnMidauybzWCwrdGKagxAzbwTdXsfi1swdgT9ZVC_bazLSiCa90cVA_XJokZFUHW8258w38eQRcP4FnFdykX5b02bTubmKK5O5UtFNnLO76IkXKFq6pc4?purpose=inline"
    ),
    "smart_search": (
        "https://images.openai.com/static-rsc-4/yPGVaybfxFaxwdlGFM23RcItetD1OacQMkCxN5uwM7wtlEHbIHOuDcjf-BzN2It8aQdgZ-StdpSTrbyaGslXr4sKggJVTKtndYVfgI2TyvibRPHUk8opJZaYcxtvsig7pLD4qMy0blCHFyQxISeUX1wg1smreBoQ68soEzu0dXc?purpose=inline"
    ),
}

HERO_IMAGE_URL = (
    "https://images.unsplash.com/photo-1607082348824-0a96f2a4b9da"
    "?auto=format&fit=crop&w=1800&q=85&v=smartcart-hero-3"
)

PLATFORM_ICONS = {
    "Amazon": "📦",
    "Flipkart": "🛒",
    "Myntra": "👗",
    "Nykaa": "💄"
}
PLATFORM_COLORS = {
    "Amazon": "#F7F1E8",
    "Flipkart": "#E5D4BF",
    "Myntra": "#F7F1E8",
    "Nykaa": "#E5D4BF"
}
SUBCATEGORIES = {
    "Fashion": [
        "All", "Shirts", "T-Shirts", "Tops", "Jeans",
        "Trousers", "Dresses", "Jackets", "Kurtas",
        "Watches", "Bags"
    ],
    "Shoes": [
        "All", "Sneakers", "Sports Shoes", "Sandals",
        "Heels", "Flats", "Formal Shoes", "Boots"
    ],
    "Beauty": [
        "All", "Skincare", "Makeup", "Haircare",
        "Shampoo", "Conditioner", "Perfume", "Body Care"
    ],
    "Mobiles": [
        "All", "Smartphones", "Apple iPhone", "Samsung Phones",
        "Phone Accessories", "Screen Guards", "Cases",
        "Chargers", "Earphones", "Cables"
    ],
    "Laptops": [
        "All", "Gaming Laptops", "HP", "Dell", "Lenovo",
        "ASUS", "Acer", "Apple MacBook"
    ]
}
SUBCATEGORY_TERMS = {
    "Shirts": ["shirt"],
    "T-Shirts": ["t-shirt", "t shirt", "tee"],
    "Tops": ["top"],
    "Jeans": ["jeans", "denim"],
    "Trousers": ["trousers", "pants"],
    "Dresses": ["dress", "gown"],
    "Jackets": ["jacket", "coat"],
    "Kurtas": ["kurta", "kurti"],
    "Watches": ["watch"],
    "Bags": ["bag", "backpack", "handbag"],
    "Sneakers": ["sneaker"],
    "Sports Shoes": [
        "sports shoe", "running shoe", "training shoe"
    ],
    "Sandals": ["sandal"],
    "Heels": ["heel"],
    "Flats": ["flat"],
    "Formal Shoes": ["formal shoe", "oxford shoe"],
    "Boots": ["boot"],
    "Skincare": [
        "skin care", "skincare", "moisturizer", "serum"
    ],
    "Makeup": [
        "makeup", "lipstick", "foundation", "mascara"
    ],
    "Haircare": ["hair care", "haircare", "hair oil"],
    "Shampoo": ["shampoo"],
    "Conditioner": ["conditioner"],
    "Perfume": ["perfume", "fragrance"],
    "Body Care": ["body lotion", "body wash", "body care"],
    "Smartphones": [
        "smartphone", "mobile phone", "cell phone"
    ],
    "Apple iPhone": ["iphone", "apple iphone"],
    "Samsung Phones": ["samsung"],
    "Phone Accessories": [
        "phone case", "back cover", "mobile cover",
        "screen guard", "screen protector", "charger",
        "cable", "earphone", "headphone", "earbuds",
        "phone holder"
    ],
    "Screen Guards": [
        "screen guard", "screen protector"
    ],
    "Cases": [
        "phone case", "back cover", "mobile cover", "case"
    ],
    "Chargers": ["charger", "charging adapter"],
    "Earphones": ["earphone", "headphone", "earbuds"],
    "Cables": [
        "usb cable", "charging cable", "data cable"
    ],
    "Gaming Laptops": ["gaming laptop"],
    "HP": ["hp"],
    "Dell": ["dell"],
    "Lenovo": ["lenovo"],
    "ASUS": ["asus"],
    "Acer": ["acer"],
    "Apple MacBook": ["macbook"]
}

# -------------------- SESSION STATE --------------------
defaults = {
    "page": "home",
    "category": "",
    "platform": "",
    "product_type": "All",
    "product_type_widget": "All",
    "menu_open": False,
    "assistant_results": None,
    "text_results": None,
    "image_results": None,
    "detected_category": "Unknown"
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -------------------- CUSTOM DESIGN --------------------
hide_sidebar = st.session_state.page == "results"
sidebar_css = """
[data-testid="stSidebar"] {
    display: none !important;
}
""" if hide_sidebar else ""
st.markdown(f"""
<style>
.stApp {{
    background: #F7F1E8;
    color: #65432F;
}}
.block-container {{
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}}
h1, h2, h3 {{
    color: #65432F;
}}
.hero {{
    background-image: linear-gradient(90deg, rgba(247,241,232,0.97) 0%, rgba(247,241,232,0.88) 43%, rgba(247,241,232,0.12) 100%), url("https://images.unsplash.com/photo-1607082348824-0a96f2a4b9da?auto=format&fit=crop&w=1800&q=85&v=smartcart-hero-3");
    background-color: #F7F1E8;
    background-size: cover;
    background-position: center right;
    padding: 35px;
    border-radius: 24px;
    margin-bottom: 25px;
}}
.hero h1 {{
    color: #65432F;
    font-size: 42px;
    line-height: 1.2;
    margin-bottom: 12px;
}}
.hero p {{
    color: #65432F;
    font-size: 17px;
}}
.eyebrow {{
    color: #9B7AB5;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}}
.section-subtitle {{
    color: #806958;
    font-size: 15px;
    margin-top: -8px;
    margin-bottom: 22px;
}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 17px;
    border-color: #E5D4BF;
    background: #F7F1E8;
}}
/* Keep all three home discovery cards exactly the same height. */
div[data-testid="stHorizontalBlock"]:has(.st-key-home_browse) {{
    align-items: stretch !important;
}}
div[data-testid="stHorizontalBlock"]:has(.st-key-home_browse) > div[data-testid="stColumn"] {{
    display: flex !important;
    align-items: stretch !important;
}}
div[data-testid="stHorizontalBlock"]:has(.st-key-home_browse) > div[data-testid="stColumn"] > div {{
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    min-width: 0 !important;
    flex: 1 1 0 !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_browse),
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_image),
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_text) {{
    height: 486px !important;
    min-height: 486px !important;
    max-height: 486px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_browse) [data-testid="stVerticalBlock"],
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_image) [data-testid="stVerticalBlock"],
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_text) [data-testid="stVerticalBlock"] {{
    height: 100% !important;
    min-height: 0 !important;
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    box-sizing: border-box !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_browse) [data-testid="stImage"],
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_image) [data-testid="stImage"],
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_text) [data-testid="stImage"] {{
    height: 190px !important;
    min-height: 190px !important;
    max-height: 190px !important;
    overflow: hidden !important;
    flex: 0 0 190px !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_browse) [data-testid="stImage"] img,
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_image) [data-testid="stImage"] img,
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_text) [data-testid="stImage"] img {{
    display: block !important;
    width: 100% !important;
    height: 190px !important;
    min-height: 190px !important;
    object-fit: cover !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_browse) [data-testid="stButton"],
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_image) [data-testid="stButton"],
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_text) [data-testid="stButton"] {{
    margin-top: auto !important;
    flex: 0 0 auto !important;
}}
div.stButton > button {{
    border-radius: 11px;
    min-height: 42px;
    border: 1px solid #E5D4BF;
    transition: all 0.2s ease;
}}
div.stButton > button:hover {{
    border-color: #9B7AB5;
    color: #65432F;
    background: #F7F1E8;
}}
div.stButton > button[kind="primary"] {{
    background: #65432F;
    border: none;
    color: #F7F1E8;
}}
div.stButton > button[kind="primary"]:hover {{
    color: #F7F1E8;
    background: #9B7AB5;
}}
div[data-testid="stPopover"] {{
    position: fixed;
    right: 24px;
    bottom: 24px;
    z-index: 9999;
}}
div[data-testid="stImage"] img {{
    border-radius: 12px;
}}
div[data-testid="stVerticalBlockBorderWrapper"] img {{
    object-fit: contain;
}}
[data-testid="stSidebar"] {{
    background: #F7F1E8;
}}
[data-testid="stSidebar"] h2 {{
    color: #65432F;
}}
{sidebar_css}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD PRODUCT DATA --------------------
@st.cache_data
def load_products():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, low_memory=False)
    text_columns = [
        "product_id", "platform", "category", "product_name",
        "brand", "features", "description", "image_url",
        "product_url"
    ]
    for column in text_columns:
        if column not in df.columns:
            df[column] = ""
    for column in ["price", "rating"]:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )
    for column in [
        "platform", "category", "product_name",
        "brand", "features", "description"
    ]:
        df[column] = df[column].fillna("").astype(str)
    return df

# -------------------- LOAD IMAGES --------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_image(url):
    if not isinstance(url, str):
        return None
    if not url.startswith(("http://", "https://")):
        return None
    try:
        response = requests.get(
            url,
            timeout=3,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()
        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()
        if not content_type.startswith("image/"):
            return None
        image = Image.open(BytesIO(response.content))
        image.load()
        return image.convert("RGB")
    except (
        requests.RequestException,
        UnidentifiedImageError,
        OSError,
        ValueError
    ):
        return None

def show_category_image(category):
    image_url = CATEGORY_IMAGES.get(category)
    if not image_url:
        return
    image = load_image(image_url)
    if image is None:
        st.caption(
            f"{CATEGORY_ICONS.get(category, '🛍️')} {category}"
        )
        return
    image = ImageOps.fit(
        image,
        CATEGORY_IMAGE_SIZE,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5)
    )
    # A subtle cream/beige wash helps the different category photos sit
    # naturally inside the SmartCart palette, while keeping each photo distinct.
    warm_tint = Image.new("RGB", image.size, (247, 241, 232))
    image = Image.blend(image, warm_tint, 0.10)
    st.image(image, use_container_width=True)

def show_custom_image(image_url, alt_text="Shopping inspiration"):
    """Display a distinct remote image using the existing image loader."""
    image = load_image(image_url)
    if image is None:
        st.markdown(
            "<div style='height:180px;border-radius:12px;"
            "background:linear-gradient(135deg,#F7F1E8,#E5D4BF);'></div>",
            unsafe_allow_html=True,
        )
        return
    image = ImageOps.fit(
        image,
        CATEGORY_IMAGE_SIZE,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    st.image(image, use_container_width=True)

def resize_product_image(image):
    if image is None:
        return None
    image = image.convert("RGB")
    fitted = ImageOps.contain(
        image,
        IMAGE_SIZE,
        method=Image.Resampling.LANCZOS
    )
    canvas = Image.new(
        "RGB", IMAGE_SIZE, (247, 241, 232)
    )
    x = (IMAGE_SIZE[0] - fitted.width) // 2
    y = (IMAGE_SIZE[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas

def get_value(row, *keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None:
            try:
                if not pd.isna(value):
                    return value
            except (TypeError, ValueError):
                pass
    return None

def get_image(row):
    image_columns = [
        "image_url", "product_image", "image",
        "image_link", "imageURLHighRes", "imageURL", "images"
    ]
    for key in image_columns:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                urls = ast.literal_eval(value)
                if isinstance(urls, list):
                    for url in urls:
                        image = load_image(url)
                        if image is not None:
                            return image
            except (ValueError, SyntaxError):
                continue
        else:
            image = load_image(value)
            if image is not None:
                return image
    return None

def reset_product_type():
    """Reset the saved selection and its dropdown widget."""
    st.session_state.product_type = "All"
    st.session_state.product_type_widget = "All"

def go_to(page):
    st.session_state.page = page
    st.session_state.menu_open = False
    st.rerun()

# -------------------- PRODUCT FILTERING --------------------
def apply_subcategory(products, product_type):
    if product_type == "All":
        return products
    columns = [
        column for column in [
            "product_name", "features", "description", "brand"
        ]
        if column in products.columns
    ]
    if not columns:
        return products.iloc[0:0]
    searchable_text = (
        products[columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
    terms = SUBCATEGORY_TERMS.get(
        product_type,
        [product_type.lower()]
    )
    pattern = "|".join(
        re.escape(term) for term in terms
    )
    return products[
        searchable_text.str.contains(
            pattern,
            case=False,
            na=False,
            regex=True
        )
    ]

def category_products(products, category):
    return products[
        products["category"].str.strip().str.lower()
        == category.lower()
    ]

def platform_products(products, platform):
    return products[
        products["platform"].str.strip().str.lower()
        == platform.lower()
    ]

def get_available_categories(products):
    available = set(
        products["category"].str.strip().str.lower()
    )
    return [
        category for category in CATEGORY_ICONS
        if category.lower() in available
    ]

def get_available_platforms(products, category):
    category_df = category_products(products, category)
    available = set(
        category_df["platform"].str.strip().str.lower()
    )
    return [
        platform for platform in PLATFORM_ICONS
        if platform.lower() in available
    ]

def get_available_types(products, category, platform):
    """Return the category's product types without repeatedly scanning the dataset."""
    return SUBCATEGORIES.get(category, ["All"])

def get_valid_selection(products):
    categories = get_available_categories(products)
    category = st.session_state.category
    if not category or category not in categories:
        return None
    platforms = get_available_platforms(products, category)
    platform = st.session_state.platform
    if not platform or platform not in platforms:
        return None
    types = get_available_types(products, category, platform)
    product_type = st.session_state.product_type
    if not product_type or product_type not in types:
        return None
    return category, platform, types

# -------------------- CATEGORY CARD --------------------

# -------------------- CATEGORY CARD --------------------
def show_category_card(category, count, key_prefix):
    icon = CATEGORY_ICONS.get(category, "🛍️")

    with st.container(border=True):
        show_category_image(category)

        st.markdown(f"### {icon} {category}")
        st.caption(f"{count:,} products to explore")

        if st.button(
            f"Explore {category} →",
            key=f"{key_prefix}_{category}",
            use_container_width=True
        ):
            st.session_state.category = category
            st.session_state.platform = ""
            reset_product_type()
            go_to("setup")


# -------------------- PRODUCT CARDS --------------------

def show_products(products):
    if not isinstance(products, pd.DataFrame) or products.empty:
        st.info(
            "No matching products found. "
            "Try changing your search or filters."
        )
        return

    st.caption(f"Showing {len(products):,} products")

    cards = []

    for _, product in products.iterrows():
        image = get_image(product)

        name = get_value(
            product, "product_name", "name", "title"
        ) or "Product"

        brand = get_value(product, "brand")
        price = get_value(product, "price")
        rating = get_value(product, "rating")
        platform = get_value(product, "platform")
        url = get_value(
            product, "product_url", "product_link", "url"
        )

        # Cards without images need less vertical space.
        height = 0

        if image is not None:
            height += IMAGE_SIZE[1] + 12

        height += 100
        height += (len(str(name)) // 35) * 20

        if brand:
            height += 28

        if price is not None:
            height += 28

        if rating is not None:
            height += 28

        if platform:
            height += 28

        if isinstance(url, str) and url.startswith(
            ("http://", "https://")
        ):
            height += 50

        cards.append({
            "image": image,
            "name": name,
            "brand": brand,
            "price": price,
            "rating": rating,
            "platform": platform,
            "url": url,
            "height": height
        })

    # Balance the three columns so cards remain organised.
    column_cards = [[], [], []]
    column_heights = [0, 0, 0]

    for card in cards:
        column_index = column_heights.index(
            min(column_heights)
        )

        column_cards[column_index].append(card)
        column_heights[column_index] += card["height"]

    columns = st.columns(3, gap="small")

    for column, cards_in_column in zip(
        columns, column_cards
    ):
        with column:
            for card in cards_in_column:
                with st.container(border=True):

                    # Show an image only when one loaded successfully.
                    # No placeholder is shown for missing images.
                    if card["image"] is not None:
                        st.image(
                            resize_product_image(card["image"]),
                            use_container_width=True
                        )

                    # Product details remain visible with or without an image.
                    st.markdown(f"**{card['name']}**")

                    if card["brand"]:
                        st.caption(
                            f"Brand: {card['brand']}"
                        )

                    if card["price"] is not None:
                        try:
                            st.markdown(
                                f"**₹{float(card['price']):,.2f}**"
                            )
                        except (ValueError, TypeError):
                            st.markdown(
                                f"**₹{card['price']}**"
                            )

                    if card["rating"] is not None:
                        try:
                            st.write(
                                f"⭐ {float(card['rating']):g}"
                            )
                        except (ValueError, TypeError):
                            st.write(
                                f"⭐ {card['rating']}"
                            )

                    if card["platform"]:
                        st.caption(
                            f"Available on {card['platform']}"
                        )

                    url = card["url"]

                    if isinstance(url, str) and url.startswith(
                        ("http://", "https://")
                    ):
                        st.link_button(
                            "View Product ↗",
                            url,
                            use_container_width=True
                        )


# -------------------- LOAD DATA --------------------
df = load_products()
if df.empty:
    st.error(
        "The product dataset could not be loaded. "
        "Check data/processed/master_products.csv."
    )
    st.stop()

# -------------------- SIDEBAR --------------------
if st.session_state.page != "results":
    with st.sidebar:
      
        st.markdown("""
        <div style="
    padding: 8px 0 12px 0;
">
    <div style="
        font-size: 29px;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: #65432F;
        line-height: 1.3;
    ">
        🛒 SmartCart
    </div>
    <div style="
        font-size: 13px;
        color: #806958;
        margin-top: 5px;
        letter-spacing: 0.2px;
    ">
        Your personal shopping companion
    </div>
</div>
""", unsafe_allow_html=True)

        st.divider()
        if st.button("🏠 Home", use_container_width=True):
            go_to("home")
        if st.button("🛍️ Browse Products", use_container_width=True):
            go_to("setup")
        if st.button(
            "📸 Image Recommendations",
            use_container_width=True
        ):
            go_to("image")
        if st.button("🔎 Text Search", use_container_width=True):
            go_to("text")
        st.divider()
        st.caption("Find something you'll love ✨")

# -------------------- FLOATING SHOPPING ASSISTANT --------------------
with st.popover("💬 Chat with SmartCart"):
    st.markdown("### ✨ Your Shopping Assistant")
    st.write("Tell me what you're looking for.")
    with st.form("floating_assistant_form"):
        assistant_query = st.text_input(
            "Shopping request",
            placeholder="e.g. Samsung phone under 20000"
        )
        assistant_submit = st.form_submit_button(
            "Find Products",
            use_container_width=True
        )
    if assistant_submit:
        if not assistant_query.strip():
            st.warning("Please enter a shopping request.")
        else:
            try:
                with st.spinner("Finding products for you..."):
                    results = shopping_assistant(
                        assistant_query,
                        str(DATA_PATH)
                    )
                st.session_state.assistant_results = results
                go_to("assistant_results")
            except Exception as error:
                st.error(f"Search failed: {error}")

# -------------------- HOME PAGE --------------------
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">YOUR PERSONAL SHOPPING SPACE</div>
        <h1>Find your next favourite.</h1>
        <p>
            Discover products, explore your favourite brands,
            and find something you'll love - all in one place.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("Where would you like to start?")
    st.markdown(
        '<div class="section-subtitle">'
        'Choose how you want to discover your next favourite product.'
        '</div>',
        unsafe_allow_html=True
    )
    first, second, third = st.columns(3, gap="medium")
    with first:
        with st.container(border=True):
            show_custom_image(HOME_DISCOVERY_IMAGES["browse"], "Browse products")
            st.markdown("### 🛍️ Browse Products")
            st.write(
                "Explore products by category, platform, and price."
            )
            if st.button(
                "Start Exploring →",
                key="home_browse",
                use_container_width=True,
                type="primary"
            ):
                st.session_state.category = ""
                st.session_state.platform = ""
                reset_product_type()
                go_to("setup")
    with second:
        with st.container(border=True):
            show_custom_image(HOME_DISCOVERY_IMAGES["image_search"], "Image search")
            st.markdown("### 📸 Image Search")
            st.write(
                "Upload a product photo to find similar items."
            )
            if st.button(
                "Find Similar Items →",
                key="home_image",
                use_container_width=True
            ):
                go_to("image")
    with third:
        with st.container(border=True):
            show_custom_image(HOME_DISCOVERY_IMAGES["smart_search"], "Smart search")
            st.markdown("### ✨ Smart Search")
            st.write(
                "Describe what you need and let SmartCart help."
            )
            if st.button(
                "Search Products →",
                key="home_text",
                use_container_width=True
            ):
                go_to("text")
    st.divider()
    st.subheader("Explore Categories")
    st.markdown(
        '<div class="section-subtitle">'
        'A little inspiration for your next shopping find.'
        '</div>',
        unsafe_allow_html=True
    )
    categories = get_available_categories(df)
    category_columns = st.columns(
        min(3, max(1, len(categories))),
        gap="medium"
    )
    for index, category in enumerate(categories):
        with category_columns[index % len(category_columns)]:
            count = len(category_products(df, category))
            show_category_card(
                category,
                count,
                "home_category"
            )
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">SMARTER SHOPPING STARTS HERE</div>
        <h1>Not sure what to choose? ✨</h1>
        <p>
            Tell our shopping assistant what you need,
            or upload a picture to discover similar products.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button(
        "Chat with SmartCart 💬",
        use_container_width=True
    ):
        st.toast(
            "Use the floating Chat with SmartCart button "
            "to start a conversation."
        )
    st.caption("SmartCart · Discover something you'll love.")

# -------------------- SHOPPING SETUP --------------------
elif st.session_state.page == "setup":
    st.markdown(
        '<div class="eyebrow">LET’S GO SHOPPING</div>',
        unsafe_allow_html=True
    )
    st.title("What are you looking for? 🛍️")
    st.markdown(
        '<div class="section-subtitle">'
        'Choose a category, shopping platform, and product type to get started.'
        '</div>',
        unsafe_allow_html=True
    )
    categories = get_available_categories(df)
    if not categories:
        st.warning(
            "No shopping categories are available in the product dataset."
        )
        st.stop()

    # Clear selections that are no longer valid.
    if st.session_state.category not in categories:
        st.session_state.category = ""
        st.session_state.platform = ""
        reset_product_type()

    st.subheader("01 · Choose a category")
    category_columns = st.columns(
        min(3, len(categories)),
        gap="medium"
    )
    for index, option in enumerate(categories):
        with category_columns[index % len(category_columns)]:
            count = len(category_products(df, option))
            with st.container(border=True):
                show_category_image(option)
                st.markdown(
                    f"### {CATEGORY_ICONS[option]} {option}"
                )
                st.caption(f"{count:,} products")
                selected = st.session_state.category == option
                if st.button(
                    "Selected ✓" if selected else "Choose Category",
                    key=f"category_{option}",
                    use_container_width=True,
                    type="primary" if selected else "secondary"
                ):
                    if st.session_state.category != option:
                        st.session_state.category = option
                        st.session_state.platform = ""
                        reset_product_type()
                    # Streamlit automatically reruns after a button click.

    st.divider()
    st.subheader("02 · Choose your shopping platform")
    if not st.session_state.category:
        st.caption(
            "Choose a category above to see the available platforms."
        )
    else:
        available_platforms = get_available_platforms(
            df, st.session_state.category
        )
        if not available_platforms:
            st.info(
                "No platforms currently have products in this category."
            )
        else:
            if st.session_state.platform not in available_platforms:
                st.session_state.platform = ""
                reset_product_type()
            platform_columns = st.columns(
                len(available_platforms),
                gap="medium"
            )
            for column, option in zip(
                platform_columns, available_platforms
            ):
                with column:
                    with st.container(border=True):
                        st.markdown(
                            f"<div style='background:"
                            f"{PLATFORM_COLORS.get(option, '#E5D4BF')};"
                            "border-radius:14px;padding:22px;text-align:center;'>"
                            f"<div style='font-size:42px;'>"
                            f"{PLATFORM_ICONS.get(option, '🛒')}</div>"
                            f"<div style='font-size:20px;font-weight:700;"
                            f"color:#38264E;'>{option}</div></div>",
                            unsafe_allow_html=True
                        )
                        st.write("")
                        selected = (
                            st.session_state.platform == option
                        )
                        if st.button(
                            "Selected ✓" if selected else "Choose Platform",
                            key=f"platform_{option}",
                            use_container_width=True,
                            type="primary" if selected else "secondary"
                        ):
                            if st.session_state.platform != option:
                                st.session_state.platform = option
                                reset_product_type()
                            # Streamlit automatically reruns after a button click.

    # Automatically scroll to the next step after a selection.
    if st.session_state.category:
        target_heading = (
            "03 · Choose a product type"
            if st.session_state.platform
            else "02 · Choose your shopping platform"
        )
        components.html(
            f"""
            <script>
            const targetHeading = {target_heading!r};
            function scrollToNextStep() {{
                const doc = window.parent.document;
                const headings = doc.querySelectorAll("h2, h3");
                for (const heading of headings) {{
                    if (heading.innerText.trim().includes(targetHeading)) {{
                        heading.scrollIntoView({{
                            behavior: "smooth",
                            block: "start"
                        }});
                        return;
                    }}
                }}
            }}
            setTimeout(scrollToNextStep, 200);
            </script>
            """,
            height=1,
        )

    st.divider()
    st.subheader("03 · Choose a product type")
    if (
        not st.session_state.category
        or not st.session_state.platform
    ):
        st.caption(
            "Choose a category and platform first to see product types."
        )
    else:
        available_types = get_available_types(
            df,
            st.session_state.category,
            st.session_state.platform
        )
        # "All" is the default, so choosing a product type is optional.
        type_options = available_types
        # If the category changed, reset to All if the old value is invalid.
        if st.session_state.product_type_widget not in type_options:
            st.session_state.product_type_widget = "All"
        chosen_type = st.selectbox(
            "What would you like to explore?",
            options=type_options,
            key="product_type_widget"
        )
        # Keep the actual selection synchronized with the widget.
        st.session_state.product_type = chosen_type

    if (
        st.session_state.category
        and st.session_state.platform
        and st.session_state.product_type
    ):
        st.markdown(
            f"**Your choices:** "
            f"{CATEGORY_ICONS.get(st.session_state.category, '🛍️')} "
            f"{st.session_state.category}  ·  "
            f"{PLATFORM_ICONS.get(st.session_state.platform, '🛒')} "
            f"{st.session_state.platform}  ·  "
            f"{st.session_state.product_type}"
        )
    can_continue = bool(
        st.session_state.category
        and st.session_state.platform
        and st.session_state.product_type
    )
    if st.button(
        "Show My Products →",
        type="primary",
        use_container_width=True,
        disabled=not can_continue
    ):
        go_to("results")

# -------------------- PRODUCT RESULTS --------------------
elif st.session_state.page == "results":
    if st.button("☰  Menu", key="results_menu"):
        st.session_state.menu_open = not st.session_state.menu_open
        st.rerun()
    if st.session_state.menu_open:
        with st.container(border=True):
            st.markdown("### Where would you like to go?")
            menu_one, menu_two, menu_three, menu_four = st.columns(4)
            with menu_one:
                if st.button("🏠 Home", use_container_width=True):
                    go_to("home")
            with menu_two:
                if st.button("🛍️ Browse", use_container_width=True):
                    go_to("setup")
            with menu_three:
                if st.button("📸 Image Search", use_container_width=True):
                    go_to("image")
            with menu_four:
                if st.button("🔎 Text Search", use_container_width=True):
                    go_to("text")

    current = get_valid_selection(df)
    if current is None:
        st.warning(
            "Please choose a category, platform, and product type first."
        )
        if st.button("Choose Products"):
            go_to("setup")
        st.stop()

    category, platform, _ = current
    st.title(f"{CATEGORY_ICONS.get(category, '🛍️')} {category}")
    st.write(f"Explore {category.lower()} products on {platform}.")
    if st.button("← Change Category or Platform"):
        go_to("setup")

    products = platform_products(
        category_products(df, category),
        platform
    )
    products = apply_subcategory(
        products,
        st.session_state.product_type
    )
    if products.empty:
        st.info(
            "No products match this selection. "
            "Choose another product type or platform."
        )
        st.stop()

    st.subheader("Refine your search")
    with st.expander("🎯 Filters and sorting", expanded=True):
        filter_one, filter_two, filter_three = st.columns(3)
        valid_prices = products["price"].dropna()
        with filter_one:
            if not valid_prices.empty:
                minimum = float(valid_prices.min())
                maximum = float(valid_prices.max())
                if minimum < maximum:
                    price_range = st.slider(
                        "Price range (₹)",
                        min_value=minimum,
                        max_value=maximum,
                        value=(minimum, maximum)
                    )
                else:
                    price_range = (minimum, maximum)
                    st.caption(f"Price: ₹{minimum:,.2f}")
            else:
                price_range = None
                st.caption("No price information available.")
        with filter_two:
            rating_choice = st.selectbox(
                "Minimum rating",
                ["Any Rating", "3+", "4+", "4.5+"]
            )
        with filter_three:
            sort_choice = st.selectbox(
                "Sort products",
                [
                    "Recommended",
                    "Price: Low to High",
                    "Price: High to Low",
                    "Rating: High to Low",
                    "Name: A to Z"
                ]
            )
        brands = sorted(
            brand for brand in products["brand"].unique()
            if brand.strip()
        )
        selected_brands = st.multiselect(
            "Brands",
            brands,
            placeholder="All brands"
        )

    if price_range is not None:
        products = products[
            products["price"].between(
                price_range[0], price_range[1]
            ) | products["price"].isna()
        ]
    if rating_choice != "Any Rating":
        minimum_rating = float(rating_choice.replace("+", ""))
        products = products[
            (products["rating"] >= minimum_rating)
            | products["rating"].isna()
        ]
    if selected_brands:
        products = products[
            products["brand"].isin(selected_brands)
        ]
    if sort_choice == "Price: Low to High":
        products = products.sort_values(
            "price", na_position="last"
        )
    elif sort_choice == "Price: High to Low":
        products = products.sort_values(
            "price", ascending=False, na_position="last"
        )
    elif sort_choice == "Rating: High to Low":
        products = products.sort_values(
            "rating", ascending=False, na_position="last"
        )
    elif sort_choice == "Name: A to Z":
        products = products.sort_values(
            "product_name", na_position="last"
        )

    st.divider()
    st.subheader("Your Products")
    st.write(f"Found {len(products):,} matching products.")
    page_size = 12
    total_pages = max(
        1, (len(products) + page_size - 1) // page_size
    )
    current_page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key="product_page_number"
    )
    start = (current_page - 1) * page_size
    show_products(products.iloc[start:start + page_size])
    st.caption(f"Page {current_page} of {total_pages}")

# -------------------- IMAGE RECOMMENDATIONS --------------------
elif st.session_state.page == "image":
    st.markdown(
        '<div class="eyebrow">SHOP WITH A PICTURE</div>',
        unsafe_allow_html=True
    )
    st.title("📸 Find Similar Products")
    st.write(
        "Upload a product photo to discover visually similar items."
    )
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="recommendation_image"
    )
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Your uploaded image",
                width=260
            )
            if st.button(
                "Find Similar Products ✨",
                type="primary"
            ):
                with st.spinner("Finding similar products..."):
                    results, detected_category = recommend_from_image(
                        image,
                        top_k=10
                    )
                st.session_state.image_results = results
                st.session_state.detected_category = detected_category
                st.rerun()
        except Exception as error:
            st.error(f"Could not process the image: {error}")

    results = st.session_state.image_results
    if isinstance(results, pd.DataFrame) and not results.empty:
        st.success(
            "Detected category: "
            f"{st.session_state.get('detected_category', 'Unknown')}"
        )
        show_products(results)
    elif isinstance(results, list) and results:
        st.success(
            "Detected category: "
            f"{st.session_state.get('detected_category', 'Unknown')}"
        )
        show_products(pd.DataFrame(results))

# -------------------- TEXT SEARCH --------------------
elif st.session_state.page == "text":
    st.markdown(
        '<div class="eyebrow">TELL US WHAT YOU NEED</div>',
        unsafe_allow_html=True
    )
    st.title("🔎 Smart Text Search")
    st.write(
        "Describe the product you want, your preferred brand, or budget."
    )
    with st.form("text_search_form"):
        query = st.text_input(
            "What are you looking for?",
            placeholder="e.g. Samsung phone under 20000"
        )
        submitted = st.form_submit_button(
            "Find Products ✨",
            type="primary",
            use_container_width=True
        )
    if submitted:
        if not query.strip():
            st.warning("Please enter a search.")
        else:
            try:
                with st.spinner("Finding products for you..."):
                    st.session_state.text_results = shopping_assistant(
                        query,
                        str(DATA_PATH)
                    )
            except Exception as error:
                st.error(f"Search failed: {error}")

    results = st.session_state.text_results
    if isinstance(results, pd.DataFrame):
        show_products(results)
    elif isinstance(results, str):
        st.info(results)

# -------------------- SHOPPING ASSISTANT RESULTS --------------------
elif st.session_state.page == "assistant_results":
    st.markdown(
        '<div class="eyebrow">YOUR SHOPPING MATCHES</div>',
        unsafe_allow_html=True
    )
    st.title("💬 Shopping Assistant")
    st.write("Here are the products matching your request.")
    if st.button("← Back to Home"):
        go_to("home")
    results = st.session_state.assistant_results
    if isinstance(results, pd.DataFrame):
        show_products(results)
    elif isinstance(results, str):
        st.info(results)
    else:
        st.info("No matching products were found.")