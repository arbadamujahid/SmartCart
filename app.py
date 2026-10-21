from pathlib import Path
from io import BytesIO
import ast
import re
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps, UnidentifiedImageError
from image_recommendation.integrated_recommender import recommend_from_image
from scripts.ai_shopping_assistant import shopping_assistant

# -------------------- PAGE SETTINGS --------------------

st.set_page_config(
    page_title="SmartCart",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    "Fashion": "https://images.unsplash.com/photo-1483985988355-763728e1935b?auto=format&fit=crop&w=1200&q=90",
    "Beauty": "https://images.unsplash.com/photo-1596462502278-27bfdc403348?auto=format&fit=crop&w=1200&q=90",
    "Mobiles": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?auto=format&fit=crop&w=1200&q=90",
    "Laptops": "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?auto=format&fit=crop&w=1200&q=90",
    "Shoes": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=1200&q=90",
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
PLATFORM_IMAGES = {
    # Exact image links supplied by the user, in the order they identified them.
    "Amazon": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT9X2zD4qHOAhe0Yh7QYBHY5relSIx0AoH3Z8upkD0IGg&s=10",
    "Flipkart": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTS6FIH5T38IhsPtLhFa2XklJCKaJh0mMH67XoVUoG67w&s=10",
    "Myntra": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQOxHzKiNan1OpfFmezt4TsNnygsutpSDbTrtSQvWx8_g&s=10",
    "Nykaa": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSWkl3IJ0zrV3LzFQyNhhKm_32hbRUqzgbYKYs5whiuFg&s=10",
}
SUBCATEGORIES = {
    "Fashion": [
        "All", "Shirts", "T-Shirts", "Tops", "Shorts", "Jeans",
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
    "Shorts": ["shorts", "short"],
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
/* SmartCart AI: compact sidebar button + softly tinted popover */
[data-testid="stSidebar"] [data-testid="stPopover"] {{
    width: 100% !important;
    margin: 2px 0 0 0 !important;
}}
/* Streamlit may wrap the popover trigger in an extra div, so target descendants. */
[data-testid="stSidebar"] [data-testid="stPopover"] button {{
    width: 100% !important;
    min-height: 44px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    padding: 0.62rem 0.85rem !important;
    border: 1px solid #80639A !important;
    border-radius: 10px !important;
    background: #80639A !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 7px rgba(101, 67, 47, 0.10) !important;
    transition: background 160ms ease, box-shadow 160ms ease, transform 160ms ease !important;
}}
[data-testid="stSidebar"] [data-testid="stPopover"] button:hover {{
    background: #6F527F !important;
    border-color: #6F527F !important;
    box-shadow: 0 4px 10px rgba(101, 67, 47, 0.15) !important;
    color: #FFFFFF !important;
    transform: translateY(-1px) !important;
}}
[data-testid="stSidebar"] [data-testid="stPopover"] button:focus-visible {{
    outline: 2px solid #9B7AB5 !important;
    outline-offset: 2px !important;
}}
/* Streamlit can render the popover panel outside the sidebar DOM. */
[data-testid="stPopoverBody"] {{
    background: #F5EFF9 !important;
    border: 1px solid #D7C4E3 !important;
    border-radius: 16px !important;
    box-shadow: 0 12px 30px rgba(55, 35, 24, 0.17) !important;
    padding: 0.9rem !important;
}}
[data-testid="stPopoverBody"] h3 {{
    color: #65432F !important;
    letter-spacing: -0.25px !important;
}}
[data-testid="stPopoverBody"] [data-testid="stCaptionContainer"] {{
    color: #806958 !important;
}}
[data-testid="stPopoverBody"] [data-testid="stTextInput"] input {{
    border: 1px solid #D7C4E3 !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
    color: #65432F !important;
}}
[data-testid="stPopoverBody"] [data-testid="stTextInput"] input:focus {{
    border-color: #80639A !important;
    box-shadow: 0 0 0 2px rgba(128, 99, 154, 0.14) !important;
}}
[data-testid="stPopoverBody"] [data-testid="stFormSubmitButton"] button {{
    width: 100% !important;
    min-height: 41px !important;
    border: none !important;
    border-radius: 10px !important;
    background: #65432F !important;
    color: #FFF9F2 !important;
    font-weight: 650 !important;
}}
[data-testid="stPopoverBody"] [data-testid="stFormSubmitButton"] button:hover {{
    background: #80639A !important;
    color: #FFFFFF !important;
}}
div[data-testid="stImage"] img {{
    border-radius: 12px;
}}
/* Image-containing cards use a soft white surface against the cream page. */
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stImage"]) {{
    background: #FFFFFF !important;
    border-color: #EDE4DA !important;
    border-radius: 18px !important;
    box-shadow: 0 3px 12px rgba(101, 67, 47, 0.045) !important;
}}

div[data-testid="stImage"] {{
    overflow: hidden !important;
    border-radius: 13px !important;
}}

div[data-testid="stImage"] img {{
    display: block !important;
    object-fit: contain !important;
    border-radius: 13px !important;
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
</style>
""", unsafe_allow_html=True)

# Category and platform buttons become full-image tiles with a readable overlay.
# Streamlit's keyed button wrappers let each tile keep native click behavior.
_tile_css = ["<style>"]
for _category, _url in CATEGORY_IMAGES.items():
    for _key in (f"home_category_{_category}", f"category_{_category}"):
        _tile_css.append(f"""
        .st-key-{_key} button {{
            min-height: 188px !important;
            height: 188px !important;
            border: 0 !important;
            border-radius: 16px !important;
            padding: 0 0 17px 17px !important;
            display: flex !important;
            align-items: flex-end !important;
            justify-content: flex-start !important;
            text-align: left !important;
            color: #FFFFFF !important;
            font-size: 19px !important;
            font-weight: 800 !important;
            letter-spacing: 1.1px !important;
            text-shadow: 0 2px 8px rgba(0,0,0,.55) !important;
            background-image: linear-gradient(0deg, rgba(35,24,18,.76) 0%, rgba(35,24,18,.22) 54%, rgba(35,24,18,.02) 100%), url('{_url}') !important;
            background-size: cover !important;
            background-position: center !important;
            box-shadow: 0 5px 15px rgba(101,67,47,.10) !important;
            transition: transform .18s ease, box-shadow .18s ease !important;
        }}
        .st-key-{_key} button:hover {{
            border: 0 !important;
            color: #FFFFFF !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 9px 20px rgba(101,67,47,.18) !important;
            filter: saturate(1.06) !important;
        }}
        .st-key-{_key} button p {{ color: #FFFFFF !important; }}
        """)
for _platform, _url in PLATFORM_IMAGES.items():
    _key = f"platform_{_platform}"
    _tile_css.append(f"""
    .st-key-{_key} button {{
        min-height: 170px !important;
        height: 170px !important;
        border: 1px solid rgba(101,67,47,.12) !important;
        border-radius: 16px !important;
        padding: 0 0 16px 15px !important;
        display: flex !important;
        align-items: flex-end !important;
        justify-content: flex-start !important;
        text-align: left !important;
        color: #FFFFFF !important;
        font-size: 18px !important;
        font-weight: 800 !important;
        letter-spacing: .7px !important;
        text-shadow: 0 1px 3px rgba(0,0,0,.55) !important;
        background-color: {PLATFORM_COLORS[_platform]} !important;
        /* Fill the entire clickable tile with the exact marketplace image. */
        background-image: linear-gradient(to top, rgba(25,18,14,.62), rgba(25,18,14,0) 58%), url('{_url}') !important;
        background-size: 100% 100%, cover !important;
        background-repeat: no-repeat, no-repeat !important;
        background-position: center, center !important;
        box-shadow: 0 5px 15px rgba(101,67,47,.10) !important;
        transition: transform .18s ease, box-shadow .18s ease !important;
    }}
    .st-key-{_key} button:hover {{
        border: 0 !important;
        color: #FFFFFF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 9px 20px rgba(101,67,47,.18) !important;
    }}
    .st-key-{_key} button p {{ color: #FFFFFF !important; }}
    """)
_tile_css.append("</style>")
st.markdown("\n".join(_tile_css), unsafe_allow_html=True)

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

def resize_product_image(image, background=(255, 255, 255)):
    if image is None:
        return None
    image = image.convert("RGB")
    fitted = ImageOps.contain(
        image,
        IMAGE_SIZE,
        method=Image.Resampling.LANCZOS
    )
    # Other pages retain their default; results cards request a white image stage.
    canvas = Image.new("RGB", IMAGE_SIZE, background)
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
    # Scroll to the top only when entering results from another page.
    # Avoid jumping upward when filters or pagination trigger a rerun.
    if page == "results" and st.session_state.get("page") != "results":
        st.session_state.scroll_results_to_top = True
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
def show_category_card(category, count, key_prefix):
    """Render a vintage-inspired image tile that acts as the category button."""
    key = f"{key_prefix}_{category}"
    if st.button(
        category.upper(),
        key=key,
        use_container_width=True,
        help=f"Explore {category} products",
    ):
        st.session_state.category = category
        st.session_state.platform = ""
        reset_product_type()
        go_to("setup")
    st.caption(f"{count:,} products")



# -------------------- PRODUCT CARDS --------------------

def show_products(products, show_count=True):
    if not isinstance(products, pd.DataFrame) or products.empty:
        st.info(
            "No matching products found. "
            "Try changing your search or filters."
        )
        return

    if show_count:
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
            "card_id": len(cards),
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
                with st.container(border=True, key=f"product_card_{card['card_id']}"):

                    if card["image"] is not None:
                        st.image(
                            resize_product_image(card["image"], background=(255, 255, 255)),
                            use_container_width=True
                        )

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

# -------------------- PAGE-SPECIFIC SIDEBAR --------------------
if st.session_state.page == "results":
    # Results page: the sidebar is for product filtering only.
    with st.sidebar:
        st.markdown("### Filter products")
        st.caption("Choose a category and product type.")
        st.divider()

        st.markdown("**Category**")
        sidebar_categories = get_available_categories(df)
        for sidebar_category in sidebar_categories:
            is_selected = st.session_state.category == sidebar_category
            if st.button(
                f"{CATEGORY_ICONS.get(sidebar_category, '🛍️')}  {sidebar_category}"
                + ("  ✓" if is_selected else ""),
                key=f"results_sidebar_category_{sidebar_category}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.category = sidebar_category
                available_platforms = get_available_platforms(df, sidebar_category)
                st.session_state.platform = (
                    available_platforms[0] if available_platforms else ""
                )
                reset_product_type()
                st.rerun()

        st.divider()
        st.markdown("**Product type**")
        sidebar_platforms = (
            get_available_platforms(df, st.session_state.category)
            if st.session_state.category else []
        )
        if sidebar_platforms:
            if st.session_state.platform not in sidebar_platforms:
                st.session_state.platform = sidebar_platforms[0]
            sidebar_types = get_available_types(
                df, st.session_state.category, st.session_state.platform
            )
            if not sidebar_types:
                sidebar_types = [PRODUCT_TYPE_PLACEHOLDER]
            if st.session_state.product_type not in sidebar_types:
                reset_product_type()
            current_type = (
                st.session_state.product_type
                if st.session_state.product_type in sidebar_types
                else sidebar_types[0]
            )
            selected_sidebar_type = st.selectbox(
                "Choose a type",
                sidebar_types,
                index=sidebar_types.index(current_type),
                key="results_sidebar_product_type",
                label_visibility="collapsed"
            )
            if selected_sidebar_type != st.session_state.product_type:
                st.session_state.product_type = selected_sidebar_type
                st.rerun()
        else:
            st.caption("Choose a category with available products.")
else:
    with st.sidebar:
        st.markdown("""
        <div style="padding: 8px 0 12px 0;">
            <div style="font-size: 29px; font-weight: 800; letter-spacing: -0.8px;
                        color: #65432F; line-height: 1.3;">
                🛒 SmartCart
            </div>
            <div style="font-size: 13px; color: #806958; margin-top: 5px;
                        letter-spacing: 0.2px;">
                Your personal shopping companion
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        if st.button("🏠 Home", key="nav_home", use_container_width=True):
            go_to("home")
        if st.button("🛍️ Browse Products", key="nav_browse", use_container_width=True):
            go_to("setup")
        if st.button("📸 Image Recommendations", key="nav_image", use_container_width=True):
            go_to("image")
        if st.button("🔎 Text Search", key="nav_text", use_container_width=True):
            go_to("text")
        st.divider()
        # Purple assistant button opens a compact pop-up, keeping the sidebar uncluttered.
        with st.popover("💬 Ask AI Assistant", use_container_width=True):
            st.markdown("### Your shopping assistant")
            st.caption("Tell me what you need and I’ll help you find it.")
            with st.form("sidebar_assistant_form", clear_on_submit=False):
                sidebar_assistant_query = st.text_input(
                    "What are you looking for?",
                    placeholder="e.g. phone under ₹20,000",
                    label_visibility="collapsed"
                )
                sidebar_assistant_submit = st.form_submit_button(
                    "Ask SmartCart", use_container_width=True
                )
            if sidebar_assistant_submit:
                if not sidebar_assistant_query.strip():
                    st.warning("Please enter what you're looking for.")
                else:
                    try:
                        assistant_response = shopping_assistant(
                            sidebar_assistant_query, str(DATA_PATH)
                        )
                        st.session_state["sidebar_assistant_response"] = assistant_response
                    except Exception as exc:
                        st.session_state["sidebar_assistant_response"] = (
                            f"Sorry, the assistant couldn't process that request: {exc}"
                        )
            if st.session_state.get("sidebar_assistant_response") is not None:
                st.divider()
                st.markdown("**Assistant response**")
                sidebar_response = st.session_state["sidebar_assistant_response"]
                if isinstance(sidebar_response, pd.DataFrame):
                    st.caption(f"Found {len(sidebar_response)} matching products.")
                    if not sidebar_response.empty:
                        st.dataframe(sidebar_response.head(5), use_container_width=True)
                    else:
                        st.caption("No matching products found. Try different keywords.")
                else:
                    st.write(str(sidebar_response))
        st.divider()
        st.caption("Find something you'll love ✨")

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
            selected = st.session_state.category == option
            if st.button(
                option.upper() + ("  ✓" if selected else ""),
                key=f"category_{option}",
                use_container_width=True,
                type="primary" if selected else "secondary",
                help=f"Choose {option}",
            ):
                if st.session_state.category != option:
                    st.session_state.category = option
                    st.session_state.platform = ""
                    reset_product_type()

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
            for column, option in zip(platform_columns, available_platforms):
                with column:
                    selected = st.session_state.platform == option
                    if st.button(
                        option.upper() + ("  ✓" if selected else ""),
                        key=f"platform_{option}",
                        use_container_width=True,
                        type="primary" if selected else "secondary",
                        help=f"Shop on {option}",
                    ):
                        if st.session_state.platform != option:
                            st.session_state.platform = option
                            reset_product_type()
        
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
        type_options = available_types
        if st.session_state.product_type not in type_options:
            reset_product_type()
        st.caption("Choose a specific type, or keep All products selected.")
        selected_type = st.selectbox(
            "Product type",
            type_options,
            index=type_options.index(st.session_state.product_type),
            key="product_type_widget"
        )
        st.session_state.product_type = selected_type

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
    # Results-page styling, kept separate from the homepage.
    st.markdown("""
    <style>
    /* Results header spacing only. */
    .main .block-container { padding-top: 1rem !important; }
    .st-key-results_header { margin-top: 0 !important; }
    .st-key-results_header h1 { margin: 0 0 .15rem 0 !important; padding: 0 !important; }
    .st-key-results_header [data-testid="stMarkdownContainer"] p { margin-bottom: .25rem !important; }

    /* Warm beige-brown filter panel: blends with SmartCart's cream/brown palette. */
    [data-testid="stExpander"] {
        background: #E5D4BF !important;
        background-color: #E5D4BF !important;
        border: 1px solid #CDB398 !important;
        border-radius: 15px !important;
        margin: .2rem 0 .75rem 0 !important;
        box-shadow: 0 3px 10px rgba(101,67,47,.08) !important;
        overflow: hidden !important;
    }
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    [data-testid="stExpander"] [data-testid="stVerticalBlock"],
    [data-testid="stExpander"] [data-testid="stHorizontalBlock"] {
        background: transparent !important;
        background-color: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary svg,
    [data-testid="stExpander"] label,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stExpander"] [data-testid="stCaptionContainer"] {
        color: #65432F !important;
    }
    [data-testid="stExpander"] summary { font-weight: 750 !important; padding: .75rem .95rem !important; }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: .5rem .9rem .9rem !important; }
    [data-testid="stExpander"] input,
    [data-testid="stExpander"] [data-baseweb="select"] > div,
    [data-testid="stExpander"] [data-baseweb="tag"] {
        background: #FFFCF8 !important;
        background-color: #FFFCF8 !important;
        color: #65432F !important;
        border-color: #CDB398 !important;
    }
    [data-testid="stExpander"] [data-testid="stSlider"] {
        color: #65432F !important;
    }

    /* Streamlit versions place the container key on different wrapper levels.
       Style both the keyed wrapper and its bordered child to avoid selector misses. */
    div[class*="st-key-product_card_"] {
        background: #EFE2D2 !important;
        background-color: #EFE2D2 !important;
        border-radius: 18px !important;
    }
    div[class*="st-key-product_card_"] [data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-product_card_"],
    div[data-testid="stVerticalBlockBorderWrapper"]:has(div[class*="st-key-product_card_"]) {
        background: #EFE2D2 !important;
        background-color: #EFE2D2 !important;
        border: 1px solid #D7BFA7 !important;
        border-radius: 18px !important;
        padding: .65rem !important;
        box-shadow: 0 3px 10px rgba(101,67,47,.08) !important;
        transition: background-color .18s ease, box-shadow .18s ease !important;
    }
    div[class*="st-key-product_card_"]:hover,
    div[class*="st-key-product_card_"] [data-testid="stVerticalBlockBorderWrapper"]:hover,
    div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-product_card_"]:hover {
        background: #E7D6C3 !important;
        background-color: #E7D6C3 !important;
        box-shadow: 0 6px 15px rgba(101,67,47,.12) !important;
    }
    /* Pure white image area; no lavender canvas or coloured image padding. */
    div[class*="st-key-product_card_"] [data-testid="stImage"],
    div[class*="st-key-product_card_"] [data-testid="stImage"] > div,
    div[class*="st-key-product_card_"] [data-testid="stImage"] img {
        background: #FFFFFF !important;
        background-color: #65432F !important;
        border: 0 !important;
        border-radius: 12px !important;
        padding: 0 !important;
        object-fit: contain !important;
    }
    div[class*="st-key-product_card_"] [data-testid="stImage"] {
        margin: 0 0 .4rem 0 !important;
        overflow: hidden !important;
        box-shadow: none !important;
    }
    div[class*="st-key-product_card_"] [data-testid="stCaptionContainer"] { color: #65432F !important; }
    </style>
    """, unsafe_allow_html=True)
    if st.session_state.pop("scroll_results_to_top", False):
        components.html("""
        <script>
        (() => {
          const scrollTop = () => {
            try {
              const doc = window.parent.document;
              const main = doc.querySelector('section.main');
              const stMain = doc.querySelector('[data-testid="stMain"]');
              const appView = doc.querySelector('[data-testid="stAppViewContainer"]');
              const block = doc.querySelector('.main .block-container');
              window.parent.scrollTo({top: 0, left: 0, behavior: 'auto'});
              if (main) main.scrollTo({top: 0, left: 0, behavior: 'auto'});
              if (stMain) stMain.scrollTo({top: 0, left: 0, behavior: 'auto'});
              if (appView) appView.scrollTo({top: 0, left: 0, behavior: 'auto'});
              if (block) block.scrollTop = 0;
            } catch (e) {}
          };
          scrollTop();
          requestAnimationFrame(scrollTop);
          setTimeout(scrollTop, 150);
          setTimeout(scrollTop, 500);
          setTimeout(scrollTop, 900);
        })();
        </script>
        """, height=1, scrolling=False)
    current = get_valid_selection(df)
    if current is None:
        st.warning(
            "Please choose a category, platform, and product type first."
        )
        if st.button("Choose Products"):
            go_to("setup")
        st.stop()

    category, platform, _ = current
    with st.container(key="results_header"):
        header_title, header_action = st.columns([4, 1.65], vertical_alignment="center")
        with header_title:
            st.title(f"{CATEGORY_ICONS.get(category, '🛍️')} {category}")
            st.caption(f"Explore {category.lower()} products on {platform}.")
        with header_action:
            if st.button("← Change selection", key="results_change_selection", use_container_width=True):
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

    # One styled expander only; closed by default to save space.
    with st.expander("🎯 Filters & sorting", expanded=False):
        filter_one, filter_two, filter_three, filter_four = st.columns(
            [1.5, 1, 1.2, 1.3],
            gap="medium"
        )
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
                st.caption("Price data unavailable")
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
        with filter_four:
            selected_brands = st.multiselect(
                "Brand",
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

    page_size = 12
    total_pages = max(
        1, (len(products) + page_size - 1) // page_size
    )
    current_page = st.session_state.get("product_page_number", 1)
    current_page = min(max(int(current_page), 1), total_pages)
    start = (current_page - 1) * page_size
    page_products = products.iloc[start:start + page_size]
    st.caption(f"{len(products):,} products · showing {start + 1}–{start + len(page_products)}")
    with st.container(key="results_products_panel"):
        show_products(page_products, show_count=False)

    st.divider()
    page_left, page_center, page_right = st.columns([1, 2, 1])
    with page_left:
        if st.button(
            "← Previous",
            disabled=current_page <= 1,
            use_container_width=True,
            key="products_prev_page"
        ):
            st.session_state.product_page_number = current_page - 1
            st.rerun()
    with page_center:
        st.markdown(
            f"<div style='text-align:center; padding-top:8px;'>"
            f"Page {current_page} of {total_pages}</div>",
            unsafe_allow_html=True
        )
    with page_right:
        if st.button(
            "Next →",
            disabled=current_page >= total_pages,
            use_container_width=True,
            key="products_next_page"
        ):
            st.session_state.product_page_number = current_page + 1
            st.rerun()

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