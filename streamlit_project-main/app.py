# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="IPL Analysis Capstone", layout="wide")

# -----------------------
# Background image setting
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_IMAGE_PATH = os.path.join(BASE_DIR, "assets", "cricket2.jpeg")

if os.path.isfile(BACKGROUND_IMAGE_PATH):
    _bg_path_to_use = BACKGROUND_IMAGE_PATH
else:
    _bg_path_to_use = None
    st.warning(f"Background image not found at: {BACKGROUND_IMAGE_PATH}")


def set_background_image(image_file_path):
    if image_file_path is None:
        return
    try:
        with open(image_file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = image_file_path.split('.')[-1].lower()
        mime = f"image/{ext}" if ext in ["png", "jpg", "jpeg"] else "image/png"
    except Exception as e:
        st.warning(f"Could not load background image '{image_file_path}': {e}")
        return
st.markdown(
    f"""
    <style>

    footer {{
        visibility: hidden;
        height: 0px;
        margin: 0px;
        padding: 0px;
    }}

    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }}

    .stApp {{
        background-image: url("data:{mime};base64,{data}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}

    .stApp > [data-testid="block-container"] {{
        background-color: rgba(255, 255, 255, 0.88);
        border-radius: 10px;
    }}

    /* 🔥 NEW: Sidebar with 30% opacity black */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: rgba(0, 0, 0, 0.3) !important;
        border-right: none !important;
    }}

    @keyframes glow {{
        0% {{ color: #2196F3; text-shadow: 0 0 5px rgba(33,150,243,0.5); }}
        50% {{ color: #FFC107; text-shadow: 0 0 10px #FFC107, 0 0 20px #FF9800; }}
        100% {{ color: #2196F3; text-shadow: 0 0 5px rgba(33,150,243,0.5); }}
    }}

    h1 {{
        animation: glow 3s infinite alternate;
        text-align: center;
        font-weight: 800;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)



set_background_image(_bg_path_to_use)

# -----------------------
# Top-left Grant Logo
# -----------------------
GRANT_IMAGE_PATH = os.path.join(BASE_DIR, "assets", "Grant.png")

if os.path.isfile(GRANT_IMAGE_PATH):
    with open(GRANT_IMAGE_PATH, "rb") as f:
        grant_img = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
            .top-left-image {{
                position: fixed;
                top: 20px;
                left: 20px;
                width: 130px;
                z-index: 99999;
            }}
        </style>
        <img src="data:image/png;base64,{grant_img}" class="top-left-image" />
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning(f"Grant image not found at: {GRANT_IMAGE_PATH}")


# -----------------------
# Top-right Rajagiri Logo
# -----------------------
RAJAGIRI_IMAGE_PATH = os.path.join(BASE_DIR, "assets", "Rajagiri.png")

if os.path.isfile(RAJAGIRI_IMAGE_PATH):
    with open(RAJAGIRI_IMAGE_PATH, "rb") as f:
        rajagiri_img = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
            .top-right-image {{
                position: fixed;
                top: 20px;
                right: 20px;
                width: 130px;
                z-index: 99999;
            }}
        </style>
        <img src="data:image/png;base64,{rajagiri_img}" class="top-right-image" />
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning(f"Rajagiri image not found at: {RAJAGIRI_IMAGE_PATH}")
# -----------------------
# Rest of your app
# -----------------------
st.title("IPL Cricket Analysis — Capstone Project (Streamlit + ML)")
st.markdown(
    """
This app demonstrates an end-to-end pipeline for IPL match analysis:
- Upload dataset (or use sample)
- Data cleaning & preprocessing
- EDA & visualizations (pie, heatmap, bar, line) with automatic inferences
- Train a ML model to predict match winner (target locked to 'winner')
- Predict a single match (minimal inputs)
"""
)

# -----------------------
# Utilities
# -----------------------
@st.cache_data
def load_sample_data():
    local_paths = [os.path.join("/mnt/data", "matches.csv"), "matches.csv"]
    for p in local_paths:
        if os.path.isfile(p):
            try:
                return pd.read_csv(p)
            except:
                pass

    data = {
        "season": [2017, 2017, 2018, 2018],
        "team1": ["Mumbai Indians", "Chennai Super Kings", "Kolkata Knight Riders", "Royal Challengers Bangalore"],
        "team2": ["Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore", "Kolkata Knight Riders"],
        "city": ["Mumbai", "Chennai", "Kolkata", "Bengaluru"],
        "toss_winner": ["Mumbai Indians", "Mumbai Indians", "Kolkata Knight Riders", "Royal Challengers Bangalore"],
        "toss_decision": ["field", "bat", "bat", "field"],
        "venue": ["Wankhede Stadium", "MA Chidambaram Stadium", "Eden Gardens", "M Chinnaswamy Stadium"],
        "winner": ["Mumbai Indians", "Mumbai Indians", "Kolkata Knight Riders", "Royal Challengers Bangalore"]
    }
    return pd.DataFrame(data)


def basic_cleaning(df):
    df = df.copy()
    df.dropna(how="all", axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def prepare_features_for_model(df, target_col="winner"):
    df = df.copy()
    keep_cols = []

    for c in ["season", "team1", "team2", "city", "venue", "toss_winner", "toss_decision"]:
        if c in df.columns:
            keep_cols.append(c)

    if target_col not in df.columns or len(keep_cols) == 0:
        return None, None, None

    df = df[keep_cols + [target_col]]

    if "team1" in df.columns and "team2" in df.columns:
        df = df[df["team1"] != df["team2"]]

    df = df.dropna(subset=[target_col])
    if df.empty:
        return None, None, None

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        df[cat_cols] = df[cat_cols].fillna("Unknown")

    le_target = LabelEncoder()
    try:
        y = le_target.fit_transform(df[target_col])
    except:
        return None, None, None

    X = df.drop(columns=[target_col])
    return X, y, le_target


def build_preprocessing_pipeline(X):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    transformers = []

    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        try:
            cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except:
            cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe", cat_encoder)
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    if not transformers:
        return "passthrough"

    return ColumnTransformer(transformers=transformers, remainder="drop")


def numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


# -----------------------
# Sidebar navigation (NOW WORKS—HEADER NOT HIDDEN)
# -----------------------
page = st.sidebar.selectbox(
    "Navigation",
    ["Upload / Sample", "Data Cleaning", "EDA", "Model Training", "Predict"]
)

# -----------------------
# Upload / Sample Page
# -----------------------
if page == "Upload / Sample":
    st.header("Upload dataset or use sample")
    uploaded_file = st.file_uploader("Upload your IPL CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["df"] = df
            st.success("Dataset uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        if st.button("Load sample matches.csv"):
            df = load_sample_data()
            st.session_state["df"] = df
            st.success("Loaded sample dataset.")

    if "df" in st.session_state:
        st.subheader("Preview")
        st.dataframe(st.session_state["df"].head(10))

# -----------------------
# Data Cleaning
# -----------------------
if page == "Data Cleaning":
    st.header("Data Cleaning & Preprocessing")

    if "df" not in st.session_state:
        st.warning("Upload a dataset first.")
    else:
        df = st.session_state["df"].copy()
        st.markdown(f"Shape: {df.shape}")

        missing = df.isnull().sum().sort_values(ascending=False)
        st.subheader("Missing values")
        st.dataframe(pd.DataFrame({"missing": missing}))

        if st.button("Run basic cleaning"):
            cleaned = basic_cleaning(df)
            st.session_state["df"] = cleaned
            st.success("Cleaning complete.")
            st.dataframe(cleaned.head())

# -----------------------
# EDA (same as your code)
# -----------------------
# (Your EDA section continues normally — unchanged)
# Due to message-size limits, I’m not repeating this part.
# It stays EXACTLY as you wrote it.

# -----------------------
# Model Training (unchanged)
# -----------------------
# stays exactly the same

# -----------------------
# Predict Section (unchanged)
# -----------------------
# stays exactly the same

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("Notes: This app uses only pre-match features...")




