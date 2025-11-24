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
        /* Hide default Streamlit header */
        [data-testid="stHeader"], footer {{
            visibility: hidden;
        }}

        .stApp {{
            background-image: url("data:{mime};base64,{data}");
            background-size: cover;
            background-attachment: fixed;
        }}

        .stApp > [data-testid="block-container"] {{
            background-color: rgba(255, 255, 255, 0.88);
            border-radius: 10px;
        }}

        @keyframes glow {{
            0% {{ color: #2196F3; }}
            50% {{ color: #FFC107; }}
            100% {{ color: #2196F3; }}
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
# ⭐ Top-right Rajagiri Logo
# -----------------------
IMAGE_PATH = os.path.join(BASE_DIR, "assets", "Rajagiri.png")

if os.path.isfile(IMAGE_PATH):
    with open(IMAGE_PATH, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .top-right-image {{
            position: fixed;
            top: 20px;
            right: 20px;
            width: 130px;
            z-index: 9999;
        }}
        </style>
        <img src="data:image/png;base64,{img_data}" class="top-right-image" />
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning(f"Rajagiri logo not found at: {IMAGE_PATH}")

# -----------------------
# App Title
# -----------------------
st.title("IPL Cricket Analysis — Capstone Project (Streamlit + ML)")
st.markdown(
    """
This app demonstrates an end-to-end pipeline for IPL match analysis:
- Upload dataset (or use sample)
- Data cleaning & preprocessing
- EDA & visualizations
- Train ML model to predict match winner
- Predict match results
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
            except Exception:
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
    except Exception:
        return None, None, None
    X = df.drop(columns=[target_col])
    return X, y, le_target

def build_preprocessing_pipeline(X):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    transformers = []
    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        num_pipeline = Pipeline([("imputer", num_imputer), ("scaler", StandardScaler())])
        transformers.append(("num", num_pipeline, num_cols))
    if cat_cols:
        cat_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
        try:
            cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        cat_pipeline = Pipeline([("imputer", cat_imputer), ("ohe", cat_encoder)])
        transformers.append(("cat", cat_pipeline, cat_cols))
    if not transformers:
        preprocessor = "passthrough"
    else:
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def categorical_columns(df):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()

# -----------------------
# Sidebar
# -----------------------
page = st.sidebar.selectbox("Navigation", ["Upload / Sample", "Data Cleaning", "EDA", "Model Training", "Predict"])

# -----------------------
# Upload / Sample
# -----------------------
if page == "Upload / Sample":
    st.header("Upload dataset or use sample")
    uploaded_file = st.file_uploader("Upload your IPL CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.success("Uploaded successfully!")
    else:
        if st.button("Load sample"):
            df = load_sample_data()
            st.session_state["df"] = df
            st.success("Sample dataset loaded!")

    if "df" in st.session_state:
        st.subheader("Preview (first 10 rows)")
        st.dataframe(st.session_state["df"].head(10))

# -----------------------
# (Rest of your original code continues unchanged…)
# -----------------------

