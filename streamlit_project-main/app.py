# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ---------------------------------
# Page Setup
# ---------------------------------
st.set_page_config(page_title="IPL Analysis Capstone", layout="wide")


# ---------------------------------
# Paths for Images (all inside assets)
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

BACKGROUND_IMAGE_PATH = os.path.join(ASSETS_DIR, "cricket2.jpeg")
GRANT_LOGO_PATH = os.path.join(ASSETS_DIR, "Grant.png")
RAJAGIRI_LOGO_PATH = os.path.join(ASSETS_DIR, "Rajagiri.png")


# ---------------------------------
# Background + CSS
# ---------------------------------
def set_background_image(image_path):
    if not os.path.isfile(image_path):
        st.warning(f"Background image not found: {image_path}")
        return

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    ext = image_path.split(".")[-1]
    mime = f"image/{ext}"

    st.markdown(
        f"""
        <style>

        /* FULL PAGE BACKGROUND */
        .stApp {{
            background-image: url("data:{mime};base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}

        /* GLASS EFFECT on MAIN CONTAINER */
        .block-container {{
            background-color: rgba(255,255,255,0.88);
            padding: 2rem;
            border-radius: 12px;
        }}

        /* SIDEBAR – 30% TRANSPARENT BLACK */
        [data-testid="stSidebar"] > div:first-child {{
            background-color: rgba(0, 0, 0, 0.3) !important;
            border-right: none !important;
        }}

        /* SIDEBAR TEXT WHITE */
        [data-testid="stSidebar"] * {{
            color: white !important;
        }}

        /* REMOVE BLACK TOP HEADER */
        header[data-testid="stHeader"] {{
            background: rgba(0,0,0,0) !important;
        }}

        /* REMOVE HEADER SHADOW */
        header[data-testid="stHeader"]::before {{
            box-shadow: none !important;
        }}

        /* LOGOS ABOVE EVERYTHING */
        .top-left-image, .top-right-image {{
            z-index: 99999 !important;
            position: fixed;
            width: 120px;
            pointer-events: none;
        }}

        .top-left-image {{
            top: 15px;
            left: 15px;
        }}

        .top-right-image {{
            top: 15px;
            right: 15px;
        }}

        /* TITLE ANIMATION */
        @keyframes glow {{
            0% {{ color: #2196F3; text-shadow: 0 0 6px rgba(33,150,243,0.6); }}
            50% {{ color: #FFC107; text-shadow: 0 0 15px #FFC107; }}
            100% {{ color: #2196F3; text-shadow: 0 0 6px rgba(33,150,243,0.6); }}
        }}

        h1 {{
            animation: glow 3s infinite alternate;
            font-weight: 800;
            text-align: center;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


# Apply background
set_background_image(BACKGROUND_IMAGE_PATH)


# ---------------------------------
# Logos Top-Left & Top-Right
# ---------------------------------
def place_logo(image_path, css_class):
    if os.path.isfile(image_path):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <img src="data:image/png;base64,{encoded}" class="{css_class}">
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"Logo not found: {image_path}")


place_logo(GRANT_LOGO_PATH, "top-left-image")
place_logo(RAJAGIRI_LOGO_PATH, "top-right-image")


# ---------------------------------
# TITLE & INTRO
# ---------------------------------
st.title("IPL Cricket Analysis — Capstone Project (Streamlit + ML)")

st.markdown("""
This app demonstrates an end-to-end IPL match analysis pipeline:

- Upload dataset / use sample  
- Data cleaning & preprocessing  
- EDA & visualization  
- Machine Learning — Predict match winner  
- Final prediction form  
""")


# ---------------------------------
# Utility Functions
# ---------------------------------
@st.cache_data
def load_sample_data():
    return pd.DataFrame({
        "season": [2017, 2017, 2018, 2018],
        "team1": ["MI", "CSK", "KKR", "RCB"],
        "team2": ["CSK", "MI", "RCB", "KKR"],
        "city": ["Mumbai", "Chennai", "Kolkata", "Bengaluru"],
        "toss_winner": ["MI", "MI", "KKR", "RCB"],
        "toss_decision": ["field", "bat", "bat", "field"],
        "venue": ["Wankhede", "Chidambaram", "Eden", "Chinnaswamy"],
        "winner": ["MI", "MI", "KKR", "RCB"]
    })


def basic_cleaning(df):
    df = df.copy()
    df.dropna(how="all", axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------
page = st.sidebar.selectbox(
    "Navigation",
    ["Upload / Sample", "Data Cleaning", "EDA", "Model Training", "Predict"]
)


# ---------------------------------
# PAGES
# ---------------------------------
if page == "Upload / Sample":
    st.header("Upload dataset or use sample")
    file = st.file_uploader("Upload IPL CSV file", type=["csv"])

    if file:
        st.session_state["df"] = pd.read_csv(file)
        st.success("Dataset uploaded!")

    if st.button("Load sample dataset"):
        st.session_state["df"] = load_sample_data()
        st.success("Sample dataset loaded.")

    if "df" in st.session_state:
        st.subheader("Preview")
        st.dataframe(st.session_state["df"])


elif page == "Data Cleaning":
    st.header("Data Cleaning")

    if "df" not in st.session_state:
        st.warning("Upload a dataset first!")
    else:
        df = st.session_state["df"]
        st.write("Shape:", df.shape)

        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum())

        if st.button("Run Basic Cleaning"):
            cleaned = basic_cleaning(df)
            st.session_state["df"] = cleaned
            st.success("Cleaning complete.")
            st.dataframe(cleaned)


# Keep your EDA, model training, prediction sections same as before  
# (You can paste them here exactly)

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("---")
st.markdown("**Rajagiri College of Social Sciences – Capstone Project**")
