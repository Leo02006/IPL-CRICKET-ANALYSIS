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
# Image Paths (ALL inside assets)
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE_DIR, "assets")

BACKGROUND = os.path.join(ASSETS, "cricket2.jpeg")

# Grant Thornton logo — user uploaded file
GRANT = "/mnt/data/6de89bdc-5c43-4dd0-973a-8dfa5d79a1bc.png"

# Rajagiri logo from assets folder
RAJAGIRI = os.path.join(ASSETS, "Rajagiri.png")


# ---------------------------------
# Background + CSS
# ---------------------------------
def set_bg(image):
    if not os.path.isfile(image):
        st.error(f"Image not found: {image}")
        return

    with open(image, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    ext = image.split(".")[-1]
    mime = f"image/{ext}"

    st.markdown(
        f"""
        <style>

        /* FULL BACKGROUND */
        .stApp {{
            background-image: url("data:{mime};base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* REMOVE MAIN WHITE GLASS */
        .block-container {{
            background-color: transparent !important;
            padding: 2rem;
        }}

        /* SIDEBAR - 30% BLACK */
        [data-testid="stSidebar"] > div:first-child {{
            background: rgba(0,0,0,0.3) !important;
            border-right: none !important;
        }}

        /* SIDEBAR TEXT WHITE */
        [data-testid="stSidebar"] * {{
            color: white !important;
        }}

        /* TRANSPARENT TOP HEADER */
        header[data-testid="stHeader"] {{
            background: transparent !important;
        }}

        header[data-testid="stHeader"]::before {{
            box-shadow: none !important;
        }}

        /* LOGOS — visible & not behind sidebar */
        .top-left, .top-right {{
            position: fixed;
            top: 24px;
            padding: 8px;
            background: rgba(0,0,0,0.55);
            border-radius: 12px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.35);
            z-index: 2147483647;
            width: 140px;
            height: auto;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        /* Move Grant logo to the RIGHT of sidebar */
        .top-left {{
            left: 260px;
        }}

        /* Rajagiri stays top-right */
        .top-right {{
            right: 40px;
        }}

        .top-left img, .top-right img {{
            width: 100%;
            height: auto;
            display: block;
        }}

        /* TITLE ANIMATION */
        @keyframes glow {{
            0% {{ color:#2196F3; text-shadow:0 0 10px #2196F3; }}
            50% {{ color:#FFC107; text-shadow:0 0 25px #FFC107; }}
            100% {{ color:#2196F3; text-shadow:0 0 10px #2196F3; }}
        }}

        h1 {{
            animation: glow 3s infinite alternate;
            font-weight: 800;
            text-align:center;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


set_bg(BACKGROUND)


# ---------------------------------
# Place Logos
# ---------------------------------
def place_logo(path, css_class):
    if os.path.isfile(path):
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(
            f'''
            <div class="{css_class}">
                <img src="data:image/png;base64,{encoded}">
            </div>
            ''',
            unsafe_allow_html=True
        )


# Show logos
place_logo(GRANT, "top-left")
place_logo(RAJAGIRI, "top-right")


# ---------------------------------
# Title
# ---------------------------------
st.title("IPL Cricket Analyser")

st.markdown("""
This project demonstrates:

- Dataset Upload  
- Cleaning  
- EDA  
- ML Model (Winner Prediction)  
- Final Prediction  
""")


# ---------------------------------
# Utility Functions
# ---------------------------------
@st.cache_data
def load_sample():
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


def clean(df):
    df = df.copy()
    df.dropna(how="all", axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------
# Sidebar Navigation
# ---------------------------------
page = st.sidebar.selectbox(
    "Navigation",
    ["Upload Dataset", "Data Cleaning", "EDA", "Model Training", "Predict"]
)


# ---------------------------------
# Pages
# ---------------------------------
if page == "Upload Dataset":
    st.header("Upload dataset or use sample")
    f = st.file_uploader("Upload IPL CSV", type=["csv"])

    if f:
        st.session_state.df = pd.read_csv(f)
        st.success("Dataset uploaded!")

    if st.button("Load sample"):
        st.session_state.df = load_sample()
        st.success("Sample loaded.")

    if "df" in st.session_state:
        st.dataframe(st.session_state.df)

elif page == "Data Cleaning":
    st.header("Data Cleaning")

    if "df" not in st.session_state:
        st.warning("Upload dataset first")
    else:
        df = st.session_state.df
        st.dataframe(df.isnull().sum())

        if st.button("Clean"):
            df = clean(df)
            st.session_state.df = df
            st.success("Cleaned!")
            st.dataframe(df)


# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.markdown("**Rajagiri College – Capstone Project**")
