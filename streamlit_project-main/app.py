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
# Resolve path relative to this file:
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
        /* Hide default Streamlit header & footer (optional) */
        [data-testid="stHeader"], footer {{
            visibility: hidden;
            height: 0px;
            margin: 0px;
            padding: 0px;
        }}

        /* Main page padding */
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }}

        /* Background image on app root */
        .stApp {{
            background-image: url("data:{mime};base64,{data}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}

        /* White glass overlay for main content */
        .stApp > [data-testid="block-container"] {{
            background-color: rgba(255, 255, 255, 0.88);
            border-radius: 10px;
        }}

        /* Sidebar transparent */
        [data-testid="stSidebar"] > div:first-child {{
            background-color: transparent !important;
            border-right: none;
        }}

        /* Title glow animation */
        @keyframes glow {{
            0% {{ color: #2196F3; text-shadow: 0 0 5px rgba(33, 150, 243, 0.5); }}
            50% {{ color: #FFC107; text-shadow: 0 0 10px #FFC107, 0 0 20px #FF9800; }}
            100% {{ color: #2196F3; text-shadow: 0 0 5px rgba(33, 150, 243, 0.5); }}
        }}
        h1 {{
            animation: glow 3s infinite alternate;
            text-align: center;
            font-weight: 800;
            padding-bottom: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Call background function
set_background_image(_bg_path_to_use)

# -----------------------
# Rest of your app
# -----------------------
# st.title("IPL Analysis Capstone")
# ...


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
    # prefer /mnt/data if present
    local_paths = [os.path.join("/mnt/data", "matches.csv"), "matches.csv"]
    for p in local_paths:
        if os.path.isfile(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    # fallback sample
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

# Inference helpers
def infer_pie_from_counts(counts_df, slice_col, value_col="count"):
    if counts_df.empty:
        return "No data to infer from."
    top = counts_df.head(3)
    total = counts_df[value_col].sum()
    parts = []
    for _, row in top.iterrows():
        pct = row[value_col] / total if total else 0
        parts.append(f"{row[slice_col]} ({pct:.1%})")
    if len(counts_df) == 1:
        return f"All data belongs to {parts[0]}."
    top_sum = counts_df.head(3)[value_col].sum() if total else 0
    return f"Top slices: {', '.join(parts)}. These top categories make up {top_sum/total:.1%} of total." if total else "No total to compute percentages."

def infer_pie_from_numeric(agg_df, slice_col, val_col):
    if agg_df.empty:
        return "No data to infer from."
    top = agg_df.head(3)
    total = agg_df[val_col].sum()
    parts = []
    for _, row in top.iterrows():
        parts.append(f"{row[slice_col]} ({row[val_col]:,.0f})")
    return f"Top by {val_col}: {', '.join(parts)}. The top category accounts for { (top.iloc[0][val_col]/total if total else 0):.1% } of the total {val_col}."

def infer_corr(corr_df):
    if corr_df.size == 0:
        return "Not enough numeric columns for correlation."
    corr = corr_df.copy()
    np.fill_diagonal(corr.values, 0)
    try:
        max_idx = np.unravel_index(np.nanargmax(corr.values), corr.shape)
        min_idx = np.unravel_index(np.nanargmin(corr.values), corr.shape)
    except ValueError:
        return "Not enough data to compute correlations."
    max_pair = (corr.index[max_idx[0]], corr.columns[max_idx[1]])
    min_pair = (corr.index[min_idx[0]], corr.columns[min_idx[1]])
    max_val = corr.values[max_idx]
    min_val = corr.values[min_idx]
    parts = []
    if abs(max_val) >= 0.3:
        parts.append(f"Strongest positive correlation: {max_pair[0]} vs {max_pair[1]} = {max_val:.2f}")
    if abs(min_val) >= 0.3:
        parts.append(f"Strongest negative correlation: {min_pair[0]} vs {min_pair[1]} = {min_val:.2f}")
    if not parts:
        return "No strong correlations found (all absolute correlations < 0.3)."
    return ". ".join(parts) + "."

def infer_pivot(pivot_table):
    if pivot_table.size == 0:
        return "Pivot produced an empty table."
    idx = np.unravel_index(np.nanargmax(pivot_table.values), pivot_table.shape)
    row_label = pivot_table.index[idx[0]]
    col_label = pivot_table.columns[idx[1]]
    max_val = pivot_table.values[idx]
    return f"Highest aggregated cell: {row_label} × {col_label} = {max_val:.2f}. Use this to spot strong interactions between categories."

def infer_bar(grouped_df, x_col, y_col):
    if grouped_df is None or grouped_df.empty:
        return "No data to infer from."
    top = grouped_df.sort_values(by=y_col, ascending=False).head(3)
    total = grouped_df[y_col].sum()
    parts = []
    for _, row in top.iterrows():
        parts.append(f"{row[x_col]} ({row[y_col]:,.0f})")
    return f"Top {x_col}: {', '.join(parts)}. Top 3 account for {(top[y_col].sum()/total if total else 0):.1%} of total."

def infer_line(df_line, x_col, y_col):
    if df_line is None or df_line.empty or len(df_line) < 2:
        return "Not enough data points to infer a trend."
    if np.issubdtype(df_line[x_col].dtype, np.number):
        x = df_line[x_col].astype(float).values
        y = df_line[y_col].astype(float).values
        try:
            coef = np.polyfit(x, y, 1)
            slope = coef[0]
            if abs(slope) < 1e-6:
                trend = "no clear trend (slope ≈ 0)."
            elif slope > 0:
                trend = f"upward trend (slope ≈ {slope:.3f})."
            else:
                trend = f"downward trend (slope ≈ {slope:.3f})."
            return f"{y_col} shows an {trend}"
        except Exception:
            return "Unable to compute trend (numeric fit failed)."
    else:
        grouped = df_line.groupby(x_col)[y_col].mean().reset_index()
        if len(grouped) < 2:
            return "Not enough categories to infer trend."
        first = grouped.iloc[0][y_col]
        last = grouped.iloc[-1][y_col]
        if last > first:
            return f"{y_col} increases from first to last category (avg {first:.2f} → {last:.2f})."
        elif last < first:
            return f"{y_col} decreases from first to last category (avg {first:.2f} → {last:.2f})."
        else:
            return f"No change in average {y_col} from first to last category."

# -----------------------
# Sidebar navigation
# -----------------------
page = st.sidebar.selectbox("Navigation", ["Upload / Sample", "Data Cleaning", "EDA", "Model Training", "Predict"])

# -----------------------
# Upload / Sample
# -----------------------
if page == "Upload / Sample":
    st.header("Upload dataset or use sample")
    uploaded_file = st.file_uploader("Upload your IPL CSV (matches.csv recommended)", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["df"] = df
            st.success("Uploaded dataset saved to session.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        if st.button("Load sample matches.csv (local)"):
            df = load_sample_data()
            st.session_state["df"] = df
            st.success("Loaded sample dataset (with 'winner' as sample target).")

    if "df" in st.session_state:
        st.subheader("Preview dataset (first 10 rows)")
        st.dataframe(st.session_state["df"].head(10), height=300)
        if st.checkbox("Show dataset info"):
            buffer = io.StringIO()
            st.session_state["df"].info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

# -----------------------
# Data Cleaning
# -----------------------
if page == "Data Cleaning":
    st.header("Data Cleaning & Preprocessing")
    if "df" not in st.session_state:
        st.warning("Upload a dataset first (go to 'Upload / Sample').")
    else:
        df = st.session_state["df"].copy()
        st.markdown(f"Initial shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.subheader("Missing values per column")
        missing = df.isnull().sum().sort_values(ascending=False)
        st.dataframe(pd.DataFrame({"missing": missing}))
        if st.button("Run basic cleaning (drop full-null cols, duplicates, standardize names)"):
            df_clean = basic_cleaning(df)
            st.session_state["df"] = df_clean
            st.success("Basic cleaning complete.")
            st.write(df_clean.head())
        if st.checkbox("Show sample of problematic rows (teams equal)"):
            if "team1" in df.columns and "team2" in df.columns:
                bad = df[df["team1"] == df["team2"]]
                st.write(bad.head())

# -----------------------
# EDA (pie, heatmap, bar, line) - SHOW ALL BY DEFAULT BUT OPTIONAL LIMIT
# -----------------------
if page == "EDA":
    st.header("Exploratory Data Analysis (EDA) — All charts + Inferences")
    if "df" not in st.session_state:
        st.warning("Upload dataset first.")
    else:
        df = st.session_state["df"].copy()
        df_clean = basic_cleaning(df)

        # UI control: whether to limit categories shown (default: OFF -> show all)
        with st.expander("Display options (show all by default)"):
            limit_toggle = st.checkbox("Limit displayed categories", value=False, help="When checked, charts will be trimmed to the chosen number of categories to improve readability/performace.")
            limit_value = st.slider("Max categories to display when limiting", min_value=10, max_value=2000, value=200, step=10) if limit_toggle else None

        def apply_limit(df_obj):
            if limit_toggle and (limit_value is not None):
                return df_obj.head(limit_value)
            return df_obj

        all_cols = df_clean.columns.tolist()
        num_cols = numeric_columns(df_clean)
        cat_cols = categorical_columns(df_clean)

        # --- PIE CHART ---
        st.subheader("Pie Chart")
        if not all_cols:
            st.info("Dataset is empty after cleaning.")
        else:
            pie_options = cat_cols if cat_cols else all_cols
            pie_col = st.selectbox("Pie — categorical column (slices)", options=pie_options, index=0, key="pie_col")
            pie_use_numeric = st.checkbox("Use numeric column for slice size (sum by category)?", value=False, key="pie_use_num")
            pie_num_col = None
            if pie_use_numeric and num_cols:
                pie_num_col = st.selectbox("Pie — numeric column to aggregate", options=num_cols, key="pie_num_col")
            try:
                import plotly.express as px
                if pie_use_numeric and pie_num_col:
                    pie_df = df_clean.groupby(pie_col)[pie_num_col].sum().reset_index().sort_values(by=pie_num_col, ascending=False)
                    pie_df_disp = apply_limit(pie_df)
                    if pie_df_disp.empty:
                        st.info("No data for pie (numeric aggregate returned empty).")
                    else:
                        fig_pie = px.pie(pie_df_disp, names=pie_col, values=pie_num_col, title=f"Sum of {pie_num_col} by {pie_col}", hole=0.3)
                        st.plotly_chart(fig_pie, use_container_width=True)
                        st.markdown("*Inference:*")
                        st.write(infer_pie_from_numeric(pie_df, pie_col, pie_num_col))
                else:
                    counts = df_clean[pie_col].value_counts().reset_index()
                    counts.columns = [pie_col, "count"]
                    counts_disp = apply_limit(counts)
                    if counts_disp.empty:
                        st.info("No counts available for chosen column.")
                    else:
                        fig_pie = px.pie(counts_disp, names=pie_col, values="count", title=f"Counts of {pie_col}", hole=0.3)
                        st.plotly_chart(fig_pie, use_container_width=True)
                        st.markdown("*Inference:*")
                        st.write(infer_pie_from_counts(counts, pie_col, "count"))
            except Exception as e:
                st.error(f"Failed to render pie chart: {e}")

        st.markdown("---")

        # --- HEATMAP ---
        st.subheader("Heatmap")
        heat_type = st.selectbox("Heatmap type", ["Correlation (numeric)", "Pivot (categorical vs categorical aggregated numeric)"], key="heat_type")
        if heat_type == "Correlation (numeric)":
            numeric_cols = num_cols
            if len(numeric_cols) < 2:
                st.info("Need at least two numeric columns for correlation heatmap.")
            else:
                corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], index=0, key="corr_method")
                try:
                    corr = df_clean[numeric_cols].corr(method=corr_method)
                    fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols)*0.5), max(4, len(numeric_cols)*0.5)))
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                    ax.set_title(f"Numeric correlation ({corr_method})")
                    st.pyplot(fig)
                    st.markdown("*Inference:*")
                    st.write(infer_corr(corr))
                except Exception as e:
                    st.error(f"Failed to render correlation heatmap: {e}")
        else:
            if len(cat_cols) < 1 or len(num_cols) < 1:
                st.info("Pivot heatmap requires at least one categorical and one numeric column.")
            else:
                heat_x = st.selectbox("Pivot — index (categorical)", options=cat_cols, key="heat_x")
                heat_y_candidates = [c for c in cat_cols if c != heat_x] or cat_cols
                heat_y = st.selectbox("Pivot — columns (categorical)", options=heat_y_candidates, key="heat_y")
                heat_val = st.selectbox("Pivot — numeric value (aggregated)", options=num_cols, key="heat_val")
                heat_agg = st.selectbox("Aggregation", ["sum", "mean", "median"], index=1, key="heat_agg")
                try:
                    pivot = df_clean.groupby([heat_x, heat_y])[heat_val].agg(heat_agg).reset_index()
                    pivot_table = pivot.pivot(index=heat_x, columns=heat_y, values=heat_val).fillna(0)
                    fig, ax = plt.subplots(figsize=(max(6, pivot_table.shape[1]*0.6), max(4, pivot_table.shape[0]*0.6)))
                    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
                    ax.set_title(f"{heat_agg.title()} of {heat_val} by {heat_x} vs {heat_y}")
                    st.pyplot(fig)
                    st.markdown("*Inference:*")
                    st.write(infer_pivot(pivot_table))
                except Exception as e:
                    st.error(f"Failed to render pivot heatmap: {e}")

        st.markdown("---")

        # --- BAR CHART ---
        st.subheader("Bar Chart")
        if not all_cols:
            st.info("No columns to show.")
        else:
            default_x = "team1" if "team1" in all_cols else all_cols[0]
            bar_x = st.selectbox("Bar chart — X axis", options=all_cols, index=all_cols.index(default_x) if default_x in all_cols else 0, key="bar_x")
            bar_y_options = ["(Count)"] + all_cols
            bar_y = st.selectbox("Bar chart — Y axis (choose a column or leave as 'Count')", options=bar_y_options, index=0, key="bar_y")
            if bar_y != "(Count)":
                if bar_y in num_cols:
                    agg = st.selectbox("Aggregation for numeric Y", options=["sum", "mean", "median"], key="bar_agg")
                else:
                    agg = "count"
            else:
                agg = "count"
            try:
                import plotly.express as px
                if agg == "count":
                    if bar_y != "(Count)" and bar_y in cat_cols:
                        grouped = df_clean.groupby([bar_x, bar_y]).size().reset_index(name="count")
                        grouped_disp = apply_limit(grouped)
                        if grouped_disp.empty:
                            st.info("No data available for grouped bar chart.")
                        else:
                            fig1 = px.bar(grouped_disp, x=bar_x, y="count", color=bar_y, title=f"Counts of {bar_x} by {bar_y}", height=450)
                            st.plotly_chart(fig1, use_container_width=True)
                            st.markdown("*Inference:*")
                            grouped_tot = grouped.groupby(bar_x)["count"].sum().reset_index()
                            st.write(infer_bar(grouped_tot, bar_x, "count"))
                    else:
                        counts = df_clean[bar_x].value_counts().reset_index()
                        counts.columns = [bar_x, "count"]
                        counts_disp = apply_limit(counts)
                        if counts_disp.empty:
                            st.info("No data for counts.")
                        else:
                            fig1 = px.bar(counts_disp, x=bar_x, y="count", title=f"Counts of {bar_x}", text="count", height=450)
                            fig1.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig1, use_container_width=True)
                            st.markdown("*Inference:*")
                            st.write(infer_bar(counts, bar_x, "count"))
                else:
                    df_tmp = df_clean[[bar_x, bar_y]].dropna()
                    if df_tmp.empty:
                        st.info("No data after dropping NA for selected columns.")
                        st.markdown("*Inference:* No data after dropping NA.")
                    else:
                        if agg == "sum":
                            grouped = df_tmp.groupby(bar_x)[bar_y].sum().reset_index()
                        elif agg == "mean":
                            grouped = df_tmp.groupby(bar_x)[bar_y].mean().reset_index()
                        else:
                            grouped = df_tmp.groupby(bar_x)[bar_y].median().reset_index()
                        grouped = grouped.sort_values(by=bar_y, ascending=False)
                        grouped_disp = apply_limit(grouped)
                        if grouped_disp.empty:
                            st.info("No grouped data to display after applying limit.")
                        else:
                            fig1 = px.bar(grouped_disp, x=bar_x, y=bar_y, title=f"{agg.title()} of {bar_y} by {bar_x}", text=bar_y, height=450)
                            fig1.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig1, use_container_width=True)
                            st.markdown("*Inference:*")
                            st.write(infer_bar(grouped, bar_x, bar_y))
            except Exception as e:
                st.error(f"Failed to render bar chart: {e}")

        st.markdown("---")

        # --- LINE CHART ---
        st.subheader("Line Chart")
        if not all_cols:
            st.info("No columns to show.")
        else:
            default_line_x = "season" if "season" in all_cols else all_cols[0]
            line_x = st.selectbox("Line chart — X axis", options=all_cols, index=all_cols.index(default_line_x) if default_line_x in all_cols else 0, key="line_x")
            line_y_options = num_cols if num_cols else all_cols
            line_y = st.selectbox("Line chart — Y axis (must be numeric)", options=line_y_options, index=0, key="line_y")
            line_agg = st.selectbox("If X is categorical, how to aggregate Y over X?", options=["sum", "mean", "median"], index=1, key="line_agg")
            try:
                df_line = df_clean[[line_x, line_y]].dropna()
                import plotly.express as px
                if df_line.empty:
                    st.info("No data available for the selected X/Y after dropping NA.")
                    st.markdown("*Inference:* No data available.")
                else:
                    if line_x in num_cols:
                        df_line_sorted = df_line.sort_values(by=line_x)
                        df_line_disp = apply_limit(df_line_sorted)
                        fig3 = px.line(df_line_disp, x=line_x, y=line_y, markers=True, title=f"{line_y} vs {line_x}", height=400)
                        st.plotly_chart(fig3, use_container_width=True)
                        st.markdown("*Inference:*")
                        st.write(infer_line(df_line_sorted, line_x, line_y))
                    else:
                        if line_agg == "sum":
                            grouped = df_line.groupby(line_x)[line_y].sum().reset_index()
                        elif line_agg == "mean":
                            grouped = df_line.groupby(line_x)[line_y].mean().reset_index()
                        else:
                            grouped = df_line.groupby(line_x)[line_y].median().reset_index()
                        grouped_plot = grouped.sort_values(by=line_y, ascending=False)
                        grouped_disp = apply_limit(grouped_plot)
                        if grouped_disp.empty:
                            st.info("No aggregated data to plot.")
                        else:
                            fig3 = px.line(grouped_disp, x=line_x, y=line_y, markers=True, title=f"{line_agg.title()} of {line_y} by {line_x}", height=400)
                            fig3.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig3, use_container_width=True)
                            st.markdown("*Inference:*")
                            st.write(infer_line(grouped, line_x, line_y))
            except Exception as e:
                st.error(f"Failed to render line chart: {e}")

# -----------------------
# Model Training — target locked to 'winner'
# -----------------------
if page == "Model Training":
    st.header("Model Training — Predict Match Winner (target: winner)")
    if "df" not in st.session_state:
        st.warning("Upload dataset first.")
    else:
        raw = st.session_state["df"].copy()
        df_clean = basic_cleaning(raw)

        st.markdown("Target column is fixed to: *winner*")

        if "winner" not in df_clean.columns:
            st.error("Dataset does not contain required column: 'winner'. Please upload a dataset with that column name.")
        else:
            if st.button("Prepare features & train model"):
                X, y, le_target = prepare_features_for_model(df_clean, target_col="winner")
                if X is None or y is None:
                    st.error("Target not found or invalid. Make sure 'winner' exists and rows are valid after cleaning.")
                else:
                    st.write("Feature columns:", X.columns.tolist())
                    preprocessor = build_preprocessing_pipeline(X)
                    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                    pipeline = Pipeline(steps=[("preproc", preprocessor), ("clf", clf)])

                    unique_classes = np.unique(y)
                    if len(unique_classes) < 2:
                        st.error("Target column contains only one class after cleaning. Cannot train a classifier.")
                    else:
                        stratify_arg = y if len(unique_classes) > 1 else None
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)
                        except Exception:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        with st.spinner("Training model..."):
                            try:
                                pipeline.fit(X_train, y_train)
                            except Exception as e:
                                st.error(f"Training failed: {e}")
                                st.stop()
                        try:
                            y_pred = pipeline.predict(X_test)
                            acc = accuracy_score(y_test, y_pred)
                            st.success(f"Model trained. Test accuracy: {acc:.4f}")
                        except Exception as e:
                            st.error(f"Prediction on test set failed: {e}")
                            st.stop()
                        st.subheader("Classification Report")
                        try:
                            target_names = list(le_target.classes_) if hasattr(le_target, "classes_") else None
                            if target_names:
                                st.text(classification_report(y_test, y_pred, target_names=target_names))
                            else:
                                st.text(classification_report(y_test, y_pred))
                        except Exception:
                            st.text(classification_report(y_test, y_pred))
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(7, 5))
                        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                        ax.set_title("Confusion Matrix")
                        ax.set_ylabel("True Label")
                        ax.set_xlabel("Predicted Label")
                        st.pyplot(fig)
                        # Save model pipeline
                        try:
                            with open("model_pipeline.pkl", "wb") as f:
                                pickle.dump({"pipeline": pipeline, "le_target": le_target}, f)
                            st.success("Saved model_pipeline.pkl (contains pipeline + le_target)")
                            st.write("Saved label classes:", list(le_target.classes_))
                        except Exception as e:
                            st.error(f"Failed to save model: {e}")

# -----------------------
# Predict — Single match only
# -----------------------
if page == "Predict":
    st.header("Predict — Single match (minimal inputs)")

    pipeline = None
    le_target = None

    try:
        with open("model_pipeline.pkl", "rb") as f:
            saved = pickle.load(f)
            pipeline = saved.get("pipeline")
            le_target = saved.get("le_target")
            if pipeline is not None and le_target is not None:
                st.success("Loaded trained pipeline")
            else:
                st.warning("Loaded pipeline file but missing expected items ('pipeline' / 'le_target').")
    except FileNotFoundError:
        st.warning("No trained pipeline found. Train a model first on the 'Model Training' page.")
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")

    if pipeline is None or le_target is None:
        st.info("Load or train a model to enable predictions.")
    else:
        default_team_options = ["Mumbai Indians", "Chennai Super Kings", "Kolkata Knight Riders",
                                "Royal Challengers Bangalore", "Sunrisers Hyderabad"]
        if "df" in st.session_state:
            raw_df = st.session_state["df"]
            t1 = raw_df['team1'].dropna().unique().tolist() if 'team1' in raw_df.columns else []
            t2 = raw_df['team2'].dropna().unique().tolist() if 'team2' in raw_df.columns else []
            combined = sorted(list(set(t1).union(set(t2))))
            team_options_global = combined if combined else default_team_options
        else:
            team_options_global = default_team_options

        st.subheader("Single match prediction")
        team1 = st.selectbox("Team 1", team_options_global, key="single_team1")
        team2_options = [t for t in team_options_global if t != team1]
        if not team2_options:
            team2_options = team_options_global
        team2 = st.selectbox("Team 2", team2_options, key="single_team2")

        toss_options = [team1, team2]
        st.session_state["single_toss_winner"] = st.session_state.get("single_toss_winner", toss_options[0])
        toss_winner = st.selectbox("Toss winner", toss_options, index=toss_options.index(st.session_state["single_toss_winner"]), key="single_toss_winner")

        st.session_state["single_toss_decision"] = st.session_state.get("single_toss_decision", "bat")
        toss_decision = st.selectbox("Toss decision", ["bat", "field"], index=["bat", "field"].index(st.session_state["single_toss_decision"]), key="single_toss_decision")

        season_val = st.number_input("Season (year)", min_value=2008, max_value=2100, value=2026, key="single_season")

        in_df = pd.DataFrame([{
            "season": int(season_val),
            "team1": team1,
            "team2": team2,
            "city": "Unknown",
            "venue": "Unknown",
            "toss_winner": toss_winner,
            "toss_decision": toss_decision
        }])

        if st.button("Predict single match winner"):
            try:
                pred = pipeline.predict(in_df)
                try:
                    label = le_target.inverse_transform(pred)[0]
                except Exception:
                    label = str(pred[0])
                st.success(f"Predicted winner: *{label}*")
                # only attempt predict_proba if available
                if hasattr(pipeline, "predict_proba") or (hasattr(pipeline, "named_steps") and hasattr(pipeline.named_steps.get("clf"), "predict_proba")):
                    try:
                        proba = pipeline.predict_proba(in_df)[0]
                        probs = dict(zip(le_target.classes_, proba))
                        probs_series = pd.Series(probs).sort_values(ascending=False).map(lambda x: f"{x:.2%}")
                        st.subheader("Prediction probabilities")
                        st.dataframe(probs_series, use_container_width=True)
                    except Exception:
                        st.info("Probability estimates not available for this classifier.")
                else:
                    st.info("Probability estimates not available for this classifier.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.write("[LinkedIn](https://www.linkedin.com/in/ashley-immanuel-81609731b/)")

team_info = """
**Team 4**  
Ashley Immanuel  
Gautham K Surendran  
Mathew Sibi  
Leo Bernard  
Nibin Biju  
Midhul Sasikumar  
Shikha Sachin

Sonu Babu
"""

# Display the team info
st.markdown(team_info)


# Footer
st.markdown("---")
st.markdown("Notes: This app uses only pre-match features. Improve accuracy by adding player stats, recent form, head-to-head stats, roster changes, etc.")




