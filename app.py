# Streamlit Real Estate Price Estimator with XGBoost

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import joblib
import os

# ─── App Config ───────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🏠 Real Estate Price Estimator")

# ─── Load Data ────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_df.csv")
    df = df.dropna(subset=['ListedPrice', 'Bedroom', 'Bathroom', 'Area', 'State', 'City'])
    df = df.query("ListedPrice <= 2500000 and Area <= 7500")
    df['Bedroom'] = df['Bedroom'].astype(int)
    df = df[df['Bedroom'] > 0]
    df['BedroomsGroupedNum'] = df['Bedroom'].apply(lambda x: x if x < 6 else 6)
    return df

df = load_data()

# ─── Sidebar Inputs ───────────────────────────────────
st.sidebar.header("🏗️ Enter Home Features")

# Bedrooms first
bedroom_input = st.sidebar.selectbox("🛏️ Bedrooms", ["1", "2", "3", "4", "5", "6+"], index=2)
bedroom = 6 if bedroom_input == "6+" else int(bedroom_input)

# Limit bathroom options to realistic range based on bedroom
valid_bathroom_options = {
    1: [1],
    2: [1, 2],
    3: [1, 2, 3],
    4: [2, 3],
    5: [2, 3, 4],
    6: [3, 4, 5, 6]
}
bathroom_options = valid_bathroom_options.get(bedroom, [1, 2, 3])
bathroom_input = st.sidebar.selectbox("🛁 Bathrooms", [str(b) for b in bathroom_options])
bathroom = int(bathroom_input)

# Other inputs
sqft = st.sidebar.slider("📐 Square Footage", min_value=300, max_value=7500, value=2000)
state = st.sidebar.selectbox("🏳️ State", sorted(df['State'].unique()))
