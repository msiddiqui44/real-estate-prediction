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

# Limit bathroom options based on bedroom count
valid_bathroom_options = {
    1: [1],
    2: [1, 2],
    3: [1, 2, 3],
    4: [2, 3, 4],
    5: [3, 4, 5],
    6: [4, 5, 6]
}
bathroom_options = valid_bathroom_options.get(bedroom, [1, 2, 3])
bathroom_input = st.sidebar.selectbox("🛁 Bathrooms", [str(b) for b in bathroom_options])
bathroom = int(bathroom_input)

# Other inputs
sqft = st.sidebar.slider("📐 Square Footage", min_value=300, max_value=7500, value=2000)
state = st.sidebar.selectbox("🏳️ State", sorted(df['State'].unique()))

# ─── Model Prep ───────────────────────────────────────
X = df[['Bedroom', 'Bathroom', 'Area', 'State']]
y = df['ListedPrice']

num_feats = ['Bedroom', 'Bathroom', 'Area']
cat_feats = ['State']

numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median'))])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_feats),
    ('cat', categorical_transformer, cat_feats)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── Load or Train XGBoost Model ─────────────────────
MODEL_PATH = "xgb_model.joblib"
@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.warning(f"⚠️ Failed to load model. Removing and retraining. Reason: {e}")
            os.remove(MODEL_PATH)

    with st.spinner("🔁 Training new XGBoost model..."):
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, MODEL_PATH)
        return pipeline

model = load_or_train_model()

# ─── Predict & Visualize ─────────────────────────────
if st.sidebar.button("🚀 Predict Price"):
    user_input = pd.DataFrame({
        'Bedroom': [bedroom],
        'Bathroom': [bathroom],
        'Area': [sqft],
        'State': [state]
    })
    predicted_price = max(model.predict(user_input)[0], 0)

    st.markdown("### 📊 Estimated Market Price")
    st.markdown(f"<h1 style='color: yellow; font-weight: bold; font-size: 48px;'>${predicted_price:,.2f}</h1>", unsafe_allow_html=True)

    # [visualization code continues as before...]
