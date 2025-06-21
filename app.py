# Streamlit Real Estate Price Estimator

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib
import os

# ─── App Config ───
st.set_page_config(layout="wide")
st.title("Real Estate Price Estimator")

# ─── Load Data ───
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

# ─── Sidebar Inputs ───
st.sidebar.header("Enter Home Features")
bedroom_input = st.sidebar.selectbox("Bedrooms", ["1", "2", "3", "4", "5", "6+"], index=2)
bedroom = 6 if bedroom_input == "6+" else int(bedroom_input)
bathroom_input = st.sidebar.selectbox("Bathrooms", ["1", "2", "3", "4", "5", "6+"], index=1)
bathroom = 6 if bathroom_input == "6+" else int(bathroom_input)
sqft = st.sidebar.slider("Square Footage", min_value=300, max_value=7500, value=2000)
state = st.sidebar.selectbox("State", sorted(df['State'].unique()))

# ─── Model Prep ───
X = df[['Bedroom', 'Bathroom', 'Area', 'State']]
y = df['ListedPrice']

num_feats = ['Bedroom', 'Bathroom', 'Area']
cat_feats = ['State']

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_feats),
    ('cat', categorical_transformer, cat_feats)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── Model Load/Train ───
MODEL_PATH = "rf_model.joblib"
@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=20, random_state=42))
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model

model = load_or_train_model()

# ─── Prediction and Visualizations ───
if st.sidebar.button("Predict Price"):
    user_input = pd.DataFrame({
        'Bedroom': [bedroom],
        'Bathroom': [bathroom],
        'Area': [sqft],
        'State': [state]
    })
    predicted_price = max(model.predict(user_input)[0], 0)

    st.markdown("### Estimated Market Price")
    st.markdown(f"<h1 style='color: yellow;'>${predicted_price:,.2f}</h1>", unsafe_allow_html=True)

    # ─── Choropleth Map ───
    st.subheader("Average Price by State")
    choropleth_df = df[(df['Bedroom'] == bedroom) & (df['Bathroom'] == bathroom)]
    if not choropleth_df.empty:
        state_price = choropleth_df.groupby("State")["ListedPrice"].mean().reset_index()
        fig_state = px.choropleth(
            state_price,
            locations="State",
            locationmode="USA-states",
            color="ListedPrice",
            color_continuous_scale="Plasma",
            scope="usa",
            labels={"ListedPrice": "Avg Listed Price"},
            title='Based on selected Bedrooms and Bathrooms'
        )
        fig_state.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            font_color='white',
            geo=dict(bgcolor='black'),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_state, use_container_width=True)
    else:
        st.info("No data available for the selected bedroom and bathroom combination.")

    # ─── Scatter Plot ───
    st.subheader("Price vs Square Footage")
    df_scatter = (
        df[['Area', 'ListedPrice', 'Bedroom']]
        .rename(columns={'Area': 'sqft', 'ListedPrice': 'price', 'Bedroom': 'Bedrooms'})
        .astype({'Bedrooms': 'int'})
        .assign(BedroomsGroupedNum=lambda d: d['Bedrooms'].apply(lambda x: x if x < 6 else 6))
    )
    xs = df_scatter['sqft'].to_numpy()
    ys = df_scatter['price'].to_numpy()
    m, b = np.polyfit(xs, ys, 1)
    sorted_xs = np.sort(xs)

    fig_scatter = go.Figure(data=[
        go.Scatter(
            x=df_scatter["sqft"],
            y=df_scatter["price"],
            mode="markers",
            marker=dict(
                size=7,
                opacity=1.0,
                color=df_scatter["BedroomsGroupedNum"],
                coloraxis="coloraxis"
            ),
            hovertemplate="Sqft: %{x}<br>Price: $%{y:,.0f}<br>Bedrooms: %{marker.color}",
            name="Listings"
        ),
        go.Scatter(
            x=sorted_xs,
            y=m * sorted_xs + b,
            mode='lines',
            line=dict(color='white', width=10),
            name='Trend Line'
        ),
        go.Scatter(
            x=[sqft],
            y=[predicted_price],
            mode="markers+text",
            marker=dict(
                color="yellow",
                size=20,
                symbol="star",
                line=dict(color="black", width=2)
            ),
            name="Your Estimate",
            hoverinfo="text",
            hovertext=[f"Predicted Price: ${predicted_price:,.0f}<br>Sqft: {sqft}"]
        )
    ])
    fig_scatter.update_layout(
        title='Based on all listings',
        paper_bgcolor='black',
        plot_bgcolor='black',
        font_color='white',
        xaxis_title="Square Footage",
        yaxis_title="Price ($)",
        margin=dict(l=60, r=160, t=60, b=60),
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=14)
        ),
        coloraxis=dict(
            colorscale="Blues",
            colorbar=dict(
                title="Bedrooms",
                tickvals=[1, 2, 3, 4, 5, 6],
                ticktext=["1", "2", "3", "4", "5", "6+"],
                len=0.7,
                y=0.5
            )
        )
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ─── Binned Sqft vs Price ───
    st.subheader("Average Price by Binned Square Footage")
    bin_edges = list(range(500, 8001, 500))
    labels = [f"{bin_edges[i]}–{bin_edges[i+1]-1}" for i in range(len(bin_edges)-1)]

    df_binned = df.copy()
    df_binned['SqftBin'] = pd.cut(df_binned['Area'], bins=bin_edges, labels=labels, include_lowest=True)
    bin_avg = df_binned.groupby('SqftBin')['ListedPrice'].mean().reset_index().dropna()

    fig_bins = px.bar(
        bin_avg,
        x='SqftBin',
        y='ListedPrice',
        labels={'SqftBin': 'Square Footage Range', 'ListedPrice': 'Average Price'},
        color='ListedPrice',
        color_continuous_scale='Plasma',
        title='Average Price by Square Footage Range (All Listings)'
    )
    fig_bins.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        font_color='white'
    )
    st.plotly_chart(fig_bins, use_container_width=True)

else:
    st.write("Adjust the sidebar inputs and click **Predict Price** to generate insights.")
