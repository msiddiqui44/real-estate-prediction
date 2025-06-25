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

bedroom_input = st.sidebar.selectbox("🛏️ Bedrooms", ["1", "2", "3", "4", "5", "6+"], index=2)
bedroom = 6 if bedroom_input == "6+" else int(bedroom_input)

# Limit bathroom options based on selected bedroom
valid_bathroom_options = {
    1: [1],
    2: [1, 2],
    3: [1, 2, 3],
    4: [2, 3, 4],
    5: [2, 3, 4, 5],
    6: [3, 4, 5, 6]
}
bathroom_choices = valid_bathroom_options.get(bedroom, [1, 2, 3])
bathroom_input = st.sidebar.selectbox("🛁 Bathrooms", [str(b) for b in bathroom_choices])
bathroom = int(bathroom_input)

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

    # ─── Choropleth Map ─────────────────────────────
    st.subheader("🗺️ Average Price by State (filtered by Bedroom & Bathroom)")
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
            labels={"ListedPrice": "Avg Listed Price"}
        )
        fig_state.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            font_color='white',
            geo=dict(bgcolor='black'),
            title='Average Price by State',
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_state, use_container_width=True)
    else:
        st.info("No data available for the selected bedroom and bathroom combination.")

    # ─── Scatter Plot ──────────────────────────────
    st.subheader("📈 Price vs Square Footage (All Listings)")
    df_scatter = (
        df[['Area','ListedPrice','Bedroom']]
        .rename(columns={'Area':'sqft','ListedPrice':'price','Bedroom':'Bedrooms'})
        .assign(BedroomsGroupedNum=lambda d: d['Bedrooms'].apply(lambda x: x if x < 6 else 6))
    )
    xs = df_scatter['sqft'].to_numpy()
    ys = df_scatter['price'].to_numpy()
    m, b = np.polyfit(xs, ys, 1)
    sorted_xs = np.sort(xs)

    fig_scatter = go.Figure(data=[
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=7, opacity=1.0, color=df_scatter["BedroomsGroupedNum"], coloraxis="coloraxis"),
            hovertemplate="Sqft: %{x}<br>Price: $%{y:,.0f}<br>Bedrooms: %{marker.color}",
            name="Listings"
        ),
        go.Scatter(
            x=sorted_xs,
            y=m * sorted_xs + b,
            mode='lines',
            line=dict(color='white', width=3),
            name='Trend Line'
        ),
        go.Scatter(
            x=[sqft],
            y=[predicted_price],
            mode="markers+text",
            marker=dict(color="yellow", size=20, symbol="star", line=dict(color="black", width=2)),
            name="Your Estimate",
            hoverinfo="text",
            hovertext=[f"Predicted Price: ${predicted_price:,.0f}<br>Sqft: {sqft}"]
        )
    ])
    fig_scatter.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        font_color='white',
        xaxis_title="Square Footage",
        yaxis_title="Price ($)",
        title="All Listings",
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
        coloraxis=dict(
            colorscale="Blues",
            colorbar=dict(
                title="Bedrooms",
                tickvals=[1, 2, 3, 4, 5, 6],
                ticktext=["1", "2", "3", "4", "5", "6+"]
            )
        )
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ─── Bar Chart ─────────────────────────────────
    st.subheader("🏙️ Avg Price by City (Top 15 in State + Filters)")
    city_df = df[(df['Bedroom'] == bedroom) & (df['Bathroom'] == bathroom) & (df['State'] == state)]
    if not city_df.empty:
        top_cities = city_df['City'].value_counts().nlargest(15).index
        city_df = city_df[city_df['City'].isin(top_cities)]
        avgprice = city_df.groupby("City")["ListedPrice"].mean().reset_index().sort_values("ListedPrice", ascending=False)
        fig_bar = px.bar(
            avgprice,
            x="City",
            y="ListedPrice",
            labels={"ListedPrice": "Average Price"},
            color="ListedPrice",
            color_continuous_scale="Plasma"
        )
        fig_bar.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            font_color='white',
            title='Top 15 Most Active Cities in Selected State'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ─── Box Plot ──────────────────────────────────
    st.subheader(f"📦 Listed Prices by Bedrooms in {state}")
    state_box = df[df['State'] == state].copy()
    state_box = state_box[state_box['BedroomsGroupedNum'] > 0]
    color_map = {
        1: "#FF6F61", 2: "#F7B801", 3: "#00A6A6",
        4: "#9D00FF", 5: "#FF1493", 6: "#00FF7F"
    }
    fig_box = px.box(
        state_box,
        x="BedroomsGroupedNum",
        y="ListedPrice",
        color="BedroomsGroupedNum",
        points="outliers",
        color_discrete_map=color_map
    )
    fig_box.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        font_color='white',
        xaxis=dict(
            title="Bedrooms",
            tickvals=[1, 2, 3, 4, 5, 6],
            ticktext=["1", "2", "3", "4", "5", "6+"]
        ),
        yaxis_title="Listed Price ($)",
        showlegend=False,
        title="Boxplot by Bedroom Count (Filtered by State)"
    )
    st.plotly_chart(fig_box, use_container_width=True)

else:
    st.write("👈 Adjust input features on the sidebar and click **Predict Price** to see the estimated home price.")
