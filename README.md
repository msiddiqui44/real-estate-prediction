# Real Estate Price Estimator

This [Streamlit web application](https://mj-real-estate-predictor.streamlit.app)  predicts the market price of residential homes based on user-defined inputs (bedrooms, bathrooms, square footage, and state) and visualizes real estate trends interactively.

## Features
- Interactive Price Prediction using a trained Random Forest Regressor
- Choropleth Map of average prices by U.S. state
- Scatter Plot of price vs. square footage with trend line and prediction marker
- Bar Chart of top cities by average price (filtered by bedrooms and bathrooms)
- Box Plot of listed prices by bedroom count (per selected state)

## Dataset
**Source**: [US House Listings 2023 on Kaggle](https://www.kaggle.com/datasets/febinphilips/us-house-listings-2023?resource=download)

The dataset used is `cleaned_df.csv`, which includes columns:
- `ListedPrice`
- `Bedroom`
- `Bathroom`
- `Area`
- `State`
- `City`

Rows with missing or invalid values are filtered out during loading.

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/msiddiqui44/real-estate-prediction.git
   cd real-estate-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your `cleaned_df.csv` file to the project root.

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Example Use
1. Select bedroom and bathroom count, square footage, and state.
2. Click "Predict Price".
3. View:
   - Estimated market price
   - State-level price heatma
   - Price vs. square footage chart with your prediction marked
   - City average prices
   - Box plot of prices by bedroom count

## Model
- Algorithm: Random Forest Regressor
- Features Used: Bedroom, Bathroom, Area, State
- Pipeline: Includes preprocessing for numerical and categorical features
---

Made using Streamlit and Plotly

**Created by:** Mustafa Siddiqui and Joshua Golconda
