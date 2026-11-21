
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.joblib")

artifact = load_model()

model = artifact["model"]
metrics = artifact["metrics"]
feature_names = artifact["feature_names"]

# Application heading
st.title("🏠 House Price Prediction System")

st.write(
    "Predict California district-level median "
    "house values using Machine Learning."
)

st.info(
    "This project uses the California Housing Dataset. "
    "It predicts district-level median values, "
    "not individual property selling prices."
)

# Model evaluation
st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("R² Score", f"{metrics['r2']:.3f}")
col2.metric("MAE", f"{metrics['mae']:.3f}")
col3.metric("RMSE", f"{metrics['rmse']:.3f}")

st.divider()

# User input section
st.subheader("Enter Housing Details")

col1, col2 = st.columns(2)

with col1:

    med_inc = st.number_input(
        "Median Income (MedInc)",
        min_value=0.0,
        value=5.0
    )

    house_age = st.number_input(
        "House Age (HouseAge)",
        min_value=0.0,
        value=25.0
    )

    ave_rooms = st.number_input(
        "Average Rooms (AveRooms)",
        min_value=0.0,
        value=5.0
    )

    ave_bedrms = st.number_input(
        "Average Bedrooms (AveBedrms)",
        min_value=0.0,
        value=1.0
    )

with col2:

    population = st.number_input(
        "Population",
        min_value=0.0,
        value=1000.0
    )

    ave_occup = st.number_input(
        "Average Occupancy (AveOccup)",
        min_value=0.0,
        value=3.0
    )

    latitude = st.number_input(
        "Latitude",
        min_value=32.0,
        max_value=42.0,
        value=36.0
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-125.0,
        max_value=-114.0,
        value=-120.0
    )

# Prediction button
if st.button("Predict House Price", type="primary"):

    input_data = pd.DataFrame([{
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    # Maintain training feature order
    input_data = input_data[feature_names]

    # Predict
    prediction = model.predict(input_data)[0]

    # Convert units to dollars
    predicted_price = prediction * 100000

    st.success("Prediction completed!")

    st.subheader("Predicted Median House Value")

    st.metric(
        "Estimated Value",
        f"${predicted_price:,.2f}"
    )

    st.caption(
        "This is an educational model estimate, "
        "not a professional property valuation."
    )
