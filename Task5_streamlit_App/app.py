from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

MODEL_PATH = Path(__file__).parent / "model.joblib"

# name: (label, min, max, default, step)
FEATURES = {
    "MedInc": ("Median income (in $10,000s)", 0.5, 15.0, 3.9, 0.1),
    "HouseAge": ("House age (years)", 1, 52, 29, 1),
    "AveRooms": ("Average rooms per household", 1.0, 10.0, 5.4, 0.1),
    "AveBedrms": ("Average bedrooms per household", 0.5, 5.0, 1.1, 0.1),
    "Population": ("Block population", 3, 10000, 1425, 50),
    "AveOccup": ("Average occupants per household", 1.0, 10.0, 3.0, 0.1),
    "Latitude": ("Latitude", 32.5, 42.0, 34.2, 0.01),
    "Longitude": ("Longitude", -124.4, -114.3, -118.5, 0.01),
}


@st.cache_resource
def load_model():
    """Load the saved model with joblib. If model.joblib is missing, train one."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    from sklearn.datasets import fetch_california_housing
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeRegressor(max_depth=8, random_state=42)),
    ])
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


model = load_model()

st.title("🏠 California House Price Predictor")
st.write("Adjust the sliders to describe a neighbourhood and get a real-time price prediction.")

st.sidebar.header("Input features")
values = {}
for key, (label, lo, hi, default, step) in FEATURES.items():
    if isinstance(step, int) and isinstance(lo, int):
        values[key] = st.sidebar.slider(label, int(lo), int(hi), int(default), int(step))
    else:
        values[key] = st.sidebar.slider(label, float(lo), float(hi), float(default), float(step))

input_df = pd.DataFrame([values])[list(FEATURES.keys())]

prediction = float(model.predict(input_df)[0]) * 100_000

st.subheader("Predicted median house value")
st.metric(label="Estimated price", value=f"${prediction:,.0f}")

if prediction < 150_000:
    st.info("Lower-priced area")
elif prediction < 300_000:
    st.success("Mid-priced area")
else:
    st.warning("High-priced area")

with st.expander("Your input values"):
    st.dataframe(input_df, use_container_width=True)

st.subheader("Location")
st.map(pd.DataFrame({"lat": [values["Latitude"]], "lon": [values["Longitude"]]}))

st.caption("Model: Scikit-Learn pipeline trained on the California Housing dataset (Task 4).")
