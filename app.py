import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🏠 House Price Predictor")
st.markdown("### Enter details below")

sqft = st.slider("Total Square Feet", 500, 5000, 1000)
bath = st.slider("Bathrooms", 1, 5, 2)
bhk = st.slider("BHK", 1, 5, 2)

locations = ['Whitefield', 'Sarjapur Road', 'Electronic City', 'other']  # replace later
location = st.selectbox("Location", locations)

# 👉 STEP 3 + 4 GO INSIDE THIS BLOCK
if st.button("Predict"):

    input_data = np.zeros(len(model.feature_names_in_))

    input_data[0] = sqft
    input_data[1] = bath
    input_data[2] = bhk

    try:
        loc_index = list(model.feature_names_in_).index(location)
        input_data[loc_index] = 1
    except:
        pass

    prediction = model.predict([input_data])

    st.success(f"🏷️ Estimated Price: ₹ {round(prediction[0],2)} Lakhs")