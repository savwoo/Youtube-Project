import streamlit as st
import numpy as np
import pandas as pd
import joblib

mean_model = joblib.load("mean_model.pkl")
lower_model = joblib.load("lower_model.pkl")
upper_model = joblib.load("upper_model.pkl")

st.title("YouTube View Predictor")
st.write("Enter a video's engagement numbers to estimate its view count.")

likes = st.number_input("Likes", min_value=0, value=50000, step=1000)
dislikes = st.number_input("Dislikes", min_value=0, value=2000, step=100)
comment_count = st.number_input("Comment Count", min_value=0, value=3000, step=100)

if st.button("Predict Views"):
    X_new = np.log1p(pd.DataFrame([{
        "likes": likes,
        "dislikes": dislikes,
        "comment_count": comment_count
    }]))

    point_estimate = mean_model.predict(X_new)[0]
    lower_bound = np.expm1(lower_model.predict(X_new))[0]
    upper_bound = np.expm1(upper_model.predict(X_new))[0]

    st.metric("Predicted Views", f"{int(point_estimate):,}")
    st.write(f"**80% prediction interval:** {int(lower_bound):,} to {int(upper_bound):,}")
