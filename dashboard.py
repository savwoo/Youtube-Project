import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import TweedieRegressor
from sklearn.ensemble import GradientBoostingRegressor
import os

@st.cache_resource
def train_models():
    path = os.path.join(os.path.dirname(__file__), "clean_youtube_data.json")
    df = pd.read_json(path, orient="records", lines=True)

    features = ["likes", "dislikes", "comment_count"]
    ml_df = df.dropna(subset=features + ["views"]).copy()

    X = np.log1p(ml_df[features])
    y = ml_df["views"].astype(float)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    y_log_train = np.log1p(y_train)

    mean_model = TweedieRegressor(power=1.5, link="log", alpha=0.0, max_iter=10000)
    mean_model.fit(X_train, y_train)

    lower_model = GradientBoostingRegressor(loss="quantile", alpha=0.1, random_state=42)
    upper_model = GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=42)
    lower_model.fit(X_train, y_log_train)
    upper_model.fit(X_train, y_log_train)

    return mean_model, lower_model, upper_model

st.title("YouTube View Predictor")
st.write("Enter a video's engagement numbers to estimate its view count.")

mean_model, lower_model, upper_model = train_models()

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
