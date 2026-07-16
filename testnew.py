import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import TweedieRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------
df = pd.read_json("youtube_cleaned.json", orient="records", lines=True)
print("Dataset shape:", df.shape)

features = ["likes", "dislikes", "comment_count"]
ml_df = df.dropna(subset=features + ["views"]).copy()

# Log transform features only. Target stays on its raw scale because the
# GLM's log link handles the skew on the model side.
X = np.log1p(ml_df[features])
y = ml_df["views"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------
# 2. Multicollinearity check (VIF)
# ---------------------------------------------------------------
vif = pd.DataFrame({
    "feature": X_train.columns,
    "VIF": [
        variance_inflation_factor(X_train.values, i)
        for i in range(X_train.shape[1])
    ],
})
print("\nVariance Inflation Factors (above ~5-10 signals collinearity):")
print(vif.to_string(index=False))

# ---------------------------------------------------------------
# 3. Mean prediction model: Tweedie GLM with log link
#    power between 1 and 2 suits heavy tailed positive data.
#    Try power=1.5 as a default; power=2 is a Gamma GLM.
# ---------------------------------------------------------------
mean_model = TweedieRegressor(power=1.5, link="log", alpha=0.0, max_iter=10000)
mean_model.fit(X_train, y_train)

y_pred = mean_model.predict(X_test)

print("\n--- Tweedie GLM (log link) test set performance ---")
print(f"R² (original scale):   {r2_score(y_test, y_pred):.4f}")
print(f"RMSE (original scale): {root_mean_squared_error(y_test, y_pred):,.0f}")
print(f"MAE  (original scale): {mean_absolute_error(y_test, y_pred):,.0f}")

# Coefficients are on the log link scale, so they read as elasticity-like
# effects: coefficient * 1% change in (1 + feature) -> % change in E[views]
print("\nCoefficients (log link scale):")
for name, coef in zip(features, mean_model.coef_):
    print(f"  {name}: {coef:.4f}")
print(f"  intercept: {mean_model.intercept_:.4f}")

# ---------------------------------------------------------------
# 4. Cross validation for a stabler read on performance
# ---------------------------------------------------------------
cv_scores = cross_val_score(
    TweedieRegressor(power=1.5, link="log", alpha=0.0, max_iter=10000),
    X, y, cv=5, scoring="r2",
)
print("\n5-fold CV R² scores:", np.round(cv_scores, 4))
print(f"Mean: {cv_scores.mean():.4f}  Std: {cv_scores.std():.4f}")

# ---------------------------------------------------------------
# 5. Prediction interval model: quantile gradient boosting
#    Fit on log1p(views) here. Quantiles survive monotonic transforms,
#    so expm1 of a log scale quantile is still the quantile of views.
# ---------------------------------------------------------------
y_log_train = np.log1p(y_train)

lower_model = GradientBoostingRegressor(
    loss="quantile", alpha=0.1, random_state=42
)
upper_model = GradientBoostingRegressor(
    loss="quantile", alpha=0.9, random_state=42
)
lower_model.fit(X_train, y_log_train)
upper_model.fit(X_train, y_log_train)

# ---------------------------------------------------------------
# 6. Predict for a new video
# ---------------------------------------------------------------
new_video = pd.DataFrame([{
    "likes": 787425,
    "dislikes": 43420,
    "comment_count": 125882,
}])
new_video_log = np.log1p(new_video)

point_estimate = mean_model.predict(new_video_log)[0]
lower_bound = np.expm1(lower_model.predict(new_video_log))[0]
upper_bound = np.expm1(upper_model.predict(new_video_log))[0]

print("\n--- New video prediction ---")
print(f"Expected views (mean estimate): {point_estimate:,.0f}")
print(f"80% prediction interval:        {lower_bound:,.0f} to {upper_bound:,.0f}")