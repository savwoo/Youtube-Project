import numpy as np

# Plug in real or hypothetical numbers
new_video = pd.DataFrame([{
    "likes": 50000,
    "dislikes": 2000,
    "comment_count": 3000
}])

# Must log-transform the input the same way we did during training
new_video_log = np.log1p(new_video)

# Predict (comes back in log scale, so reverse the transform)
predicted_log = model.predict(new_video_log)
predicted_views = np.expm1(predicted_log)

print(f"Predicted views: {int(predicted_views[0]):,}")
