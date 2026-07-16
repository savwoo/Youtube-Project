# YouTube View Predictor

An end-to-end data science project using YouTube trending video data to explore engagement patterns and predict view counts with machine learning.

---

## Project Overview

This project walks through the full data science workflow:
- Downloading and cleaning raw YouTube trending data
- Exploratory data analysis (EDA)
- Training a Tweedie GLM regression model with quantile prediction intervals
- Deploying an interactive Streamlit dashboard for real-time predictions

---

## Files

| File | Description |
|------|-------------|
| `cleanYTcode.py` | Downloads and cleans the raw dataset via KaggleHub, exports to JSON |
| `EDA_cleandata.py` | Exploratory data analysis — shape, missing values, summary stats |
| `model_input.py` | Trains the Tweedie GLM and quantile gradient boosting models, saves `.pkl` files |
| `dashboard.py` | Streamlit web app — input likes/dislikes/comments, get predicted views |
| `clean_youtube_data.py` | Utility script to extract a 100-row sample of the cleaned dataset |
| `clean_youtube_data.json` | 100-row sample of the cleaned dataset |
| `requirements.txt` | Python dependencies |

---

## Dataset

[YouTube Trending Video Dataset](https://www.kaggle.com/datasets/datasnaek/youtube-new) via KaggleHub — trending videos across 10 countries (US, CA, GB, DE, FR, IN, JP, KR, MX, RU).

### Sample Data

| Title | Views | Likes | Dislikes | Comments |
|-------|------:|------:|---------:|---------:|
| Eminem - Walk On Water (Audio) ft. Beyoncé | 17,158,579 | 787,425 | 43,420 | 125,882 |
| PLUSH - Bad Unboxing Fan Mail | 1,014,651 | 127,794 | 1,688 | 13,030 |
| Racist Superman \| Rudy Mancuso, King Bach & Lele Pons | 3,191,434 | 146,035 | 5,339 | 8,181 |
| I Dare You: GOING BALD!? | 2,095,828 | 132,239 | 1,989 | 17,518 |
| Ed Sheeran - Perfect (Official Music Video) | 33,523,622 | 1,634,130 | 21,082 | 85,067 |
| Jake Paul Says Alissa Violet CHEATED with LOGAN PAUL! | 1,309,699 | 103,755 | 4,613 | 12,143 |
| Vanoss Superhero School - New Students | 2,987,945 | 187,464 | 9,850 | 26,629 |
| WE WANT TO TALK ABOUT OUR MARRIAGE | 748,374 | 57,534 | 2,967 | 15,959 |

---

## Model

**Tweedie GLM** (log link, power=1.5) predicting raw view counts from:
- Likes
- Dislikes
- Comment count

Features are log-transformed to handle heavy skew. An **80% prediction interval** is generated using quantile gradient boosting (10th and 90th percentiles).

| Metric | Value |
|--------|-------|
| R² (test set) | 0.33 |
| Cross-validated R² (5-fold) | 0.36 |
| RMSE | ~6,075,611 views |

---

## Running Locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Clean the data** (requires Kaggle credentials)
```bash
python cleanYTcode.py
```

**3. Train the model**
```bash
python model_input.py
```

**4. Launch the dashboard**
```bash
streamlit run dashboard.py
```

---

## Dashboard

Enter a video's engagement numbers and the model returns:
- **Expected view count** (mean estimate)
- **80% prediction interval** (lower and upper bounds)
