import kagglehub
import os
import pandas as pd

path = kagglehub.dataset_download("datasnaek/youtube-new")
files = [f for f in os.listdir(path) if f.endswith(".csv")]

dfs = []
for f in files:
    country = f.replace("videos.csv", "")
    temp = pd.read_csv(os.path.join(path, f), encoding="latin1")
    temp["country"] = country
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)
print(df.shape)
print(df["country"].value_counts())




print(df.isnull().sum())
df["description"] = df["description"].fillna("")


df = df.drop_duplicates()

#fixing data types
df["trending_date"] = pd.to_datetime(df["trending_date"], format="%y.%d.%m")
df["publish_time"] = pd.to_datetime(df["publish_time"])
df["category_id"] = df["category_id"].astype("category")

#checking data
df = df[df["views"] >= 0]

print(df.info())
print(df.head())

df.to_json("youtube_cleaned.json", orient="records", lines=True)
