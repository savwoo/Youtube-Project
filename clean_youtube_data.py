import pandas as pd

df = pd.read_json("youtube_cleaned.json", orient="records", lines=True)

df_small = df.head(100)

df_small.to_json("clean_youtube_data.json", orient="records", lines=True)

print(df_small.shape)
print(df_small.head())
