import pandas as pd

# CSV を読み込む
df = pd.read_csv(f"./モデル別ファイル/論文用/15_評点差が大きい動画に対する評価/フレーム・テキスト間の動画のコサイン距離/61_FT_1.csv")

# すべての数値を 1 本の Series にまとめる
values = df.values.flatten()

# NaN を除外
values = pd.Series(values).dropna()

# 出現回数をカウント
counts = values.value_counts()

# 5 回以上出現したものだけ抽出
result = counts[counts >= 5]

print(result)
