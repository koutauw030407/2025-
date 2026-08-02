import pandas as pd
from collections import Counter

# CSV 読み込み
df = pd.read_csv("./モデル別ファイル/論文用/13_RDM/emotion_groups_all_models.csv")

counter = Counter()

# 全セルを走査して、文字列そのものをカウント
for col in df.columns:
    for val in df[col].dropna():
        val_str = str(val).strip()
        # 空欄や "" の場合はスキップ
        if val_str == "":
            continue
        counter[val_str] += 1

# 結果表示
for combo, count in counter.most_common():
    print(f"{combo}: {count}")
