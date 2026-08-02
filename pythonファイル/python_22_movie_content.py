from functools import reduce
import pandas as pd

MODELS_FRAMES = [
    "chatgpt-4o-latest", "gpt-5", "gpt-5.2-pro",
    "gpt-5.1-chat-latest", "gpt-5.2-chat-latest",
    "gpt-5-mini", "gpt-5-nano"
]

TARGET_VIDS = ["1650", "632", "1197", "1724", "1217", "1114", "1549", "522", "1010", "1094", "1520", "155", "1954", "1990", "338", "765", "420", "1442", "1406", "1869", "613", "2151", "1739", "917", "1741", "1563", "266", "239", "2120", "474", "547", "1173", "721", "1575", "24", "1593", "56", "995", "1414", "1643", "513", "232", "734"]

wide_blocks = []

for model in MODELS_FRAMES:
    # FRAMES
    path0 = f"./モデル別ファイル/論文用/{model}/1227_{model}_FRAMES_1.csv"
    df0 = pd.read_csv(path0)
    df0 = df0.rename(columns={df0.columns[0]: "emotion"}).set_index("emotion")
    df0 = df0[TARGET_VIDS]
    df0.columns = pd.MultiIndex.from_product([TARGET_VIDS, [model]])
    wide_blocks.append(df0)

    # TEXT
    path1 = f"./モデル別ファイル/論文用/{model}/1227_{model}_TEXT_1.csv"
    df1 = pd.read_csv(path1)
    df1 = df1.rename(columns={df1.columns[0]: "emotion"}).set_index("emotion")
    df1 = df1[TARGET_VIDS]
    df1.columns = pd.MultiIndex.from_product([TARGET_VIDS, [model]])
    wide_blocks.append(df1)

# 列を壊さず横結合
scores_all = pd.concat(wide_blocks, axis=1)

# 欲しい列順の MultiIndex を自分で定義
desired_cols = pd.MultiIndex.from_product(
    [TARGET_VIDS, MODELS_FRAMES], names=["video", "model"]
)

# 存在する列だけに絞って並べ替え
existing_cols = [c for c in desired_cols if c in scores_all.columns]
scores_all = scores_all[existing_cols]

scores_all.to_csv("./モデル別ファイル/論文用/15_評点差が大きい動画に対する評価/scores_FT.csv", index=False)
