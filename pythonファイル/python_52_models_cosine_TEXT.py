import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from itertools import combinations

MODELS_FRAMES = [
    "chatgpt-4o-latest",
    "gpt-5", "gpt-5.2-pro",
    "gpt-5.1-chat-latest", "gpt-5.2-chat-latest",
    "gpt-5-mini", "gpt-5-nano"
]
MODELS_OTHERS = ["gpt-4o",  "gpt-4o-mini", "gpt-5.1", "gpt-5.2", "gpt-4.1", "gpt-4.1-mini",
                 "gpt-4.1-nano", "gpt-4-turbo", "gpt-4",  "gpt-3.5-turbo"]


def force_numeric(df):
    df2 = df.copy()
    for col in df2.columns:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")
    return df2


# --- 全モデルの FRAMES データを読み込み、整形 ---
frames_data = {}

for model in MODELS_FRAMES + MODELS_OTHERS:
    df = pd.read_csv(
        f"./モデル別ファイル/論文用/05_FACTOR/05_FACTOR_{model}_TEXT.csv",
        index_col=0
    )

    # 不要行削除
    df = df.drop(index=["primary_emotion", "primary_factor"], errors="ignore")

    # 数値化
    df = force_numeric(df)

    frames_data[model] = df

# --- 全モデルで感情行を揃える（存在しない感情は 0 埋め） ---
all_emotions = pd.Index([])
for df in frames_data.values():
    all_emotions = all_emotions.union(df.index)

for model in MODELS_FRAMES + MODELS_OTHERS:
    frames_data[model] = frames_data[model].reindex(all_emotions, fill_value=0)

# --- 結果をワイド形式で格納する DataFrame ---
wide_df = pd.DataFrame()

# --- モデル間の 21 通りの比較 ---
for model_a, model_b in combinations(MODELS_FRAMES + MODELS_OTHERS, 2):

    dfA = frames_data[model_a]
    dfB = frames_data[model_b]

    # 共通動画番号（列）
    common_cols = dfA.columns.intersection(dfB.columns)

    # 結果を格納する Series（index=video_id）
    distances = {}

    for col in common_cols:
        v1 = dfA[col].values.reshape(1, -1)
        v2 = dfB[col].values.reshape(1, -1)
        dist = cosine_distances(v1, v2)[0][0]
        distances[col] = dist

    # 列名（モデルペア）
    col_name = f"{model_a} / {model_b}"

    # wide_df に追加（存在しない動画番号は自動で NaN になる）
    wide_df[col_name] = pd.Series(distances)

# --- CSV 保存 ---
save_path = "./モデル別ファイル/論文用/51_models/51_FT_cosine_TEXT.csv"
wide_df.to_csv(save_path)

print(f"完了 → {save_path}")
