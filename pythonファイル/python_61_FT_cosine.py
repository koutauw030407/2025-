import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

MODELS_FRAMES = [
    "chatgpt-4o-latest",
    "gpt-5", "gpt-5.2-pro",
    "gpt-5.1-chat-latest", "gpt-5.2-chat-latest",
    "gpt-5-mini", "gpt-5-nano"
]


def force_numeric(df):
    df2 = df.copy()
    for col in df2.columns:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")
    return df2


for model in MODELS_FRAMES:

    # --- 読み込み ---
    df_FRAMES = pd.read_csv(
        f"./モデル別ファイル/論文用/05_FACTOR/05_FACTOR_{model}_FRAMES.csv",
        index_col=0
    )
    df_TEXT = pd.read_csv(
        f"./モデル別ファイル/論文用/05_FACTOR/05_FACTOR_{model}_TEXT.csv",
        index_col=0
    )

    # --- primary_emotion / primary_factor を削除 ---
    rows_to_drop = ["primary_emotion", "primary_factor"]
    df_FRAMES = df_FRAMES.drop(index=rows_to_drop, errors="ignore")
    df_TEXT = df_TEXT.drop(index=rows_to_drop, errors="ignore")

    # --- 数値化 ---
    df_FRAMES = force_numeric(df_FRAMES)
    df_TEXT = force_numeric(df_TEXT)

    # --- 共通の動画番号（列名）を抽出 ---
    common_cols = df_FRAMES.columns.intersection(df_TEXT.columns)
    # --- 行（感情）を揃える：存在しない感情は 0 で埋める ---
    all_emotions = df_FRAMES.index.union(df_TEXT.index)

    df_FRAMES = df_FRAMES.reindex(all_emotions, fill_value=0)
    df_TEXT = df_TEXT.reindex(all_emotions, fill_value=0)

    # --- 各動画番号についてコサイン距離を計算 ---
    distances = []
    for col in common_cols:
        v1 = df_FRAMES[col].values.reshape(1, -1)
        v2 = df_TEXT[col].values.reshape(1, -1)
        dist = cosine_distances(v1, v2)[0][0]
        distances.append(dist)

    # --- 結果を保存 ---
    df_result = pd.DataFrame({
        "video_id": common_cols,
        "cosine_distance": distances
    })

    save_path = f"./モデル別ファイル/論文用/61_FT/61_FT_cosine_{model}.csv"
    df_result.to_csv(save_path, index=False)

    print(f"{model}: 完了 → {save_path}")
