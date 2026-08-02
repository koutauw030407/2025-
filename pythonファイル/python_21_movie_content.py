from functools import reduce
import pandas as pd
import os

MODELS_FRAMES = [
    "chatgpt-4o-latest", "gpt-5", "gpt-5.2-pro",
    "gpt-5.1-chat-latest", "gpt-5.2-chat-latest",
    "gpt-5-mini", "gpt-5-nano"
]
MODELS_OTHERS = ["gpt-4o",  "gpt-4o-mini", "gpt-5.1", "gpt-5.2", "gpt-4.1", "gpt-4.1-mini",
                 "gpt-4.1-nano", "gpt-4-turbo", "gpt-4",  "gpt-3.5-turbo"]

TARGET_VIDS = ["1960", "1768", "1141", "1744", "2159"]

wide_blocks = []

for model in MODELS_FRAMES + MODELS_OTHERS:
    path = f"./モデル別ファイル/論文用/{model}/1227_{model}_TEXT_1.csv"
    df = pd.read_csv(path)

    emotion_col = df.columns[0]
    df = df.rename(columns={emotion_col: "emotion"})

    df_sel = df[["emotion"] + TARGET_VIDS].copy()

    # MultiIndex 用に列名を (動画番号, モデル名) に変換
    df_sel = df_sel.rename(columns={vid: (vid, model) for vid in TARGET_VIDS})

    wide_blocks.append(df_sel)

# emotion をキーに横結合
scores_all = reduce(lambda left, right: pd.merge(
    left, right, on="emotion"), wide_blocks)

# emotion を index に
scores_all = scores_all.set_index("emotion")

# MultiIndex を構築
scores_all.columns = pd.MultiIndex.from_tuples(
    scores_all.columns, names=["video", "model"])

# 列順を 417→441→466 の順に並べ替え
scores_all = scores_all.reindex(columns=TARGET_VIDS, level=0)

scores_all.to_csv("./モデル別ファイル/論文用/15_動画ごとの差分/scores_TEXT.csv", index=False)

"""wide_blocks2 = []

for model in MODELS_FRAMES + MODELS_OTHERS:
    path = f"./モデル別ファイル/論文用/reason_csv/02_{model}_TEXT_reason.csv"
    df = pd.read_csv(path)

    item_col = df.columns[0]
    df = df.rename(columns={item_col: "item"})

    df_sel = df[["item"] + TARGET_VIDS].copy()

    df_sel = df_sel.rename(columns={vid: (vid, model) for vid in TARGET_VIDS})

    wide_blocks2.append(df_sel)

reasons_all = reduce(lambda left, right: pd.merge(
    left, right, on="item"), wide_blocks2)

reasons_all = reasons_all.set_index("item")
reasons_all.columns = pd.MultiIndex.from_tuples(
    reasons_all.columns, names=["video", "model"])
reasons_all = reasons_all.reindex(columns=TARGET_VIDS, level=0)

reasons_all.to_csv("./モデル別ファイル/論文用/15_動画ごとの差分/reasons_TEXT.csv", index=False)

"""
