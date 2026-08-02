import pandas as pd
import numpy as np
import os

# -----------------------------
# モデル名の設定（必要に応じて変更）
# -----------------------------
MODELS_FRAMES = [
    "chatgpt-4o-latest",
    "gpt-5", "gpt-5.2-pro",
    "gpt-5.1-chat-latest", "gpt-5.2-chat-latest", "gpt-5-mini", "gpt-5-nano"]
MODELS_OTHERS = ["gpt-4o",  "gpt-4o-mini", "gpt-5.1", "gpt-5.2", "gpt-4.1", "gpt-4.1-mini",
                 "gpt-4.1-nano", "gpt-4-turbo", "gpt-4",  "gpt-3.5-turbo"]

INPUT_TYPES = ["FRAMES", "TEXT"]

# 14次元の列名
feature_cols = [
    "approach", "arousal", "attention", "certainty", "commitment",
    "control", "dominance", "effort", "fairness", "identity",
    "obstruction", "safety", "upswing", "valence"
]
emotion_order = [
    "Admiration", "Adoration", "Aesthetic-Appreciation", "Amusement", "Anger", "Anxiety",
    "Awe", "Awkwardness", "Boredom", "Calmness", "Confusion", "Contempt", "Craving",
    "Disappointment", "Disgust", "Empathic-Pain", "Entrancement", "Envy", "Excitement",
    "Fear", "Guilt", "Horror", "Interest", "Joy", "Nostalgia", "Pride", "Relief", "Romance",
    "Sadness", "Satisfaction", "Sexual-Desire", "Surprise", "Sympathy", "Triumph"
]

# 結果を格納する辞書
result1 = {}
result2 = {}

# -----------------------------
# 各モデル × FRAMES/TEXT のループ
# -----------------------------

def Appraisal_Emotion_A(model, inputting):
    # ファイルパス
    path1 = f"./モデル別ファイル/論文用/17_次元/{model}/{model}_{inputting}_Cohens_d.csv"
    path2 = f"./モデル別ファイル/論文用/17_次元/{model}/{model}_{inputting}_ロジスティック回帰の係数.csv"

    if not os.path.exists(path1):
        print(f"ファイルが見つかりません: {path1}")
        return None
    if not os.path.exists(path2):
        print(f"ファイルが見つかりません: {path2}")
        return None

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    # df2: ロジスティック回帰の係数（emotion 列を含む DataFrame）
    # feature_cols: 14次元の列名
    # emotion_order: あなたが指定した34感情の順序


    # emotion を index にする
    df2_indexed = df2.set_index("emotion")

    # emotion_order に合わせて並べ替え、欠けている感情は 0 で埋める
    df2_fixed = df2_indexed.reindex(emotion_order).fillna(0)

    # 必要なら index を戻す
    df2_fixed = df2_fixed.reset_index()

    # |d| > 1 の次元数を数える
    counts1 = (df1[feature_cols].abs() > 1).sum(axis=1)
    counts2 = (df2_fixed[feature_cols].abs() > 1).sum(axis=1)

    # 結果を保存
    col_name = f"{model}_{inputting}"
    result1[col_name] = counts1.values
    result2[col_name] = counts2.values

    # emotion 名も保存（最初の1回だけ）
    if "emotion" not in result1:
        result1["emotion"] = df1["emotion"].values
    if "emotion" not in result2:
        result2["emotion"] = df2_fixed["emotion"].values


for model in MODELS_FRAMES:
    Appraisal_Emotion_A(model, "FRAMES")
for model in MODELS_FRAMES + MODELS_OTHERS:
    Appraisal_Emotion_A(model, "TEXT")

# -----------------------------
# 結果を DataFrame にまとめる
# -----------------------------

result1_df = pd.DataFrame(result1)
result2_df = pd.DataFrame(result2)

# emotion を先頭列に
cols1 = ["emotion"] + [c for c in result1_df.columns if c != "emotion"]
result1_df = result1_df[cols1]

# emotion を先頭列に
cols2 = ["emotion"] + [c for c in result2_df.columns if c != "emotion"]
result2_df = result2_df[cols2]

# -----------------------------
# CSV に保存
# -----------------------------
result1_df.to_csv(
    "./モデル別ファイル/論文用/17_次元/Cohens_d_over1_summary.csv", index=False)
result2_df.to_csv(
    "./モデル別ファイル/論文用/17_次元/ロジスティック回帰の係数_over1_summary.csv", index=False)
