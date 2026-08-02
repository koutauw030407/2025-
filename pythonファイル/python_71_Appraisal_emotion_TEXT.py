from sklearn.metrics import f1_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

model = "chatgpt-4o-latest"
inputting = "FRAMES"

MODELS_FRAMES = [
    "chatgpt-4o-latest",
    "gpt-5", "gpt-5.2-pro",
    "gpt-5.1-chat-latest", "gpt-5.2-chat-latest", "gpt-5-mini", "gpt-5-nano"]
MODELS_OTHERS = ["gpt-4o",  "gpt-4o-mini", "gpt-5.1", "gpt-5.2", "gpt-4.1", "gpt-4.1-mini",
                 "gpt-4.1-nano", "gpt-4-turbo", "gpt-4",  "gpt-3.5-turbo"]

def Appraisal_Emotion(model, inputting):
    # ==== ここは必要に応じてパスを書き換えてください ====
    emotion_path = f"./モデル別ファイル/論文用/{model}/1227_{model}_{inputting}_1.csv"  # 感情カテゴリ
    feature_path = f"./モデル別ファイル/論文用/{model}/1227_{model}_{inputting}_3.csv"  # 14次元

    # ==== 読み込み ====
    emotion_df = pd.read_csv(emotion_path)
    feature_df = pd.read_csv(feature_path)

    # 最初の列名が空欄の場合、自動的に video_id にする
    emotion_df.rename(columns={emotion_df.columns[0]: "video_id"}, inplace=True)
    feature_df.rename(columns={feature_df.columns[0]: "video_id"}, inplace=True)


    # 想定フォーマット：
    # emotion_df: [video_id, emo_1, emo_2, ..., emo_34]
    # feature_df: [video_id, approach, arousal, ..., valence]
    # ※もし列名が違う場合は、print(emotion_df.columns) などで確認して書き換えてください。

    # ==== マージ ====
    # 共通の動画ID列名を仮に "video_id" とします（違う場合は書き換え）
    id_col = "video_id"

    df = pd.merge(emotion_df, feature_df, on=id_col, how="inner")

    # 列の分割
    # ここも実際の列名に合わせて調整してください
    emotion_cols = [c for c in df.columns if c not in [id_col] and c not in [
        "approach", "arousal", "attention", "certainty", "commitment",
        "control", "dominance", "effort", "fairness", "identity",
        "obstruction", "safety", "upswing", "valence"
    ]]
    feature_cols = [
        "approach", "arousal", "attention", "certainty", "commitment",
        "control", "dominance", "effort", "fairness", "identity",
        "obstruction", "safety", "upswing", "valence"
    ]

    feature_df = pd.read_csv(feature_path)

    # 1列目（行名）を feature_name として扱う
    feature_df.rename(
        columns={feature_df.columns[0]: "feature_name"}, inplace=True)

    # 転置して「行＝動画、列＝特徴量」にする
    feature_df = feature_df.set_index("feature_name").T.reset_index()

    # index を video_id にする
    feature_df.rename(columns={"index": "video_id"}, inplace=True)

    # video_id を整数に
    feature_df["video_id"] = feature_df["video_id"].astype(int)

    # 14次元の正しい列名を付け直す
    feature_cols = [
        "approach", "arousal", "attention", "certainty", "commitment",
        "control", "dominance", "effort", "fairness", "identity",
        "obstruction", "safety", "upswing", "valence"
    ]


    emotion_df = pd.read_csv(emotion_path)

    # 1列目（感情名）を index にする
    emotion_df = emotion_df.set_index(emotion_df.columns[0])

    # 転置して「行＝動画、列＝感情」にする
    emotion_df = emotion_df.T

    # index を video_id にする
    emotion_df.index.name = "video_id"
    emotion_df = emotion_df.reset_index()

    # video_id を整数に
    emotion_df["video_id"] = emotion_df["video_id"].astype(int)

    # 1列目が video_id、2列目以降に feature_cols を割り当てる
    feature_df.columns = ["video_id"] + feature_cols

    df = pd.merge(emotion_df, feature_df, on="video_id", how="inner")
    print(df.columns.tolist())

    # df を作った後に emotion_cols と feature_cols を再定義する
    feature_cols = [
        "approach", "arousal", "attention", "certainty", "commitment",
        "control", "dominance", "effort", "fairness", "identity",
        "obstruction", "safety", "upswing", "valence"
    ]

    emotion_cols = [c for c in df.columns if c not in ["video_id"] + feature_cols]

    # ==== 欠番動画の除外 ====
    # 感情がすべて 0 かつ 14次元がすべて 1 の行を削除
    mask_missing = (df[emotion_cols].sum(axis=1) == 0) & (
        df[feature_cols].eq(1).all(axis=1))
    df_clean = df.loc[~mask_missing].copy()


    # ==== 感情スコアの二値化 ====
    # 0 → 0, 1/10/50 → 1
    emotion_bin = (df_clean[emotion_cols] > 0).astype(int)
    df_clean[emotion_cols] = emotion_bin
    mean_rows = []

    for emo in emotion_cols:
        mask = df_clean[emo] == 1
        if mask.sum() == 0:
            # その感情が一度も出ていない場合は NaN にしておく
            mean_vec = {f: np.nan for f in feature_cols}
        else:
            mean_vec = df_clean.loc[mask, feature_cols].mean().to_dict()
        mean_vec["emotion"] = emo
        mean_vec["n_samples"] = mask.sum()
        mean_rows.append(mean_vec)

    mean_df = pd.DataFrame(mean_rows)
    mean_df = mean_df[["emotion", "n_samples"] + feature_cols]
    folder_path = f"./モデル別ファイル/論文用/17_次元/{model}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    mean_df.to_csv(
        f"./モデル別ファイル/論文用/17_次元/{model}/{model}_{inputting}_14次元平均ベクトル.csv", index=False)

    def cohens_d(x1, x0):
        # x1: 感情が出ている動画の値
        # x0: 感情が出ていない動画の値
        n1, n0 = len(x1), len(x0)
        if n1 < 2 or n0 < 2:
            return np.nan
        mean1, mean0 = x1.mean(), x0.mean()
        var1, var0 = x1.var(ddof=1), x0.var(ddof=1)
        s_pooled = np.sqrt(((n1 - 1)*var1 + (n0 - 1)*var0) / (n1 + n0 - 2))
        if s_pooled == 0:
            return np.nan
        return (mean1 - mean0) / s_pooled


    d_rows = []

    for emo in emotion_cols:
        mask1 = df_clean[emo] == 1
        mask0 = df_clean[emo] == 0
        row = {"emotion": emo}
        row["n_1"] = mask1.sum()
        row["n_0"] = mask0.sum()
        for f in feature_cols:
            x1 = df_clean.loc[mask1, f]
            x0 = df_clean.loc[mask0, f]
            row[f] = cohens_d(x1, x0)
        d_rows.append(row)

    d_df = pd.DataFrame(d_rows)
    d_df = d_df[["emotion", "n_1", "n_0"] + feature_cols]

    d_df.to_csv(
        f"./モデル別ファイル/論文用/17_次元/{model}/{model}_{inputting}_Cohens_d.csv", index=False)

    X = df_clean[feature_cols].values
    Y = df_clean[emotion_cols].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, solver="liblinear")
    )
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)

    # 出現数を数える
    emotion_counts = df_clean[emotion_cols].sum(axis=0)

    # 出現数が 0 の感情を除外
    valid_emotions = emotion_counts[emotion_counts > 0].index.tolist()

    # 多ラベル分類用データ
    X = df_clean[feature_cols].values
    Y = df_clean[valid_emotions].values


    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, solver="liblinear")
    )
    clf.fit(X_train, Y_train)

    coef_rows = []

    for i, emo in enumerate(valid_emotions):
        lr = clf.estimators_[i]
        # ConstantPredictor ではないことを保証
        if hasattr(lr, "coef_"):
            coefs = lr.coef_.ravel()
            row = {"emotion": emo}
            for f, c in zip(feature_cols, coefs):
                row[f] = c
            coef_rows.append(row)

    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(
        f"./モデル別ファイル/論文用/17_次元/{model}/{model}_{inputting}_ロジスティック回帰の係数.csv", index=False)

for model in MODELS_FRAMES:
    Appraisal_Emotion(model, "FRAMES")
for model in MODELS_FRAMES + MODELS_OTHERS:
    Appraisal_Emotion(model, "TEXT")
