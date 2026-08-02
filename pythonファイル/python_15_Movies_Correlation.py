from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

MODELS_FRAMES = ["chatgpt-4o-latest",  "gpt-5", "gpt-5.2-pro", "gpt-5.1-chat-latest", "gpt-5.2-chat-latest", "gpt-5-mini", "gpt-5-nano"]
MODELS_OTHERS = ["gpt-4o",  "gpt-4o-mini", "gpt-5.1", "gpt-5.2", "gpt-4.1", "gpt-4.1-mini","gpt-4.1-nano", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]

ALL_EMOTIONS = [
    "Admiration", "Adoration", "Aesthetic-Appreciation", "Amusement", "Anger",
    "Anxiety", "Awe", "Awkwardness", "Boredom", "Calmness", "Confusion", "Contempt",
    "Craving", "Disappointment", "Disgust", "Empathic-Pain", "Entrancement", "Envy",
    "Excitement", "Fear", "Guilt", "Horror", "Interest", "Joy", "Nostalgia", "Pride",
    "Relief", "Romance", "Sadness", "Satisfaction", "Sexual-Desire", "Surprise",
    "Sympathy", "Triumph"
]
ALL_EMOTIONS = [e.lower() for e in ALL_EMOTIONS]

def get_top_with_ties(df, col, n=5):
    # 上位 n 位までの値を取得
    top_n = df.nlargest(n, col)
    threshold = top_n[col].min()  # 5位の値

    # 5位と同点のものをすべて含める
    result = df[df[col] >= threshold].sort_values(col, ascending=False)
    return result


def get_bottom_with_ties(df, col, n=5):
    bottom_n = df.nsmallest(n, col)
    threshold = bottom_n[col].max()  # 下位5位の値（最大値）

    result = df[df[col] <= threshold].sort_values(col, ascending=True)
    return result


def compute_js_disagreement(df_scores, models):
    """
    df_scores: shape = (34 emotions, M models)
    models: list of model names (columns in df_scores)
    """
    # 正規化して確率分布にする
    P = df_scores[models].values.astype(float)
    P = P / (P.sum(axis=0, keepdims=True) + 1e-12)  # shape = (34, M)

    M = len(models)
    js_values = []

    # 全モデルペアの JS 距離
    for i in range(M):
        for j in range(i+1, M):
            p = P[:, i]
            q = P[:, j]
            js = jensenshannon(p, q)  # 0〜1 の距離
            js_values.append(js)

    # 平均 JS 距離（DisagreementScore）
    return float(np.mean(js_values))


def load_emotion_scores(path):
    df = pd.read_csv(path, index_col=0)

    # 行名を文字列に統一
    df.index = df.index.astype(str).str.strip().str.lower()

    # 列名（動画ID）を数字だけにフィルタ
    video_ids = [col for col in df.columns if str(col).isdigit()]
    df = df[video_ids]

    # 欠けている感情を 0 行で補完
    for emo in ALL_EMOTIONS:
        if emo not in df.index:
            df.loc[emo] = 0.0

    # 行の順序を揃える
    df = df.loc[ALL_EMOTIONS]

    return df


def compute_emotion_disagreement(MODELS, input):
    dfs = {}
    for model in MODELS:
        path = f"./モデル別ファイル/論文用/05_FACTOR/05_FACTOR_{model}_{input}.csv"
        df = load_emotion_scores(path)
        dfs[model] = df

    # 全動画IDの union
    all_ids = sorted(set().union(*[set(df.columns) for df in dfs.values()]))

    disagreement_scores = {}

    for vid in all_ids:
        # この動画IDを持っているモデルだけを使う
        available_models = [m for m in MODELS if vid in dfs[m].columns]

        if len(available_models) < 2:
            continue

        # df_vid を作成（感情 × モデル）
        df_vid = pd.DataFrame({
            model: dfs[model][vid] for model in available_models
        }, index=ALL_EMOTIONS)

        score = compute_js_disagreement(df_vid, available_models)
        disagreement_scores[vid] = score

    return disagreement_scores

def load_all_reprs():
    repr_frames = {}
    repr_text = {}
    id_sets = []

    for model in MODELS_FRAMES:
        path_f = f"./モデル別ファイル/バックアップ/{model}/1227_{model}_FRAMES_1.csv"
        path_t = f"./モデル別ファイル/論文用/{model}/1227_{model}_TEXT_1.csv"
        
        df_factor_f = pd.read_csv(
            f"./モデル別ファイル/論文用/05_FACTOR/05_FACTOR_{model}_FRAMES.csv",
            index_col=0
        )
        df_factor_t = pd.read_csv(
            f"./モデル別ファイル/論文用/05_FACTOR/05_FACTOR_{model}_TEXT.csv",
            index_col=0
        )

        # 列名のうち「数値だけ」を抽出
        ids_f = [int(c) for c in df_factor_f.columns if str(c).isdigit()]
        ids_t = [int(c) for c in df_factor_t.columns if str(c).isdigit()]

        # 1〜2185 の範囲に限定
        ids_f = [vid for vid in ids_f if 1 <= vid <= 2185]
        ids_t = [vid for vid in ids_t if 1 <= vid <= 2185]

        # FRAMES と TEXT の共通ID
        valid_ids = sorted(list(set(ids_f).intersection(ids_t)))

        df_f = pd.read_csv(path_f, index_col=0)
        df_t = pd.read_csv(path_t, index_col=0)

        # 列名（動画ID）だけを抽出
        df_f = df_f.loc[:, df_f.columns.str.isdigit()]
        df_f.columns = df_f.columns.astype(int)

        df_t = df_t.loc[:, df_t.columns.str.isdigit()]
        df_t.columns = df_t.columns.astype(int)

        # valid_ids のみに揃える
        df_f = df_f.loc[:, valid_ids]
        df_t = df_t.loc[:, valid_ids]

        id_sets.append(set(df_f.columns).intersection(df_t.columns))
        repr_frames[model] = df_f
        repr_text[model] = df_t

    # 7モデルすべてに共通する動画ID
    common_ids = sorted(list(set.intersection(*id_sets)))

    # 共通IDに揃える
    for model in MODELS_FRAMES:
        repr_frames[model] = repr_frames[model].loc[:, common_ids]
        repr_text[model] = repr_text[model].loc[:, common_ids]

    return repr_frames, repr_text, common_ids


def compute_video_similarity(repr_frames, repr_text, common_ids):
    scores = {}

    for vid in common_ids:
        per_model_corr = []

        for model in MODELS_FRAMES:
            v_f = repr_frames[model].loc[:, vid].values.reshape(1, -1)
            v_t = repr_text[model].loc[:, vid].values.reshape(1, -1)

            corr = cosine_similarity(v_f, v_t)[0, 0]
            per_model_corr.append(corr)

        # 7モデルの平均相関
        scores[vid] = float(np.mean(per_model_corr))

    return scores


# === モデル間の感情解釈のズレ（JS Disagreement）を計算 ===

def JS_disagreement_csv(MODELS, inputting):
    LEN_MODELS = len(MODELS)
    dis_scores = compute_emotion_disagreement(MODELS, inputting)
    df_dis_scores = pd.DataFrame({
        "video_id": list(dis_scores.keys()),
        "disagreement": list(dis_scores.values())
    })
    df_dis_scores.to_csv(
        f"./モデル別ファイル/論文用/15_動画の相関/emotion_disagreement_{inputting}_JS_{LEN_MODELS}.csv", index=False)


JS_disagreement_csv(MODELS_FRAMES, "FRAMES")
JS_disagreement_csv(MODELS_FRAMES, "TEXT")
JS_disagreement_csv((MODELS_FRAMES +  MODELS_OTHERS), "TEXT")


df_frames = pd.read_csv(
    "./モデル別ファイル/論文用/15_動画の相関/emotion_disagreement_FRAMES_JS_7.csv")
df_text = pd.read_csv(
    "./モデル別ファイル/論文用/15_動画の相関/emotion_disagreement_TEXT_JS_7.csv")
df_all = pd.read_csv(
    "./モデル別ファイル/論文用/15_動画の相関/emotion_disagreement_TEXT_JS_17.csv")

df = df_frames.merge(df_text, on="video_id", how="outer",
                     suffixes=("_frames", "_text"))
df = df.merge(df_all, on="video_id", how="outer")
df.rename(columns={"disagreement": "disagreement_all"}, inplace=True)

for col in ["disagreement_frames", "disagreement_text", "disagreement_all"]:
    mean = df[col].mean()
    std = df[col].std()
    df[f"z_{col}"] = (df[col] - mean) / std

df["Z_total"] = df["z_disagreement_frames"] + \
    df["z_disagreement_text"] + df["z_disagreement_all"]

top5 = get_top_with_ties(df, "Z_total", n=5)
bottom5 = get_bottom_with_ties(df, "Z_total", n=5)

top5.to_csv(
    f"./モデル別ファイル/論文用/15_動画の相関/top5.csv", index=False)

bottom5.to_csv(
    f"./モデル別ファイル/論文用/15_動画の相関/bottom5.csv", index=False)
