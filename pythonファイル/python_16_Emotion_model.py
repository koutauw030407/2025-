
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

MODELS_FRAMES = [
    "chatgpt-4o-latest",
    "gpt-5", "gpt-5.2-pro",
    "gpt-5.1-chat-latest", "gpt-5.2-chat-latest",
    "gpt-5-mini", "gpt-5-nano"
]
MODELS_OTHERS = ["gpt-4o",  "gpt-4o-mini", "gpt-5.1", "gpt-5.2", "gpt-4.1", "gpt-4.1-mini",
                 "gpt-4.1-nano", "gpt-4-turbo", "gpt-4",  "gpt-3.5-turbo"]


def Emotion_model_analysis(MODELS, inputting):

    # 1. CSV読み込み
    model_data = {}
    for model in MODELS:
        path = f"./モデル別ファイル/論文用/{model}/1227_{model}_{inputting}_1.csv"
        df = pd.read_csv(path, index_col=0)
        model_data[model] = df

    # 2. 全モデル共通の動画（列）を抽出
    common_videos = set(model_data[MODELS[0]].columns)
    for model in MODELS[1:]:
        common_videos &= set(model_data[model].columns)
    common_videos = sorted(list(common_videos))

    # 3. 非ゼロスコアが1つでもある動画だけを残す
    valid_videos = []
    for vid in common_videos:
        keep = True
        for model in MODELS:
            col = model_data[model][vid]
            if (col != 0).sum() == 0:
                keep = False
                break
        if keep:
            valid_videos.append(vid)

    print("共通して存在し、かつ非ゼロスコアを持つ動画数:", len(valid_videos))

    # 4. 感情カテゴリごとにモデル間距離を計算（動画は valid_videos のみ）
    records = []
    distance_results = {}

    emotions = model_data[MODELS[0]].index  # 34感情

    for emo in emotions:
        row_vectors = []
        for model in MODELS:
            vec = model_data[model].loc[emo][valid_videos].values  # ★ M本の動画に限定
            row_vectors.append(vec)

        mat = np.array(row_vectors)
        dist = squareform(pdist(mat, metric="cosine"))
        dist_df = pd.DataFrame(dist, index=MODELS, columns=MODELS)
        distance_results[emo] = dist_df

        total_diff = dist_df.sum(axis=1)

        for model in MODELS:
            records.append({
                "emotion": emo,
                "model": model,
                "total_difference": total_diff[model]
            })

    # 5. 距離の強い順にソート
    diff_df = pd.DataFrame(records)
    ranked_df = diff_df.sort_values("total_difference", ascending=False)

    # 6. 感情ごとの平均スコア（動画は M 本に限定）
    mean_scores = {
        model: model_data[model][valid_videos].mean(axis=1)
        for model in MODELS
    }
    mean_scores_df = pd.DataFrame(mean_scores)

    # 7. 感情ごとの順位付け
    rank_df = mean_scores_df.rank(axis=1, ascending=False)
    rank_df_path = f"./モデル別ファイル/論文用/16_感情検出度合い/感情の平均スコアの順位_{inputting}_filtered.csv"
    rank_df.to_csv(rank_df_path, encoding="utf-8-sig")

    # 8. ロング形式に変換して結合
    rank_long = rank_df.stack().rename("rank").reset_index()
    rank_long.columns = ["emotion", "model", "rank"]
    ranked_df_2 = ranked_df.merge(
        rank_long, on=["emotion", "model"], how="left")

    mean_long = mean_scores_df.stack().rename("mean_score").reset_index()
    mean_long.columns = ["emotion", "model", "mean_score"]

    ranked_df_3 = ranked_df_2.merge(
        mean_long, on=["emotion", "model"], how="left")

    ranked_df_3_path = f"./モデル別ファイル/論文用/16_感情検出度合い/ranked_df_{inputting}_filtered.csv"
    ranked_df_3.to_csv(ranked_df_3_path, encoding="utf-8-sig")

    # 9. 感情ごとのモデル間距離の平均（小さいほど一致）
    emotion_consistency = {}
    for emo, dist_df in distance_results.items():
        consistency = dist_df.values[np.triu_indices(len(MODELS), k=1)].mean()
        emotion_consistency[emo] = consistency

    consistency_df = pd.DataFrame.from_dict(
        emotion_consistency, orient="index", columns=["mean_distance"]
    )
    consistency_df_path = f"./モデル別ファイル/論文用/16_感情検出度合い/consistency_df_{inputting}_filtered.csv"
    consistency_df.to_csv(consistency_df_path, encoding="utf-8-sig")

    # 10. z-score と Strong/Divergent 判定
    df = ranked_df_3.copy()
    df["z_diff"] = (df["total_difference"] -
                    df["total_difference"].mean()) / df["total_difference"].std()

    def strength_category(rank, z):
        if rank <= len(MODELS)*0.3 and z >= 0.5:
            return "Strong/Divergent"
        else:
            return None

    df["strength"] = df.apply(lambda row: strength_category(
        row["rank"], row["z_diff"]), axis=1)

    df_path = f"./モデル別ファイル/論文用/16_感情検出度合い/16_感情検出度合い_{inputting}_filtered.csv"
    df.to_csv(df_path, encoding="utf-8-sig")

Emotion_model_analysis(MODELS_FRAMES, "FRAMES")
Emotion_model_analysis(MODELS_FRAMES + MODELS_OTHERS, "TEXT")
