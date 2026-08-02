
import ot
import os
import re
import sys
import logging
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colors import BoundaryNorm, ListedColormap

plt.rcParams["font.family"] = "Meiryo"

emotion_order = [
    "Admiration", "Adoration", "Aesthetic-Appreciation", "Amusement", "Anger", "Anxiety",
    "Awe", "Awkwardness", "Boredom", "Calmness", "Confusion", "Contempt", "Craving",
    "Disappointment", "Disgust", "Empathic-Pain", "Entrancement", "Envy", "Excitement",
    "Fear", "Guilt", "Horror", "Interest", "Joy", "Nostalgia", "Pride", "Relief", "Romance",
    "Sadness", "Satisfaction", "Sexual-Desire", "Surprise", "Sympathy", "Triumph"
]

MODELS_FRAMES = ["chatgpt-4o-latest", "gpt-5", "gpt-5.2-pro",
                 "gpt-5.1-chat-latest", "gpt-5.2-chat-latest", "gpt-5-mini", "gpt-5-nano"]
MODELS_OTHERS = ["gpt-4o",  "gpt-4o-mini","gpt-5.1", "gpt-5.2","gpt-4.1", "gpt-4.1-mini",
                 "gpt-4.1-nano", "gpt-4-turbo", "gpt-4",  "gpt-3.5-turbo"]

factor_color_map = {
    'F1': '#DC270B',
    'F2': '#AB1C87',
    'F3': '#0074BE',
    'F4': '#00A63D',
    'F5': "#EC75A9",
    'F6': "#B34422",
    'F7': "#BCD400",
    'F8': '#F5A000',
    'F9': "#00A7E3",
    'F10': "#000575",
    'F11': "#FF00D9",
    'F12': "#8400FF",
    'F13': "#00838A",
    'F14': "#004516",
    'F15': "#FFAFD1",
    'F16': "#511200",
    'F17': "#F4F000",
    'F18': "#F06400",
    'F19': "#D7C800",
    'F20': "#7A7A7A",
}
default_color = "#FFFFFF"
results_dim_and_pairs = {}

embedding_dims = {
    ("chatgpt-4o-latest", "FRAMES"): 16,
    ("chatgpt-4o-latest", "TEXT"): 16,

    ("gpt-5", "FRAMES"): 16,
    ("gpt-5", "TEXT"): 16,

    ("gpt-5.2-pro", "FRAMES"): 17,
    ("gpt-5.2-pro", "TEXT"): 17,

    ("gpt-5.1-chat-latest", "FRAMES"): 16,
    ("gpt-5.1-chat-latest", "TEXT"): 17,

    ("gpt-5.2-chat-latest", "FRAMES"): 17,
    ("gpt-5.2-chat-latest", "TEXT"): 17,

    ("gpt-5-mini", "FRAMES"): 17,
    ("gpt-5-mini", "TEXT"): 17,

    ("gpt-5-nano", "FRAMES"): 18,
    ("gpt-5-nano", "TEXT"): 17,

    ("gpt-5.1", "TEXT"): 16,
    ("gpt-5.2", "TEXT"): 17,

    ("gpt-4o", "TEXT"): 17,
    ("gpt-4o-mini", "TEXT"): 17,

    ("gpt-4.1", "TEXT"): 18,
    ("gpt-4.1-mini", "TEXT"): 17,
    ("gpt-4.1-nano", "TEXT"): 16,

    ("gpt-4-turbo", "TEXT"): 18,
    ("gpt-4", "TEXT"): 16,

    ("gpt-3.5-turbo", "TEXT"): 17,
}

def load_factor_map(path):
    df = pd.read_csv(path)
    # emotion → factor の辞書にする
    return dict(zip(df["emotion"], df["factor"]))


def build_factor_main(df_base):
    # primary_emotion: 各動画の代表感情
    primary_emotion = df_base.loc["primary_emotion"]

    # primary_factor: 各動画の因子
    primary_factor = df_base.loc["primary_factor"]

    factor_to_emotions = {}
    for vid, fac in primary_factor.items():
        emo = primary_emotion[vid]
        factor_to_emotions.setdefault(fac, []).append(emo)

    # 最頻出の感情を「因子の代表感情」とする
    factor_main = {}
    for fac, emos in factor_to_emotions.items():
        factor_main[fac] = max(set(emos), key=emos.count)

    return factor_main


def compute_direct_match_with_factor(
        ids,
        primary_emotion_A, primary_emotion_B,
        factor_map_A, factor_map_B):

    matches = []

    for vid in ids:
        emoA = primary_emotion_A.get(vid)
        emoB = primary_emotion_B.get(vid)

        if emoA is None or emoB is None:
            continue

        # 条件1：感情カテゴリが同じ
        cond1 = (emoA == emoB)

        # 条件2：因子マップで同じ因子に属する
        facA = factor_map_A.get(emoA)
        facB = factor_map_B.get(emoB)
        cond2 = (facA is not None and facB is not None and facA == facB)

        matches.append(cond1 or cond2)

    return sum(matches) / len(matches) if matches else 0.0



def get_color_for_factor(factor):
    # 例: F4-1 → F4
    base = factor.split('-')[0]   # "-1" を除去

    # base が F14, F24 のような場合 → 末尾の数字を取る
    m = re.match(r"F(\d+)", base)
    if not m:
        return default_color

    num = int(m.group(1))

    # 末尾の数字（1桁）を取得
    last_digit = num % 10   # 14 → 4, 24 → 4

    # F4 の色を使う
    key = f"F{last_digit}"

    return factor_color_map.get(key, default_color)


def sort_ids_by_factor(ids, primary_factor):
    factor_to_ids = defaultdict(list)
    for vid in ids:
        factor = primary_factor.get(vid)
        if factor:
            factor_to_ids[factor].append(vid)

    # 因子順に並べる（F1, F2, ..., F9, F0）
    factor_order = sorted(factor_to_ids.keys(), key=lambda f: int(f[1:]) % 10)

    sorted_ids = []
    for f in factor_order:
        sorted_ids.extend(sorted(factor_to_ids[f]))
    return sorted_ids


def map_factors_by_meaning(factor_main_A, factor_main_B):
    mapping = {}
    for fA, emoA in factor_main_A.items():
        best = None
        for fB, emoB in factor_main_B.items():
            if emoA == emoB:
                best = fB
                break
        mapping[fA] = best
    return mapping


def compute_rdm_cosine(df_repr):
    X = df_repr.values.astype(np.float32)
    S = cosine_similarity(X)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return D

def normalize_base_columns(df_base):
    # 1. 現在の列名（すべて文字列）
    cols = df_base.columns.astype(str)

    # 2. 数字だけの列名を抽出
    numeric_cols = [c for c in cols if c.isdigit()]

    # 3. 数字列だけ int に変換
    numeric_cols_int = list(map(int, numeric_cols))

    # 4. 非数値列（count_50 など）はそのまま残す
    other_cols = [c for c in cols if not c.isdigit()]

    # 5. 新しい列名リストを作る
    new_columns = numeric_cols_int + other_cols

    # 6. df_base の列名を置き換える
    df_base.columns = new_columns

    return df_base


def compute_factor_boundaries(sorted_ids, primary_factor):
    boundaries = []
    prev_factor = primary_factor[sorted_ids[0]]
    for i, vid in enumerate(sorted_ids):
        f = primary_factor[vid]
        if f != prev_factor:
            boundaries.append(i)
            prev_factor = f
    return boundaries


def compute_factor_match_stats(
        T_sorted, sorted_ids_A, sorted_ids_B,
        primary_emotion_A, primary_emotion_B,
        factor_map_A, factor_map_B):

    match_B = np.argmax(T_sorted, axis=1)
    matched_ids_B = [sorted_ids_B[j] for j in match_B]

    matches = []

    for vidA, vidB in zip(sorted_ids_A, matched_ids_B):
        emoA = primary_emotion_A.get(vidA)
        emoB = primary_emotion_B.get(vidB)
        if emoA is None or emoB is None:
            continue

        # 条件1：感情カテゴリが同じ
        cond1 = (emoA == emoB)

        # 条件2：両モデルの factor_map で同じ因子に属する
        facA = factor_map_A.get(emoA)
        facB = factor_map_B.get(emoB)
        cond2 = (facA is not None and facB is not None and facA == facB)

        # どちらか満たせば一致
        matches.append(cond1 or cond2)

    if matches:
        return sum(matches) / len(matches)
    else:
        return 0.0

def make_gwot_filename(model1, model2, inputting1, inputting2, make_gwot_filename):
    if inputting1 == inputting2:
        return f"{inputting1}/14_GWOT_{inputting1}_{model1}_{model2}_{make_gwot_filename}.png"
    elif model1 == model2:
        return f"FRAMES_TEXT/14_GWOT_{inputting1}_{inputting2}_{model1}_{make_gwot_filename}.png"
    else:
        return f"14_GWOT_{model1}_{inputting1}_{model2}_{inputting2}_{make_gwot_filename}.png"


def build_factor_order(factor_to_emotions, corr_matrix, threshold=0.9):
    remaining = set(factor_to_emotions.keys())
    order = []

    # まず最初の因子は F1 など、番号順で開始
    current = sorted(remaining, key=lambda f: int(f[1:]))[0]
    order.append(current)
    remaining.remove(current)

    while remaining:
        # current 因子のカテゴリ
        emos = factor_to_emotions[current]

        # current 因子に最も近い「別因子」を探す
        best_factor = None
        best_corr = -1

        for emo in emos:
            # このカテゴリと全カテゴリの相関
            row = corr_matrix.loc[emo]

            # まだ出ていない因子のカテゴリだけを見る
            for f in remaining:
                for emo2 in factor_to_emotions[f]:
                    c = row[emo2]
                    if c > best_corr:
                        best_corr = c
                        best_factor = f

        # 閾値以上ならその因子を次に置く
        if best_corr >= threshold:
            order.append(best_factor)
            remaining.remove(best_factor)
            current = best_factor
        else:
            # 閾値を満たす因子がない場合は、残りの中で最小番号を置く
            next_factor = sorted(remaining, key=lambda f: int(f[1:]))[0]
            order.append(next_factor)
            remaining.remove(next_factor)
            current = next_factor

    return order


def find_high_distance_emotion_pairs(
        D, sorted_ids, primary_emotion_A, primary_factor_A,
        factor_to_emotions, threshold=0.9):

    # 動画ID → 行列インデックス
    vid_to_idx = {vid: i for i, vid in enumerate(sorted_ids)}

    results = []

    # 因子ごとに処理
    for fac, emos in factor_to_emotions.items():
        if len(emos) < 2:
            continue

        # emotion → その emotion に属する動画ID
        emo_to_vids = {}
        for vid in sorted_ids:
            emo = primary_emotion_A.get(vid)
            if emo in emos:
                emo_to_vids.setdefault(emo, []).append(vid)

        # emotion ペアをすべて調べる
        emo_list = list(emo_to_vids.keys())
        for i in range(len(emo_list)):
            for j in range(i + 1, len(emo_list)):
                emoA = emo_list[i]
                emoB = emo_list[j]

                vidsA = emo_to_vids[emoA]
                vidsB = emo_to_vids[emoB]

                if len(vidsA) == 0 or len(vidsB) == 0:
                    continue

                # A × B の距離を計算
                distances = []
                for va in vidsA:
                    for vb in vidsB:
                        ia = vid_to_idx[va]
                        ib = vid_to_idx[vb]
                        distances.append(D[ia, ib])

                mean_dist = np.mean(distances)

                if mean_dist >= threshold:
                    results.append((fac, emoA, emoB, mean_dist))

    return results

def RDM(model1, inputting1):
    BASE_1_path = f"./モデル別ファイル/論文用/05_FACTOR/05_FACTOR_{model1}_{inputting1}.csv"
    # 1. df_base_A を読み込む
    df_base_A = pd.read_csv(BASE_1_path, index_col=0)
    df_base_A = normalize_base_columns(df_base_A)

    # 2. 行：感情だけを抽出（最後の2行は primary_emotion / primary_factor）
    emotion_rows = df_base_A.index[:-2]

    # 3. 列：動画IDだけを抽出（int の列だけ）
    video_cols = [c for c in df_base_A.columns if isinstance(c, int)]
    video_cols = sorted(video_cols)

    # 4. 感情 × 動画ID の数値行列を作る
    df_numeric = df_base_A.loc[emotion_rows, video_cols]
    df_numeric = df_numeric.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 5. RDM を計算
    D = compute_rdm_cosine(df_numeric.T)   # ← 動画間の距離なので転置が重要

    # 6. 因子順に並べ替える
    primary_emotion_A = df_base_A.loc["primary_emotion"].to_dict()
    primary_factor_A = df_base_A.loc["primary_factor"].to_dict()

    # emotion → factor
    emotion_to_factor = {}

    for vid, emo in primary_emotion_A.items():
        fac = primary_factor_A.get(vid)
        if isinstance(fac, str):
            emotion_to_factor[emo] = fac   # "F14" のまま使う

    # factor → emotions
    factor_to_emotions = {}
    for emo, fac in emotion_to_factor.items():
        factor_to_emotions.setdefault(fac, []).append(emo)

    # 感情カテゴリの並び順（あなたの emotion_order を使う）
    emotion_order_index = {emo: i for i, emo in enumerate(emotion_order)}
    
    # --- カテゴリ相関行列 ---
    corr_matrix = df_numeric.T.corr()

    # --- 外部ペアリング ---
    pairings = {}

    for fac, emos in factor_to_emotions.items():
        for emo in emos:
            internal_corr = corr_matrix.loc[emo, emos].mean()

            if internal_corr < 0.9:
                external_corr = corr_matrix.loc[emo].drop(emos)
                best_partner = external_corr.idxmax()
                best_corr = external_corr.max()

                if best_corr >= 0.9:
                    pairings[emo] = best_partner


    factor_order = build_factor_order(factor_to_emotions, corr_matrix)

    factor_rank = {f: i for i, f in enumerate(factor_order)}

    factor_map = load_factor_map(
        f"./モデル別ファイル/論文用/06_factor_map/06_factor_map_{model1}_{inputting1}.csv")


    # --- 因子順ソートのためのキー ---
    def sort_key(vid):
        emo = primary_emotion_A[vid]
        fac = factor_map[emo]  # emotion → factor
        fac_base = fac.split("-")[0]  # F14-1 → F14

        fac_idx = factor_rank[fac_base]
        emo_idx = emotion_order_index.get(emo, 999)

        return (fac_idx, emo_idx, vid)

        # --- 並べ替え ---
    sorted_ids = sorted(video_cols, key=sort_key)

    # --- 因子境界線 ---
    boundaries = []
    prev_factor = primary_factor_A[sorted_ids[0]].split('-')[0]

    for i, vid in enumerate(sorted_ids):
        fac = primary_factor_A[vid].split('-')[0]
        if fac != prev_factor:
            boundaries.append(i)
            prev_factor = fac

    idx_sorted = [video_cols.index(vid) for vid in sorted_ids]
    D_sorted = D[np.ix_(idx_sorted, idx_sorted)]
    
    # --- 因子ごとの平均距離を計算 ---
    # --- emotion ペアの高距離を計算 ---
    high_pairs = find_high_distance_emotion_pairs(
        D_sorted, sorted_ids,
        primary_emotion_A, primary_factor_A,
        factor_to_emotions,
        threshold=0.9
    )

    # --- 結果を保存 ---
    embedding_dim = embedding_dims.get((model1, inputting1), None)

    """results_dim_and_pairs[(model1, inputting1)] = {
        "embedding_dim": embedding_dim,
        "high_pairs": len(high_pairs)
    }"""

    print(f"{model1} {inputting1}: dim={embedding_dim}, high_pairs={len(high_pairs)}")

    grouped = defaultdict(list)
    for fac, emoA, emoB, dist in high_pairs:
        grouped[fac].append((emoA, emoB, dist))

    # 出力
    for fac, pairs in grouped.items():
        print(f"因子: {fac}")
        emo_list = factor_to_emotions.get(fac, [])
        print(f"{fac}（含まれる感情: {emo_list}）")

        for emoA, emoB, dist in pairs:
            print(f"    {emoA} – {emoB}: {dist:.3f}")


    # --- プロット開始 ---
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(right=0.75)

    # --- カテゴリ境界 ---
    bounds = [0.0, 0.6, 0.7, 0.85, 1.0, D_sorted.max()]

    # --- 色（元の5色） ---
    colors = ["#050154", "#60dafc", "#299121", "#fd7f25", "#ffffff"]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    # --- RDM 本体 ---
    plt.imshow(D_sorted, cmap=cmap, norm=norm)

    # --- 因子境界線 ---
    for b in boundaries:
        plt.axhline(b - 0.5, color="black", linewidth=1.0)
        plt.axvline(b - 0.5, color="black", linewidth=1.0)

    # --- 凡例 ---
    legend_patches = [
        mpatches.Patch(color="#050154", label="50点の感情カテゴリが一致"),
        mpatches.Patch(color="#60dafc", label="上位2感情が一致"),
        mpatches.Patch(color="#299121", label="片方で強い感情が一致"),
        mpatches.Patch(color="#fd7f25", label="双方で弱い感情のみ一致"),
        mpatches.Patch(color="#ffffff",  label="一致なし"),
    ]

    plt.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=12
    )

    plt.title(f"RDM in {model1} {inputting1}")
    plt.grid(True, linestyle="--", alpha=0.3)

    try:
        plt.savefig(
            f'./モデル別ファイル/論文用/13_RDM/13_RDM_{model1}_{inputting1}.png',
            dpi=300
        )
    except MemoryError:
        print(
            f"[SKIP] MemoryError: RDM for {model1} {inputting1} skipped due to bad allocation.")
        plt.close()
        return
    except Exception as e:
        print(f"[SKIP] Unexpected error during RDM save: {e}")
        plt.close()
        return


def save_gwot_image(fig, save_path):
    """画像保存の共通関数"""
    try:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        if os.path.exists(save_path):
            print(f"[OK] Saved GWOT image: {save_path}")
        else:
            print(f"[ERROR] Save failed: {save_path}")
    except Exception as e:
        print(f"[ERROR] Exception during GWOT save: {e}")
        plt.close(fig)


def get_black_red_white():
    """黒→赤→白のカラーマップ"""
    return LinearSegmentedColormap.from_list(
        "black_red_white",
        ["black", "red", "white"]
    )


def plot_and_save_T_sorted(
    T_matrix, ids_A, ids_B,
    primary_emotion_A, primary_emotion_B,
    factor_map_A, factor_map_B,
    model1, model2, inputting1, inputting2, tag
):
    # --- 因子順に並べ替え ---
    T_sorted, boundaries_A, boundaries_B = compute_sorted_T(
        T_matrix, ids_A, primary_emotion_A, primary_emotion_B,
        factor_map_A, factor_map_B
    )

    # ★ プロットを大きく & 高解像度に
    fig = plt.figure(figsize=(12, 12), dpi=200)
    plt.subplots_adjust(right=0.82)

    # 白→赤→黒
    white_red_black = LinearSegmentedColormap.from_list(
        "white_red_black",
        ["white", "red", "black"]
    )

    vmax = T_matrix.max()

    im = plt.imshow(T_sorted, cmap=white_red_black, vmin=0, vmax=vmax)

    # --- 因子境界線（見やすい太さに調整） ---
    for b in boundaries_A:
        plt.axhline(b - 0.5, color="black", linewidth=1.5)
    for b in boundaries_B:
        plt.axvline(b - 0.5, color="black", linewidth=1.5)

    # --- カラーバー ---
    cbar = plt.colorbar(im)
    cbar.set_label("GWOT transport", fontsize=12)

    plt.title(
        f"GWOT {tag} (sorted): \n{model1} {inputting1} → {model2} {inputting2}")

    save_dir = "./モデル別ファイル/論文用/14_GWOT"
    os.makedirs(save_dir, exist_ok=True)

    filename = make_gwot_filename(model1, model2, inputting1, inputting2, tag)
    save_path = os.path.join(save_dir, filename)

    save_gwot_image(fig, save_path)

def compute_sorted_T(T, common_ids, primary_emotion_A, primary_emotion_B, factor_map_A, factor_map_B):
    """因子順に並べ替えた T と境界線を返す"""

    emotion_order_index = {emo: i for i, emo in enumerate(emotion_order)}

    def sort_key(vid, primary_emotion, factor_map):
        emo = primary_emotion[vid]
        fac = factor_map[emo].split("-")[0]
        fac_idx = int(fac[1:])
        emo_idx = emotion_order_index.get(emo, 999)
        return (fac_idx, emo_idx, vid)

    sorted_ids_A = sorted(common_ids, key=lambda v: sort_key(
        v, primary_emotion_A, factor_map_A))
    sorted_ids_B = sorted(common_ids, key=lambda v: sort_key(
        v, primary_emotion_B, factor_map_B))

    idxA_sorted = [common_ids.index(v) for v in sorted_ids_A]
    idxB_sorted = [common_ids.index(v) for v in sorted_ids_B]

    T_sorted = T[np.ix_(idxA_sorted, idxB_sorted)]

    def compute_boundaries(sorted_ids, primary_emotion, factor_map):
        boundaries = []
        prev = factor_map[primary_emotion[sorted_ids[0]]].split("-")[0]
        for i, vid in enumerate(sorted_ids):
            fac = factor_map[primary_emotion[vid]].split("-")[0]
            if fac != prev:
                boundaries.append(i)
                prev = fac
        return boundaries

    boundaries_A = compute_boundaries(
        sorted_ids_A, primary_emotion_A, factor_map_A)
    boundaries_B = compute_boundaries(
        sorted_ids_B, primary_emotion_B, factor_map_B)

    return T_sorted, boundaries_A, boundaries_B


def compute_gwot_id_match_rate(T, common_ids):
    """
    GWOT の輸送行列 T が、動画番号をどれだけ 1:1 で対応させているかを返す。
    """
    match_B = np.argmax(T, axis=1)  # A の各行が対応付けた B のインデックス

    matches = []
    for idxA, idxB in enumerate(match_B):
        vidA = common_ids[idxA]
        vidB = common_ids[idxB]

        matches.append(vidA == vidB)

    return sum(matches) / len(matches)

def GWOT(results, model1, model2, inputting1, inputting2):
    print(f"GWOT: {model1} ({inputting1}) vs {model2} ({inputting2})")

    # --- df_base を読み込む ---
    BASE_1_path = f"./モデル別ファイル/論文用/05_FACTOR/05_FACTOR_{model1}_{inputting1}.csv"
    BASE_2_path = f"./モデル別ファイル/論文用/05_FACTOR/05_FACTOR_{model2}_{inputting2}.csv"

    dfA = normalize_base_columns(pd.read_csv(BASE_1_path, index_col=0))
    dfB = normalize_base_columns(pd.read_csv(BASE_2_path, index_col=0))

    dfA.index = dfA.index.astype(str)
    dfB.index = dfB.index.astype(str)

    primary_emotion_A = dfA.loc["primary_emotion"].to_dict()
    primary_emotion_B = dfB.loc["primary_emotion"].to_dict()

    video_cols_A = [c for c in dfA.columns if isinstance(c, int)]
    video_cols_B = [c for c in dfB.columns if isinstance(c, int)]
    common_ids = sorted(set(video_cols_A) & set(video_cols_B))

    emotion_rows_A = [idx for idx in dfA.index if idx not in [
        "primary_emotion", "primary_factor"]]
    emotion_rows_B = [idx for idx in dfB.index if idx not in [
        "primary_emotion", "primary_factor"]]

    dfA = dfA.loc[emotion_rows_A, common_ids].apply(
        pd.to_numeric, errors="coerce").fillna(0)
    dfB = dfB.loc[emotion_rows_B, common_ids].apply(
        pd.to_numeric, errors="coerce").fillna(0)
    dfA = dfA.loc[emotion_rows_A, common_ids]
    dfB = dfB.loc[emotion_rows_B, common_ids]


    # --- 行と列を揃える ---
    all_emotions = sorted(set(dfA.index) | set(dfB.index))
    # A に足りない行を追加
    missing_A = [emo for emo in all_emotions if emo not in dfA.index]
    if missing_A:
        zeros = pd.DataFrame(
            0,
            index=missing_A,
            columns=dfA.columns
        )
        dfA = pd.concat([dfA, zeros], axis=0)


    # B に足りない行を追加
    missing_B = [emo for emo in all_emotions if emo not in dfB.index]
    if missing_B:
        zeros = pd.DataFrame(
            0,
            index=missing_B,
            columns=dfB.columns
        )
        dfB = pd.concat([dfB, zeros], axis=0)


    dfA = dfA.loc[all_emotions]
    dfB = dfB.loc[all_emotions]

    all_cols = sorted(set(dfA.columns) | set(dfB.columns))
    for df in [dfA, dfB]:
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            df[missing] = 0

    dfA2 = dfA[all_cols]
    dfB2 = dfB[all_cols]

    assert dfA2.shape == dfB2.shape

    # --- RDM ---
    D1 = compute_rdm_cosine(dfA.T)
    D2 = compute_rdm_cosine(dfB.T)
    rdm_corr_All = np.corrcoef(D1.flatten(), D2.flatten())[0, 1]

    # --- 因子マップ ---
    factor_map_A = load_factor_map(
        f"./モデル別ファイル/論文用/06_factor_map/06_factor_map_{model1}_{inputting1}.csv")
    factor_map_B = load_factor_map(
        f"./モデル別ファイル/論文用/06_factor_map/06_factor_map_{model2}_{inputting2}.csv")

    # --- GWOT ---
    p = q = np.ones(len(common_ids)) / len(common_ids)
    T = ot.gromov.gromov_wasserstein(D1, D2, p, q, 'square_loss')
    gwot_id_match_all = compute_gwot_id_match_rate(T, common_ids)

    # --- top250 / top750 ---
    match_B = np.argmax(T, axis=1)
    pairs = list(zip(common_ids, [common_ids[j] for j in match_B]))

    corrs = []
    for a, b in pairs:
        colA = dfA.loc[:, a]
        colB = dfB.loc[:, b]
        corr = 0 if colA.std() == 0 or colB.std(
        ) == 0 else np.corrcoef(colA, colB)[0, 1]
        corrs.append((a, b, corr))

    corrs_sorted = sorted(corrs, key=lambda x: x[2], reverse=True)
    top250_pairs = corrs_sorted[:250]
    top750_pairs = corrs_sorted[:750]

    ids_A_250 = [a for a, b, c in top250_pairs]
    ids_B_250 = [b for a, b, c in top250_pairs]
    ids_A_750 = [a for a, b, c in top750_pairs]
    ids_B_750 = [b for a, b, c in top750_pairs]

    T_250 = ot.gromov.gromov_wasserstein(
        compute_rdm_cosine(dfA.loc[:, ids_A_250].T),
        compute_rdm_cosine(dfB.loc[:, ids_B_250].T),
        np.ones(len(ids_A_250)) / len(ids_A_250),
        np.ones(len(ids_B_250)) / len(ids_B_250),
        'square_loss'
    )

    T_750 = ot.gromov.gromov_wasserstein(
        compute_rdm_cosine(dfA.loc[:, ids_A_750].T),
        compute_rdm_cosine(dfB.loc[:, ids_B_750].T),
        np.ones(len(ids_A_750)) / len(ids_A_750),
        np.ones(len(ids_B_750)) / len(ids_B_750),
        'square_loss'
    )

    # --- T_250 の描画（因子順 + 白背景 + 黒プロット） ---
    plot_and_save_T_sorted(
        T_250, ids_A_250, ids_B_250,
        primary_emotion_A, primary_emotion_B,
        factor_map_A, factor_map_B,
        model1, model2, inputting1, inputting2,
        "GW_250"
    )

    # --- T_750 の描画（因子順 + 白背景 + 黒プロット） ---
    plot_and_save_T_sorted(
        T_750, ids_A_750, ids_B_750,
        primary_emotion_A, primary_emotion_B,
        factor_map_A, factor_map_B,
        model1, model2, inputting1, inputting2,
        "GW_750"
    )
    gwot_id_match_250 = compute_gwot_id_match_rate(T_250, ids_A_250)
    gwot_id_match_750 = compute_gwot_id_match_rate(T_750, ids_A_750)


    """D1_250 = compute_rdm_cosine(dfA.loc[:, ids_A_250].T)
    D2_250 = compute_rdm_cosine(dfB.loc[:, ids_B_250].T)
    D1_750 = compute_rdm_cosine(dfA.loc[:, ids_A_750].T)
    D2_750 = compute_rdm_cosine(dfB.loc[:, ids_B_750].T)
    """
    
    dir_m_all = compute_direct_match_with_factor(common_ids, primary_emotion_A, primary_emotion_B, factor_map_A, factor_map_B)
    GWOT_M_all = compute_factor_match_stats(T, common_ids, common_ids, primary_emotion_A,
                               primary_emotion_B, factor_map_A, factor_map_B)
    """
    RDM_C_250 = np.corrcoef(
        D1_250.flatten(),
        D2_250.flatten()
    )[0, 1]
    GWOT_M_250 = compute_factor_match_stats(T_250, ids_A_250, ids_B_250, primary_emotion_A, primary_emotion_B, factor_map_A, factor_map_B)
    RDM_C_750 = np.corrcoef(D1_750.flatten(),
                            D2_750.flatten()
                            )[0, 1]
    GWOT_M_750 = compute_factor_match_stats(T_750, ids_A_750, ids_B_750, primary_emotion_A, primary_emotion_B, factor_map_A, factor_map_B)
    """
    # --- 結果を保存 ---
    results.append({
        "model1": model1,
        "model2": model2,
        "input1": inputting1,
        "input2": inputting2,
        "DIR_M_all": dir_m_all,
        "RDM_C_all": rdm_corr_All,
        "GWOT_M_all": GWOT_M_all,
        #"RDM_C_250": RDM_C_250,
        #"GWOT_M_250": GWOT_M_250,
        #"RDM_C_750": RDM_C_750,
        #"GWOT_M_750": GWOT_M_750,
        "GWOT_ID_all": gwot_id_match_all,
        "GWOT_ID_250": gwot_id_match_250,
        "GWOT_ID_750": gwot_id_match_750
    })


results = []

try:
    for model in MODELS_FRAMES + MODELS_OTHERS:
        RDM(model, "TEXT")
    for model in MODELS_FRAMES:
        RDM(model, "FRAMES")
    for i, model1 in enumerate(MODELS_FRAMES):
        GWOT(results, model1, model1, "FRAMES", "TEXT")
        for j, model2 in enumerate(MODELS_FRAMES):
            if j <= i:
                continue
            GWOT(results, model1, model2, "FRAMES", "FRAMES")
    for i, model1 in enumerate(MODELS_FRAMES + MODELS_OTHERS):
        for j, model2 in enumerate(MODELS_FRAMES + MODELS_OTHERS):
            if j <= i:
                continue
            GWOT(results, model1, model2, "TEXT", "TEXT")
    dfA = pd.DataFrame(results)
    dfA.to_csv("./モデル別ファイル/論文用/14_GWOT/14_GWOT_results.csv", index=False)

except Exception:
    exc_type, exc_value, exc_tb = sys.exc_info()
    tb_list = traceback.extract_tb(exc_tb)
    user_tb = None
    for tb in reversed(tb_list):
        if "site-packages" not in tb.filename:
            user_tb = tb
            break
    if user_tb:
        logging.error(f"{user_tb.lineno}行目:\n {exc_value}")
    else:
        tb = tb_list[-1]
        logging.error(f"{tb.lineno}行目:\n {exc_value}")
