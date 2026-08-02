from collections import Counter
import ast
import os
import re
import sys
import csv
import logging
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as mpatches
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
MODELS_OTHERS = ["gpt-4o",  "gpt-4o-mini", "gpt-5.1", "gpt-5.2", "gpt-4.1", "gpt-4.1-mini",
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


def split_emotion_groups_custom(factor_to_emotions, high_pairs):
    from collections import defaultdict, deque

    # fac → [(emoA, emoB, dist)]
    fac_edges = defaultdict(list)
    for fac, emoA, emoB, dist in high_pairs:
        fac_edges[fac].append((emoA, emoB, dist))

    fac_to_groups = {}

    for fac, emos in factor_to_emotions.items():
        emos = list(emos)

        # --- 2感情は必ず分割 ---
        if len(emos) == 2:
            # high_pairs が無いなら分割しない
            if fac not in fac_edges or len(fac_edges[fac]) == 0:
                fac_to_groups[fac] = [sorted(emos)]
            else:
                # high_pairs があるなら分割
                fac_to_groups[fac] = [[emos[0]], [emos[1]]]
            continue


        # --- 3感情の特別ルール ---
        if len(emos) == 3:
            e1, e2, e3 = emos
            edges = [(a, b) for a, b, d in fac_edges.get(fac, []) if d >= 0.9]

            if len(edges) == 0:
                fac_to_groups[fac] = [sorted(emos)]
                continue

            if len(edges) == 1:
                a, b = edges[0]
                c = [x for x in emos if x not in (a, b)][0]
                fac_to_groups[fac] = [sorted([c, a]), sorted([c, b])]
                continue

            if len(edges) >= 3:
                fac_to_groups[fac] = [[e1], [e2], [e3]]
                continue

        # --- 4感情以上 or その他 ---
        # high_pairs が無い因子は分割しない
        if fac not in fac_edges or len(fac_edges[fac]) == 0:
            fac_to_groups[fac] = [sorted(emos)]
            continue

        # --- high_pairs がある因子は「0.9以上だけ切断」して連結成分 ---
        cut_edges = set()
        for a, b, d in fac_edges[fac]:
            if d >= 0.9:
                cut_edges.add((a, b))
                cut_edges.add((b, a))

        # 0.9未満は全部つなぐ
        adj = defaultdict(list)
        for e1 in emos:
            for e2 in emos:
                if e1 != e2 and (e1, e2) not in cut_edges:
                    adj[e1].append(e2)

        visited = set()
        groups = []

        for e in emos:
            if e not in visited:
                q = deque([e])
                visited.add(e)
                comp = [e]
                while q:
                    x = q.popleft()
                    for y in adj[x]:
                        if y not in visited:
                            visited.add(y)
                            q.append(y)
                            comp.append(y)
                groups.append(sorted(comp))

        fac_to_groups[fac] = groups

    return fac_to_groups


def flatten_groups_in_factor_order(factor_to_emotions, groups):
    ordered = []
    for fac in sorted(factor_to_emotions.keys()):
        if fac in groups:
            ordered.extend(groups[fac])
        else:
            ordered.append(list(factor_to_emotions[fac]))
    return ordered


def save_wide_emotion_table(all_results, save_path):
    # モデル名をアルファベット順に
    models = sorted(set(m for m, inp in all_results.keys()))

    # 出力列の順序を作る
    columns = []
    for model in models:
        if (model, "FRAMES") in all_results:
            columns.append((model, "FRAMES"))
        if (model, "TEXT") in all_results:
            columns.append((model, "TEXT"))

    # 各モデルの emotion グループをアルファベット順に整形
    table = {}
    max_rows = 0

    for model, inp in columns:
        factor_to_emotions = all_results[(model, inp)]["factor_to_emotions"]
        groups_raw = all_results[(model, inp)]["groups"]

        # 因子順に展開（factor_map の全因子を必ず出す）
        groups = []
        for fac in sorted(factor_to_emotions.keys()):
            if fac in groups_raw:
                groups.extend(groups_raw[fac])
            else:
                groups.append(list(factor_to_emotions[fac]))

        # emotion グループをアルファベット順に並べる
        groups_sorted = sorted(groups, key=lambda g: str(g))

        # 感情数を先頭に追加
        count = len(groups_sorted)
        groups_sorted.insert(0, count)

        table[(model, inp)] = groups_sorted
        max_rows = max(max_rows, len(groups_sorted))

    # 行数を揃える
    for key in table:
        lst = table[key]
        if len(lst) < max_rows:
            lst.extend([""] * (max_rows - len(lst)))

    # CSV 書き込み
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 1行目：model
        writer.writerow(["model"] + [m for m, inp in columns])

        # 2行目：inputting
        writer.writerow(["inputting"] + [inp for m, inp in columns])

        # 3行目以降：emotion グループ
        for row_idx in range(max_rows):
            row = [""]
            for key in columns:
                row.append(str(table[key][row_idx]))
            writer.writerow(row)

        # --- 複数のグループに出現する感情を検出 ---
        emotion_to_group_count = defaultdict(int)

        for (model, inp), data in all_results.items():
            groups_raw = data["groups"]
            for fac, glist in groups_raw.items():
                for group in glist:
                    for emo in group:
                        emotion_to_group_count[emo] += 1

        duplicated_emotions = sorted(
            [emo for emo, cnt in emotion_to_group_count.items() if cnt >= 2]
        )

        # --- CSV の最後に書き込む ---
        writer.writerow([])
        writer.writerow(["emotions_appearing_in_multiple_groups"])
        for emo in duplicated_emotions:
            writer.writerow([emo])


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
    factor_to_emotions_full = {}
    for emo, fac in factor_map.items():
        factor_to_emotions_full.setdefault(fac, []).append(emo)

    high_pairs = find_high_distance_emotion_pairs(
        D_sorted, sorted_ids,
        primary_emotion_A, primary_factor_A,
        factor_to_emotions_full,  # ←ここを必ず full にする
        threshold=0.9
    )

    grouped = defaultdict(list)
    for fac, emoA, emoB, dist in high_pairs:
        grouped[fac].append((emoA, emoB, dist))
        """    # 出力
    print("\n", model1, inputting1)
    print(factor_to_emotions_full)
    for fac, pairs in grouped.items():
        emo_list = factor_to_emotions.get(fac, [])
        print(f"{fac}（含まれる感情: {emo_list}）")

        for emoA, emoB, dist in pairs:
            print(f"    {emoA} – {emoB}: {dist:.3f}")"""



    with open(f"./モデル別ファイル/論文用/13_RDM/emotion_groups.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "inputting", "factor", "group"])

        # --- high_pairs の分割結果 ---
        groups = split_emotion_groups_custom(factor_to_emotions_full, high_pairs)

        # --- 因子順に出力する（ここが重要） ---
        for fac in sorted(factor_to_emotions_full.keys()):
            if fac in groups:
                # 分割されたグループをそのまま出す
                for g in groups[fac]:
                    writer.writerow([model1, inputting1, fac, str(g)])
            else:
                # high_pairs がない因子は emotion 全体を1グループとして出す
                writer.writerow(
                    [model1, inputting1, fac, str(factor_to_emotions_full[fac])])

        # --- all_results に保存 ---
        all_results[(model1, inputting1)] = {
            "factor_to_emotions": factor_to_emotions_full,
            "groups": groups
        }



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


def count_groups(csv_path):
    counter = Counter()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            cell = row[0].strip()
            if not cell:
                continue

            # ★ リスト形式でない行はスキップ
            if not (cell.startswith("[") and cell.endswith("]")):
                # print("Skipping:", cell)  # 必要ならデバッグ
                continue

            try:
                group = ast.literal_eval(cell)
            except Exception as e:
                print("literal_eval error:", cell)
                continue

            key = tuple(group)
            counter[key] += 1

    return counter


all_results = {}

try:
    for model in MODELS_FRAMES + MODELS_OTHERS:
        RDM(model, "TEXT")
    for model in MODELS_FRAMES:
        RDM(model, "FRAMES")
    csv_path = "./モデル別ファイル/論文用/13_RDM/emotion_groups_all_models.csv"
    save_wide_emotion_table(
        all_results,
        csv_path
    )
    result = count_groups(csv_path)

    for group, count in result.items():
        print(group, ":", count)

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
