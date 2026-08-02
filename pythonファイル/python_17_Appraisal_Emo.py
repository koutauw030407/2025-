import os
import ot
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


MODELS_FRAMES = [
    "chatgpt-4o-latest",
    "gpt-5", "gpt-5.2-pro",
    "gpt-5.1-chat-latest", "gpt-5.2-chat-latest", "gpt-5-mini", "gpt-5-nano"]
MODELS_OTHERS = ["gpt-4o",  "gpt-4o-mini", "gpt-5.1", "gpt-5.2", "gpt-4.1", "gpt-4.1-mini",
                 "gpt-4.1-nano", "gpt-4-turbo", "gpt-4",  "gpt-3.5-turbo"]


def mantel_test(D1, D2, n_perm=1000, method="pearson", random_state=0):
    """
    D1, D2: (N, N) の距離行列
    戻り値: (r, p_value)
    """
    rng = np.random.RandomState(random_state)

    # 上三角成分をベクトル化
    idx = np.triu_indices_from(D1, k=1)
    v1 = D1[idx]
    v2 = D2[idx]

    if method == "pearson":
        r_obs, _ = pearsonr(v1, v2)
    else:
        raise NotImplementedError

    perm_rs = []
    for _ in range(n_perm):
        perm = rng.permutation(D1.shape[0])
        D2_perm = D2[perm][:, perm]
        v2_perm = D2_perm[idx]
        r_perm, _ = pearsonr(v1, v2_perm)
        perm_rs.append(r_perm)

    perm_rs = np.array(perm_rs)
    # 片側 or 両側は好みだが、ここでは両側
    p_value = (np.sum(np.abs(perm_rs) >= np.abs(r_obs)) + 1) / (n_perm + 1)

    return r_obs, p_value


def knn_preservation(X, Y, k=10):
    """
    X, Y: (N, d) の特徴行列
    k: 近傍数
    戻り値: preservation_rate (0〜1)
    """
    n = X.shape[0]

    nn_X = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(X)
    nn_Y = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(Y)

    neigh_X = nn_X.kneighbors(return_distance=False)
    neigh_Y = nn_Y.kneighbors(return_distance=False)

    # 各点について、自分自身(0番目)を除いた近傍集合
    neigh_X = neigh_X[:, 1:]
    neigh_Y = neigh_Y[:, 1:]

    overlaps = []
    for i in range(n):
        set_X = set(neigh_X[i])
        set_Y = set(neigh_Y[i])
        overlaps.append(len(set_X & set_Y) / k)

    return float(np.mean(overlaps))


def procrustes_error(X, Y, k=None):
    """
    X: (N, d1)
    Y: (N, d2)
    k: 共通次元（None の場合は min(d1, d2)）
    """
    from sklearn.decomposition import PCA
    from scipy.spatial import procrustes

    d1 = X.shape[1]
    d2 = Y.shape[1]

    if k is None:
        k = min(d1, d2)

    # PCA で次元を揃える
    Xk = PCA(n_components=k).fit_transform(X)
    Yk = PCA(n_components=k).fit_transform(Y)

    # Procrustes
    _, _, disparity = procrustes(Xk, Yk)
    return disparity



def save_results_to_csv(results, model, inputting):
    outdir = "./GWOT_results"
    os.makedirs(outdir, exist_ok=True)

    # 1. summary（数値だけ）
    summary = {
        "gw_distance2": results["gw_distance2"],
        "mantel_r": results["mantel_r"],
        "mantel_p": results["mantel_p"],
        "knn_preservation": results["knn_preservation"],
        "procrustes_disparity": results["procrustes_disparity"],
    }
    pd.DataFrame([summary]).to_csv(
        f"{outdir}/summary_{model}_{inputting}.csv",
        index=False
    )

    # 2. coupling_T
    pd.DataFrame(results["coupling_T"]).to_csv(
        f"{outdir}/coupling_T_{model}_{inputting}.csv"
    )

    # 3. D14
    pd.DataFrame(results["D14"]).to_csv(
        f"{outdir}/D14_{model}_{inputting}.csv"
    )

    # 4. Demo
    pd.DataFrame(results["Demo"]).to_csv(
        f"{outdir}/Demo_{model}_{inputting}.csv"
    )


def App_Emo(model, inputting):

    # 1. データ読み込み
    df14 = pd.read_csv(
        f"./モデル別ファイル/バックアップ/{model}/1227_{model}_{inputting}_3.csv"
    ).drop(columns=["Unnamed: 0"])

    dfemo = pd.read_csv(
        f"./モデル別ファイル/論文用/03_scores/03_scores_{model}_{inputting}.csv"
    ).drop(columns=["Unnamed: 0"])

    # 2. 形を揃える（動画 × 特徴）
    dfemo = dfemo.T

    df14.columns = df14.columns.astype(int)
    dfemo.columns = dfemo.columns.astype(int)

    common_frames = df14.columns.intersection(dfemo.columns)

    df14_common = df14[common_frames]      # (14, N)
    dfemo_common = dfemo[common_frames]    # (16, N)

    df14_fixed = df14_common.T      # (N, 14)
    dfemo_fixed = dfemo_common.T    # (N, 16)

    # 念のためフレーム順の整合性確認
    assert (df14_fixed.index == dfemo_fixed.index).all()

    # 3. 標準化（距離のスケールを揃える）
    scaler_14 = StandardScaler()
    X14 = scaler_14.fit_transform(df14_fixed.values)

    scaler_emo = StandardScaler()
    Xemo = scaler_emo.fit_transform(dfemo_fixed.values)

    # 4. 距離行列
    D14 = pairwise_distances(X14, metric="euclidean")
    Demo = pairwise_distances(Xemo, metric="euclidean")

    # 5. GWOT
    p = ot.unif(D14.shape[0])
    q = ot.unif(Demo.shape[0])

    gw_dist2 = ot.gromov.gromov_wasserstein2(
        D14, Demo, p, q, 'square_loss'
    )

    T = ot.gromov.gromov_wasserstein(
        D14, Demo, p, q, 'square_loss'
    )

    # 6. Mantel test（距離構造の相関）
    mantel_r, mantel_p = mantel_test(D14, Demo, n_perm=1000)

    # 7. kNN preservation（局所構造の保存度）
    knn_pres = knn_preservation(X14, Xemo, k=10)

    # 8. Procrustes（最適重ね合わせ後の誤差）
    proc_disp = procrustes_error(X14, Xemo)

    results = {
        "gw_distance2": gw_dist2,
        "mantel_r": mantel_r,
        "mantel_p": mantel_p,
        "knn_preservation": knn_pres,
        "procrustes_disparity": proc_disp,
        "coupling_T": T,
        "X14": X14,
        "Xemo": Xemo,
        "D14": D14,
        "Demo": Demo,
    }

    save_results_to_csv(results, model, inputting)
    print(f"Results saved for {model} - {inputting}")


for model in MODELS_FRAMES:
    App_Emo(model, "FRAMES")


for model in MODELS_FRAMES + MODELS_OTHERS:
    App_Emo(model, "TEXT")

