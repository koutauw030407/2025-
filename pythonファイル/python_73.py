import pandas as pd
import numpy as np
import os


MODELS_FRAMES = [
    "chatgpt-4o-latest",
    "gpt-5", "gpt-5.2-pro",
    "gpt-5.1-chat-latest", "gpt-5.2-chat-latest", "gpt-5-mini", "gpt-5-nano"]
MODELS_OTHERS = ["gpt-4o",  "gpt-4o-mini", "gpt-5.1", "gpt-5.2", "gpt-4.1", "gpt-4.1-mini",
                 "gpt-4.1-nano", "gpt-4-turbo", "gpt-4",  "gpt-3.5-turbo"]


feature_cols = [
    "approach", "arousal", "attention", "certainty", "commitment",
    "control", "dominance", "effort", "fairness", "identity",
    "obstruction", "safety", "upswing", "valence"
]


def sum_logistic_tables(models, inputting):

    summed_df = None

    for model in models:
        path = f"./モデル別ファイル/論文用/17_次元/{model}/{model}_{inputting}_ロジスティック回帰の係数.csv"

        if not os.path.exists(path):
            print("ファイルが見つかりません:", path)
            continue

        df = pd.read_csv(path)

        df = df.set_index("emotion")

        # 必要な列だけに絞る
        df = df[feature_cols]

        if summed_df is None:
            summed_df = df.copy()
        else:
            summed_df = summed_df.add(df, fill_value=0)

    # 保存
    out_path = f"./モデル別ファイル/論文用/17_次元/ロジスティック回帰の係数_和_{inputting}.csv"
    summed_df.to_csv(out_path)

    print(out_path, "に保存しました。")


# FRAMES
sum_logistic_tables(MODELS_FRAMES, "FRAMES")

# TEXT
sum_logistic_tables(MODELS_FRAMES + MODELS_OTHERS, "TEXT")
