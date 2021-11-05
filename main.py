"""main.py
以下を一貫して実行する
1.前処理:preprocess
2.訓練:trainと予測:predict

〇事前準備
・!pip3 install jpholiday
・!pip install catboost
・!pip install xgboost
"""


import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from preprocess import get_data
from train import train

warnings.simplefilter("ignore")


def seed_torch(seed=42):
    # python の組み込み関数の seed を固定
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # numpy の seed を固定
    np.random.seed(seed)
    # torch の seed を固定
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 決定論的アルゴリズムを使用する
    torch.backends.cudnn.deterministic = True


SEED = 42
OUTPUT = "/content/drive/MyDrive/SIGNATE/StudentCup2021autumn/20211106"


if __name__ in "__main__":
    seed_torch(SEED)

    # 前処理
    main_df = get_data(
        weather_path=Path("weather.csv"),
        station_path=Path("station.csv"),
        status_path=Path("status.csv"),
    )

    # 訓練と予測
    ids, submission = train(main_df, seed=SEED, output_dir=OUTPUT)

    # submissionファイル作成
    sub = pd.Series(submission)
    sub.index = ids
    sub.to_csv(f"{OUTPUT}/submmission_ensenble.csv", header=False)
