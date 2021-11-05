import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error


class TimeSeriesSplitGenerator:
    def __init__(self, n_split=12, test_day_after="2014-09-01", slide=False):
        self.test_day_after = pd.to_datetime(test_day_after)
        self.n_split = n_split
        self.test_month_period = 12
        self.month = relativedelta(months=1)

        self.slide = slide

    def split(self, X):
        for m in range(self.test_month_period):
            test_month = self.test_day_after + relativedelta(months=m)
            test_index = (test_month <= X.date) & (X.date < test_month + self.month)
            valid_index = (test_month - self.month <= X.date) & (X.date < test_month)
            train_index = X.date < test_month - self.month
            if self.slide:
                train_index = train_index & (
                    test_month - self.month - relativedelta(months=12) <= X.date
                )
            yield train_index, valid_index, test_index


def lightgbm(X_train, Y_train, X_valid, Y_valid, X_test, seed):
    bst_params = {
        "boosting_type": "gbdt",
        "metric": "rmse",
        "objective": "regression",
        "n_jobs": -1,
        "seed": seed,
        "random_state": seed,
        "learning_rate": 0.01,
        "bagging_fraction": 0.75,
        "bagging_freq": 10,
        "colsample_bytree": 0.75,
        "num_boost_round": 10000,
        "early_stopping_rounds": 10,
        "verbose_eval": 1000,
    }

    lgb_train = lgb.Dataset(X_train, Y_train)
    lgb_valid = lgb.Dataset(X_valid, Y_valid)

    model = lgb.train(
        bst_params,
        lgb_train,
        valid_names=["train", "valid"],
        valid_sets=[lgb_train, lgb_valid],
        verbose_eval=1000,
    )

    # 検証データに対する予測値を求める
    va_pred = model.predict(X_valid, num_iteration=model.best_iteration)

    # テストデータに対する予測値を求める
    te_pred = np.array(model.predict(X_test, num_iteration=model.best_iteration))

    return va_pred, te_pred, model


def catboost(X_train, Y_train, X_valid, Y_valid, X_test, seed):
    # objectの列番号を取得
    categorical_features_indices = np.where(X_train.dtypes == np.object)[0]
    lgb_train = Pool(X_train, Y_train, cat_features=categorical_features_indices)
    lgb_valid = Pool(X_valid, Y_valid, cat_features=categorical_features_indices)
    model = CatBoostRegressor(
        eval_metric="RMSE",
        loss_function="RMSE",
        num_boost_round=10000,
        logging_level="Silent",
        random_seed=seed,
    )
    model.fit(
        lgb_train,
        eval_set=lgb_valid,
        early_stopping_rounds=10,
        verbose=True,
        use_best_model=True,
    )

    # 検証データに対する予測値を求める
    va_pred = model.predict(X_valid)

    mse = mean_squared_error(Y_valid, va_pred)
    rmse = np.sqrt(mse)  # RSME = √MSEの算出
    eval_metric = rmse

    print(f"eval's rmse: {eval_metric}")

    # テストデータに対する予測値を求める
    te_pred = np.array(model.predict(X_test))

    return va_pred, te_pred, model


def xgboost(X_train, Y_train, X_valid, Y_valid, X_test, seed):

    xgb_params = {
        "objective": "reg:linear",
        "eval_metric": "rmse",
        # "verbosity": 0,
        "seed": seed,
        "eta": 0.01,
        "num_boost_round": 10000,
        # "early_stopping_rounds": 10,
        # "verbose_eval": 100,
    }

    lgb_train = xgb.DMatrix(X_train, label=Y_train)
    lgb_valid = xgb.DMatrix(X_valid, label=Y_valid)
    lgb_test = xgb.DMatrix(X_test)
    evals = [(lgb_train, "train"), (lgb_valid, "eval")]
    evals_result = {}

    model = xgb.train(
        xgb_params,
        lgb_train,
        evals=evals,
        evals_result=evals_result,
        num_boost_round=10000,
        early_stopping_rounds=10,
        verbose_eval=1000,
    )

    # 検証データに対する予測値を求める
    va_pred = model.predict(lgb_valid)

    # テストデータに対する予測値を求める
    te_pred = list(model.predict(lgb_test))

    return va_pred, te_pred, model


def train(main_df, seed, output_dir):
    scores = []
    ids = []
    submission = []

    for i, data in enumerate(TimeSeriesSplitGenerator(slide=True).split(main_df)):
        train_index, valid_index, test_index = data

        print("-------------------------------------------")
        print(
            "train:", main_df[train_index].date.min(), main_df[train_index].date.max()
        )
        print(
            "valid:", main_df[valid_index].date.min(), main_df[valid_index].date.max()
        )
        print("test: ", main_df[test_index].date.min(), main_df[test_index].date.max())
        print(f"Fold : {i}")

        train = main_df[train_index].dropna(subset=["bikes_available"])
        valid = main_df[valid_index].dropna(subset=["bikes_available"])

        test = main_df[test_index]
        test0 = test[test["predict"] == 0]
        test1 = test[test["predict"] == 1]

        rmses = []
        # 予測対象日について、一日ずつモデルを作成していく
        for start_date in test1["date"].unique():
            train_from_test = test0[test0["date"] < start_date].dropna(
                subset=["bikes_available"]
            )
            train_add = pd.concat([train, train_from_test], axis=0)

            X_train, Y_train = (
                train_add.drop(columns=["id", "predict", "bikes_available", "date"]),
                train_add["bikes_available"],
            )
            X_valid, Y_valid = (
                valid.drop(columns=["id", "predict", "bikes_available", "date"]),
                valid["bikes_available"],
            )
            ids += list(test1[test1["date"] == start_date]["id"])
            X_test, _ = (
                test1[test1["date"] == start_date].drop(
                    columns=["id", "predict", "bikes_available", "date"]
                ),
                test["bikes_available"],
            )

            va_pred1, te_pred1, model = lightgbm(
                X_train, Y_train, X_valid, Y_valid, X_test, seed
            )
            pickle.dump(model, open(f"{output_dir}/lgbm/{start_date}.pkl", "wb"))
            va_pred2, te_pred2, model = catboost(
                X_train, Y_train, X_valid, Y_valid, X_test, seed
            )
            pickle.dump(model, open(f"{output_dir}/cat/{start_date}.pkl", "wb"))
            va_pred3, te_pred3, model = xgboost(
                X_train, Y_train, X_valid, Y_valid, X_test, seed
            )
            pickle.dump(model, open(f"{output_dir}/xgb/{start_date}.pkl", "wb"))

            va_pred = (va_pred1 + va_pred2 + va_pred3) / 3
            te_pred = (te_pred1 + te_pred2 + te_pred3) / 3

            # RSME = √MSEの算出
            mse = mean_squared_error(Y_valid, va_pred)
            rmse = np.sqrt(mse)

            rmses.append(rmse)

            # テストデータに対する予測値を求める
            submission += list(te_pred)
            print("")
            print(f"Fold: {i} {start_date} RMSE:{rmse}")
            print("")

        # フォールド毎の検証時のスコアを格納
        scores.append(np.mean(rmses))

        print("")
        print("################################")
        print(f"Fold: {i} RMSE:{np.mean(rmses)}")
        print("")

    print(f"CV: {np.mean(scores)}")

    return ids, submission

    # CV: 3.4555098
