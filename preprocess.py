from pathlib import Path

import jpholiday
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def get_weekday_jp(dt):
    """曜日を追加するための関数を定義"""
    w_list = ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"]
    return w_list[dt.weekday()]


def get_data(
    weather_path: Path = Path("weather.csv"),
    station_path: Path = Path("station.csv"),
    status_path: Path = Path("status.csv"),
):

    weather_df = pd.read_csv(weather_path)
    weather_df["date"] = pd.to_datetime(weather_df["date"])

    station_df = pd.read_csv(station_path)

    status_df = pd.read_csv(status_path)

    # year, month, dayを結合してdatetime型に変換
    status_df["date"] = (
        status_df["year"].astype(str)
        + status_df["month"].astype(str).str.zfill(2)
        + status_df["day"].astype(str).str.zfill(2)
    )
    status_df["date"] = pd.to_datetime(status_df["date"])

    # dateから曜日情報を取得
    status_df["weekday"] = status_df["date"].apply(get_weekday_jp)

    main_df = status_df.merge(station_df, left_on="station_id", right_on="station_id")
    main_df = main_df.merge(weather_df, left_on="date", right_on="date")

    le = LabelEncoder()

    main_df["weekday"] = le.fit_transform(main_df["weekday"])
    main_df["city"] = le.fit_transform(main_df["city"])
    main_df["installation_date"] = le.fit_transform(main_df["installation_date"])
    main_df["events"] = le.fit_transform(main_df["events"].fillna("NaN"))

    holiday = []

    for i in tqdm(main_df["date"].dt.to_pydatetime()):
        holiday_name = jpholiday.is_holiday(i)
        if holiday_name:
            holiday.append(1)
        else:
            holiday.append(0)
    holiday = pd.DataFrame(holiday, columns=["holiday"])
    main_df = main_df.merge(holiday, left_index=True, right_index=True)

    return main_df
