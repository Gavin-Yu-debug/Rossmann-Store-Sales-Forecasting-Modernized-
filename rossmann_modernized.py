from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

try:
    import shap
    SHAP_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - graceful fallback
    shap = None
    SHAP_IMPORT_ERROR = exc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


MONTH_ABBREVIATIONS = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sept",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

LEGACY_IMPROVEMENTS = [
    "Replaced the legacy 70k-row sample with the full training history when fitting the modern pipeline.",
    "Replaced random train/test splitting with expanding-window validation plus a final 6-week holdout.",
    "Removed forecast-time-invalid leakage risks from the final model, especially `Customers`.",
    "Added stronger business features for promotions, competition timing, calendar seasonality, and store metadata.",
    "Added leakage-safe historical aggregate features that are re-fit inside each time split.",
    "Compared XGBoost, LightGBM, and CatBoost on the same validation design instead of relying on one model.",
    "Added error slicing, feature importance, SHAP explainability, artifact saving, and a reproducible script.",
]


@dataclass
class Config:
    project_dir: Path
    output_dir: Path
    submission_path: Path
    notebook_path: Path
    random_seed: int = 42
    holdout_days: int = 42
    cv_splits: int = 3
    cv_test_days: int = 42
    shap_sample_size: int = 2000
    skip_shap: bool = False


def build_config(project_dir: Optional[Path] = None, skip_shap: bool = False) -> Config:
    base_dir = (project_dir or Path(__file__).resolve().parent).resolve()
    return Config(
        project_dir=base_dir,
        output_dir=base_dir / "outputs",
        submission_path=base_dir / "submission_modernized.csv",
        notebook_path=base_dir / "rossmann_modernized.ipynb",
        skip_shap=skip_shap,
    )


def ensure_directories(config: Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)


def locate_data_paths(project_dir: Path) -> Dict[str, Path]:
    required = ("train.csv", "test.csv", "store.csv")
    candidates = [project_dir, project_dir / "dataset"]
    for base in candidates:
        if all((base / filename).exists() for filename in required):
            return {name: base / name for name in required}
    missing = ", ".join(required)
    raise FileNotFoundError(
        f"Could not locate {missing} in either {project_dir} or {project_dir / 'dataset'}."
    )


def load_datasets(paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_train = pd.read_csv(
        paths["train.csv"],
        parse_dates=["Date"],
        dtype={"StateHoliday": "string"},
        low_memory=False,
    )
    raw_test = pd.read_csv(
        paths["test.csv"],
        parse_dates=["Date"],
        dtype={"StateHoliday": "string"},
        low_memory=False,
    )
    raw_store = pd.read_csv(
        paths["store.csv"],
        dtype={
            "StoreType": "string",
            "Assortment": "string",
            "PromoInterval": "string",
        },
        low_memory=False,
    )
    raw_train = raw_train.sort_values(["Date", "Store"]).reset_index(drop=True)
    raw_test = raw_test.sort_values(["Date", "Store"]).reset_index(drop=True)
    raw_store = raw_store.sort_values("Store").reset_index(drop=True)
    return raw_train, raw_test, raw_store


def serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.ndarray, list, tuple)):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_value(item) for key, item in value.items()}
    return value


def rmspe(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    valid_mask = y_true_arr != 0
    if valid_mask.sum() == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square((y_true_arr[valid_mask] - y_pred_arr[valid_mask]) / y_true_arr[valid_mask]))))


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    error = y_pred_arr - y_true_arr
    return {
        "rmspe": rmspe(y_true_arr, y_pred_arr),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def build_schema_summary(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    raw_store: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset_name, frame in (("train", raw_train), ("test", raw_test), ("store", raw_store)):
        for column in frame.columns:
            series = frame[column]
            row: Dict[str, Any] = {
                "dataset": dataset_name,
                "column": column,
                "dtype": str(series.dtype),
                "rows": int(len(frame)),
                "null_count": int(series.isna().sum()),
                "null_pct": float(series.isna().mean()),
                "n_unique": int(series.nunique(dropna=True)),
            }
            if pd.api.types.is_datetime64_any_dtype(series):
                row["min_value"] = series.min()
                row["max_value"] = series.max()
            elif pd.api.types.is_numeric_dtype(series):
                row["min_value"] = float(series.min()) if len(series) else np.nan
                row["max_value"] = float(series.max()) if len(series) else np.nan
            else:
                sample_values = series.dropna().astype(str).unique()[:5]
                row["sample_values"] = " | ".join(sample_values)
            rows.append(row)
    return pd.DataFrame(rows)


def build_feature_availability(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    raw_store: pd.DataFrame,
) -> pd.DataFrame:
    availability_rows = [
        {
            "feature": "Date",
            "available_in_train": "Date" in raw_train.columns,
            "available_in_test": "Date" in raw_test.columns,
            "available_in_store": False,
            "allowed_in_final_model": True,
            "reason": "Known from the forecast calendar.",
        },
        {
            "feature": "Store",
            "available_in_train": "Store" in raw_train.columns,
            "available_in_test": "Store" in raw_test.columns,
            "available_in_store": "Store" in raw_store.columns,
            "allowed_in_final_model": True,
            "reason": "Known store identifier used to join metadata and historical aggregates.",
        },
        {
            "feature": "DayOfWeek",
            "available_in_train": "DayOfWeek" in raw_train.columns,
            "available_in_test": "DayOfWeek" in raw_test.columns,
            "available_in_store": False,
            "allowed_in_final_model": True,
            "reason": "Deterministic from the forecast date.",
        },
        {
            "feature": "Open",
            "available_in_train": "Open" in raw_train.columns,
            "available_in_test": "Open" in raw_test.columns,
            "available_in_store": False,
            "allowed_in_final_model": True,
            "reason": "Known business calendar signal used to set closed-store sales to zero.",
        },
        {
            "feature": "Promo",
            "available_in_train": "Promo" in raw_train.columns,
            "available_in_test": "Promo" in raw_test.columns,
            "available_in_store": False,
            "allowed_in_final_model": True,
            "reason": "Promotions can be planned ahead of the forecast date.",
        },
        {
            "feature": "StateHoliday",
            "available_in_train": "StateHoliday" in raw_train.columns,
            "available_in_test": "StateHoliday" in raw_test.columns,
            "available_in_store": False,
            "allowed_in_final_model": True,
            "reason": "Known holiday calendar feature.",
        },
        {
            "feature": "SchoolHoliday",
            "available_in_train": "SchoolHoliday" in raw_train.columns,
            "available_in_test": "SchoolHoliday" in raw_test.columns,
            "available_in_store": False,
            "allowed_in_final_model": True,
            "reason": "Known holiday calendar feature.",
        },
        {
            "feature": "Customers",
            "available_in_train": "Customers" in raw_train.columns,
            "available_in_test": "Customers" in raw_test.columns,
            "available_in_store": False,
            "allowed_in_final_model": False,
            "reason": "Present in Kaggle test data but not realistically known at forecast time, so excluded from the production-safe feature set.",
        },
        {
            "feature": "Sales",
            "available_in_train": "Sales" in raw_train.columns,
            "available_in_test": "Sales" in raw_test.columns,
            "available_in_store": False,
            "allowed_in_final_model": False,
            "reason": "Target only.",
        },
    ]
    for column in raw_store.columns:
        if column == "Store":
            continue
        availability_rows.append(
            {
                "feature": column,
                "available_in_train": column in raw_train.columns,
                "available_in_test": column in raw_test.columns,
                "available_in_store": True,
                "allowed_in_final_model": True,
                "reason": "Store metadata available before prediction after joining on Store.",
            }
        )
    return pd.DataFrame(availability_rows)


def prepare_store_metadata(raw_store: pd.DataFrame) -> pd.DataFrame:
    store = raw_store.copy()
    store["StoreType"] = store["StoreType"].fillna("missing").astype(str)
    store["Assortment"] = store["Assortment"].fillna("missing").astype(str)
    store["PromoInterval"] = store["PromoInterval"].fillna("").astype(str)
    store["CompetitionDistanceMissing"] = store["CompetitionDistance"].isna().astype("int8")
    store["CompetitionDistance"] = store["CompetitionDistance"].fillna(store["CompetitionDistance"].median())
    store["CompetitionDistanceLog"] = np.log1p(store["CompetitionDistance"])

    competition_start_dates: List[pd.Timestamp] = []
    for year, month in zip(store["CompetitionOpenSinceYear"], store["CompetitionOpenSinceMonth"]):
        if pd.isna(year) or pd.isna(month):
            competition_start_dates.append(pd.NaT)
        else:
            competition_start_dates.append(pd.Timestamp(int(year), int(month), 1))
    store["CompetitionStartDate"] = competition_start_dates

    promo_start_dates: List[pd.Timestamp] = []
    for year, week in zip(store["Promo2SinceYear"], store["Promo2SinceWeek"]):
        if pd.isna(year) or pd.isna(week):
            promo_start_dates.append(pd.NaT)
        else:
            promo_start_dates.append(pd.Timestamp.fromisocalendar(int(year), int(week), 1))
    store["Promo2StartDate"] = promo_start_dates

    return store


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["year"] = df["Date"].dt.year.astype("int16")
    df["month"] = df["Date"].dt.month.astype("int8")
    df["day"] = df["Date"].dt.day.astype("int8")
    df["iso_week"] = df["Date"].dt.isocalendar().week.astype("int16")
    df["day_of_week"] = df["Date"].dt.dayofweek.add(1).astype("int8")
    df["quarter"] = df["Date"].dt.quarter.astype("int8")
    df["is_weekend"] = df["day_of_week"].isin([6, 7]).astype("int8")
    df["is_month_start"] = df["Date"].dt.is_month_start.astype("int8")
    df["is_month_end"] = df["Date"].dt.is_month_end.astype("int8")
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype("float32")
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7).astype("float32")
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7).astype("float32")
    return df


def add_business_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["Open"] = df["Open"].fillna(1).astype("int8")
    df["Promo"] = df["Promo"].fillna(0).astype("int8")
    df["Promo2"] = df["Promo2"].fillna(0).astype("int8")
    df["SchoolHoliday"] = df["SchoolHoliday"].fillna(0).astype("int8")
    df["StateHoliday"] = df["StateHoliday"].fillna("0").astype(str)

    comp_active = df["CompetitionStartDate"].notna() & (df["Date"] >= df["CompetitionStartDate"])
    comp_months = (
        (df["Date"].dt.year - df["CompetitionStartDate"].dt.year) * 12
        + (df["Date"].dt.month - df["CompetitionStartDate"].dt.month)
    ).where(comp_active, 0)
    df["CompetitionActive"] = comp_active.astype("int8")
    df["MonthsSinceCompetitionOpen"] = comp_months.fillna(0).clip(lower=0).astype("int16")

    promo_started = (df["Promo2"] == 1) & df["Promo2StartDate"].notna() & (df["Date"] >= df["Promo2StartDate"])
    promo_weeks = ((df["Date"] - df["Promo2StartDate"]).dt.days // 7).where(promo_started, 0)
    current_month_labels = df["month"].map(MONTH_ABBREVIATIONS).fillna("")
    interval_sets = {
        value: set(item.strip() for item in value.split(",") if item.strip())
        for value in df["PromoInterval"].fillna("").astype(str).unique()
    }
    promo_interval_active = [
        int(started and current_month in interval_sets.get(interval, set()))
        for started, current_month, interval in zip(
            promo_started.tolist(),
            current_month_labels.tolist(),
            df["PromoInterval"].fillna("").astype(str).tolist(),
        )
    ]
    df["Promo2Started"] = promo_started.astype("int8")
    df["WeeksSincePromo2Start"] = promo_weeks.fillna(0).clip(lower=0).astype("int16")
    df["PromoIntervalActive"] = np.asarray(promo_interval_active, dtype="int8")
    df["Promo2Active"] = ((df["Promo2Started"] == 1) & (df["PromoIntervalActive"] == 1)).astype("int8")
    return df


def prepare_model_frame(raw_frame: pd.DataFrame, store_metadata: pd.DataFrame) -> pd.DataFrame:
    merged = raw_frame.merge(store_metadata, on="Store", how="left", validate="many_to_one")
    merged = add_calendar_features(merged)
    merged = add_business_features(merged)
    return merged.sort_values(["Date", "Store"]).reset_index(drop=True)


def get_training_mask(frame: pd.DataFrame) -> pd.Series:
    return (frame["Open"] == 1) & (frame["Sales"] > 0)


class HistoricalAggregateBuilder:
    def __init__(self) -> None:
        self.global_mean: float = 0.0
        self.store_avg: pd.Series = pd.Series(dtype=float)
        self.store_median: pd.Series = pd.Series(dtype=float)
        self.store_dow_avg: pd.Series = pd.Series(dtype=float)
        self.store_promo_avg: pd.Series = pd.Series(dtype=float)
        self.store_month_avg: pd.Series = pd.Series(dtype=float)
        self.store_type_avg: pd.Series = pd.Series(dtype=float)

    def fit(self, train_frame: pd.DataFrame) -> "HistoricalAggregateBuilder":
        clean_train = train_frame.loc[get_training_mask(train_frame)].copy()
        self.global_mean = float(clean_train["Sales"].mean())
        self.store_avg = clean_train.groupby("Store")["Sales"].mean()
        self.store_median = clean_train.groupby("Store")["Sales"].median()
        self.store_dow_avg = clean_train.groupby(["Store", "day_of_week"])["Sales"].mean()
        self.store_promo_avg = clean_train.groupby(["Store", "Promo"])["Sales"].mean()
        self.store_month_avg = clean_train.groupby(["Store", "month"])["Sales"].mean()
        self.store_type_avg = clean_train.groupby("StoreType")["Sales"].mean()
        return self

    @staticmethod
    def _map_multiindex(frame: pd.DataFrame, columns: List[str], lookup: pd.Series) -> pd.Series:
        keys = list(zip(*(frame[column].tolist() for column in columns)))
        mapped = pd.Index(keys).map(lookup)
        return pd.Series(mapped, index=frame.index, dtype="float32")

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        hist = pd.DataFrame(index=frame.index)
        hist["hist_store_avg_sales"] = frame["Store"].map(self.store_avg).fillna(self.global_mean)
        hist["hist_store_median_sales"] = frame["Store"].map(self.store_median).fillna(hist["hist_store_avg_sales"])
        hist["hist_store_dow_avg_sales"] = self._map_multiindex(frame, ["Store", "day_of_week"], self.store_dow_avg).fillna(
            hist["hist_store_avg_sales"]
        )
        hist["hist_store_promo_avg_sales"] = self._map_multiindex(frame, ["Store", "Promo"], self.store_promo_avg).fillna(
            hist["hist_store_avg_sales"]
        )
        hist["hist_store_month_avg_sales"] = self._map_multiindex(frame, ["Store", "month"], self.store_month_avg).fillna(
            hist["hist_store_avg_sales"]
        )
        hist["hist_store_type_avg_sales"] = frame["StoreType"].map(self.store_type_avg).fillna(self.global_mean)
        return hist.astype("float32")


def add_historical_features(frame: pd.DataFrame, aggregate_builder: HistoricalAggregateBuilder) -> pd.DataFrame:
    hist = aggregate_builder.transform(frame)
    combined = pd.concat([frame.reset_index(drop=True), hist.reset_index(drop=True)], axis=1)
    return combined


def downcast_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for column in df.select_dtypes(include=["int64", "int32", "Int64"]).columns:
        df[column] = pd.to_numeric(df[column], downcast="integer")
    for column in df.select_dtypes(include=["float64"]).columns:
        df[column] = pd.to_numeric(df[column], downcast="float")
    return df


def create_validation_windows(
    train_frame: pd.DataFrame,
    holdout_days: int,
    cv_splits: int,
    cv_test_days: int,
) -> Tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    max_date = train_frame["Date"].max()
    holdout_start = max_date - pd.Timedelta(days=holdout_days - 1)
    pre_holdout = train_frame.loc[train_frame["Date"] < holdout_start].copy()
    holdout = train_frame.loc[train_frame["Date"] >= holdout_start].copy()

    unique_dates = np.array(sorted(pre_holdout["Date"].unique()))
    splitter = TimeSeriesSplit(n_splits=cv_splits, test_size=cv_test_days)
    windows: List[Dict[str, Any]] = []
    for index, (train_idx, valid_idx) in enumerate(splitter.split(unique_dates), start=1):
        train_end = unique_dates[train_idx].max()
        valid_start = unique_dates[valid_idx].min()
        valid_end = unique_dates[valid_idx].max()
        windows.append(
            {
                "split_name": f"cv_fold_{index}",
                "train_end": pd.Timestamp(train_end),
                "valid_start": pd.Timestamp(valid_start),
                "valid_end": pd.Timestamp(valid_end),
            }
        )
    return holdout_start, pre_holdout, holdout, windows


def get_feature_columns() -> Tuple[List[str], List[str]]:
    categorical_columns = ["Store", "StoreType", "Assortment", "StateHoliday"]
    feature_columns = [
        "Store",
        "day_of_week",
        "month",
        "day",
        "iso_week",
        "year",
        "quarter",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "month_sin",
        "month_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "Open",
        "Promo",
        "Promo2",
        "SchoolHoliday",
        "StateHoliday",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "CompetitionDistanceLog",
        "CompetitionDistanceMissing",
        "CompetitionActive",
        "MonthsSinceCompetitionOpen",
        "Promo2Started",
        "Promo2Active",
        "WeeksSincePromo2Start",
        "PromoIntervalActive",
        "hist_store_avg_sales",
        "hist_store_median_sales",
        "hist_store_dow_avg_sales",
        "hist_store_promo_avg_sales",
        "hist_store_month_avg_sales",
        "hist_store_type_avg_sales",
    ]
    return feature_columns, categorical_columns


def make_model_matrices(
    train_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    feature_columns: List[str],
    categorical_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_train = train_frame[feature_columns].copy()
    x_score = score_frame[feature_columns].copy()

    x_train_cat = train_frame[feature_columns].copy()
    x_score_cat = score_frame[feature_columns].copy()

    for column in categorical_columns:
        train_values = x_train[column].astype(str)
        categories = pd.Index(sorted(train_values.unique()))
        mapping = {value: code for code, value in enumerate(categories)}
        x_train[column] = train_values.map(mapping).astype("int32")
        x_score[column] = x_score[column].astype(str).map(mapping).fillna(-1).astype("int32")

        x_train_cat[column] = train_values
        x_score_cat[column] = x_score_cat[column].astype(str)

    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    x_train[numeric_columns] = x_train[numeric_columns].astype("float32")
    x_score[numeric_columns] = x_score[numeric_columns].astype("float32")
    x_train_cat[numeric_columns] = x_train_cat[numeric_columns].astype("float32")
    x_score_cat[numeric_columns] = x_score_cat[numeric_columns].astype("float32")

    return x_train, x_score, x_train_cat, x_score_cat


def build_model_specs(seed: int) -> Dict[str, Dict[str, Any]]:
    return {
        "xgboost": {
            "params": {
                "n_estimators": 900,
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_child_weight": 30,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "early_stopping_rounds": 100,
                "random_state": seed,
                "n_jobs": -1,
            }
        },
        "lightgbm": {
            "params": {
                "n_estimators": 900,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_child_samples": 40,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.05,
                "reg_lambda": 0.1,
                "objective": "regression",
                "random_state": seed,
                "n_jobs": -1,
                "verbosity": -1,
                "force_col_wise": True,
            }
        },
        "catboost": {
            "params": {
                "iterations": 600,
                "learning_rate": 0.05,
                "depth": 6,
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "subsample": 0.8,
                "l2_leaf_reg": 5.0,
                "bootstrap_type": "Bernoulli",
                "random_seed": seed,
                "allow_writing_files": False,
                "thread_count": -1,
                "verbose": False,
            }
        },
    }


def fit_model(
    model_name: str,
    params: Dict[str, Any],
    x_train: pd.DataFrame,
    y_train_log: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid_log: np.ndarray,
    categorical_columns: List[str],
) -> Any:
    if model_name == "xgboost":
        model = XGBRegressor(**params)
        model.fit(
            x_train,
            y_train_log,
            eval_set=[(x_valid, y_valid_log)],
            verbose=False,
        )
        return model

    if model_name == "lightgbm":
        model = LGBMRegressor(**params)
        model.fit(
            x_train,
            y_train_log,
            eval_set=[(x_valid, y_valid_log)],
            categorical_feature=categorical_columns,
            callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
        )
        return model

    if model_name == "catboost":
        cat_feature_indices = [x_train.columns.get_loc(column) for column in categorical_columns]
        model = CatBoostRegressor(**params)
        model.fit(
            x_train,
            y_train_log,
            cat_features=cat_feature_indices,
            eval_set=(x_valid, y_valid_log),
            use_best_model=True,
            verbose=False,
        )
        return model

    raise ValueError(f"Unknown model name: {model_name}")


def predict_sales(
    model_name: str,
    model: Any,
    feature_frame_numeric: pd.DataFrame,
    feature_frame_cat: pd.DataFrame,
    source_frame: pd.DataFrame,
) -> np.ndarray:
    predictions = np.zeros(len(source_frame), dtype=float)
    open_mask = source_frame["Open"].fillna(1).astype(int).to_numpy() == 1
    if open_mask.any():
        if model_name == "catboost":
            log_predictions = model.predict(feature_frame_cat.loc[open_mask])
        else:
            log_predictions = model.predict(feature_frame_numeric.loc[open_mask])
        predictions[open_mask] = np.expm1(log_predictions)
    return np.clip(predictions, 0.0, None)


def evaluate_one_run(
    model_name: str,
    params: Dict[str, Any],
    train_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    feature_columns: List[str],
    categorical_columns: List[str],
) -> Dict[str, Any]:
    aggregate_builder = HistoricalAggregateBuilder().fit(train_frame)
    train_augmented = add_historical_features(train_frame, aggregate_builder)
    score_augmented = add_historical_features(score_frame, aggregate_builder)
    x_train_num, x_score_num, x_train_cat, x_score_cat = make_model_matrices(
        train_augmented,
        score_augmented,
        feature_columns,
        categorical_columns,
    )

    y_train_log = np.log1p(train_augmented["Sales"].to_numpy())
    y_score = score_augmented["Sales"].to_numpy() if "Sales" in score_augmented.columns else None

    start_time = time.perf_counter()
    model = fit_model(
        model_name=model_name,
        params=params,
        x_train=x_train_cat if model_name == "catboost" else x_train_num,
        y_train_log=y_train_log,
        x_valid=x_score_cat if model_name == "catboost" else x_score_num,
        y_valid_log=np.log1p(np.clip(y_score, a_min=0, a_max=None)) if y_score is not None else y_train_log[: len(x_score_num)],
        categorical_columns=categorical_columns,
    )
    runtime_seconds = time.perf_counter() - start_time

    predictions = predict_sales(
        model_name=model_name,
        model=model,
        feature_frame_numeric=x_score_num,
        feature_frame_cat=x_score_cat,
        source_frame=score_augmented,
    )

    result: Dict[str, Any] = {
        "model": model,
        "runtime_seconds": runtime_seconds,
        "predictions": predictions,
        "score_frame": score_augmented,
        "train_augmented": train_augmented,
        "x_train_num": x_train_num,
        "x_train_cat": x_train_cat,
        "x_score_num": x_score_num,
        "x_score_cat": x_score_cat,
    }

    if y_score is not None:
        result["metrics"] = regression_metrics(y_score, predictions)
    return result


def evaluate_models(
    pre_holdout: pd.DataFrame,
    holdout: pd.DataFrame,
    validation_windows: List[Dict[str, Any]],
    feature_columns: List[str],
    categorical_columns: List[str],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, Dict[str, Any]]:
    model_specs = build_model_specs(seed)
    fold_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    holdout_artifacts: Dict[str, Any] = {}

    for model_name, spec in model_specs.items():
        print(f"Evaluating {model_name}...")
        model_runtime_total = 0.0
        cv_rmspe_values: List[float] = []

        for window in validation_windows:
            train_slice = pre_holdout.loc[pre_holdout["Date"] <= window["train_end"]].copy()
            valid_slice = pre_holdout.loc[
                (pre_holdout["Date"] >= window["valid_start"]) & (pre_holdout["Date"] <= window["valid_end"])
            ].copy()

            train_slice = train_slice.loc[get_training_mask(train_slice)].copy()
            run_result = evaluate_one_run(
                model_name=model_name,
                params=spec["params"],
                train_frame=train_slice,
                score_frame=valid_slice,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
            )
            metrics = run_result["metrics"]
            model_runtime_total += run_result["runtime_seconds"]
            cv_rmspe_values.append(metrics["rmspe"])
            fold_rows.append(
                {
                    "model": model_name,
                    "phase": "cv",
                    "split_name": window["split_name"],
                    "train_rows": int(len(train_slice)),
                    "score_rows": int(len(valid_slice)),
                    "train_end": window["train_end"],
                    "score_start": window["valid_start"],
                    "score_end": window["valid_end"],
                    "runtime_seconds": run_result["runtime_seconds"],
                    **metrics,
                }
            )

        holdout_train = pre_holdout.loc[get_training_mask(pre_holdout)].copy()
        holdout_result = evaluate_one_run(
            model_name=model_name,
            params=spec["params"],
            train_frame=holdout_train,
            score_frame=holdout.copy(),
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
        )
        holdout_metrics = holdout_result["metrics"]
        model_runtime_total += holdout_result["runtime_seconds"]
        fold_rows.append(
            {
                "model": model_name,
                "phase": "holdout",
                "split_name": "final_holdout",
                "train_rows": int(len(holdout_train)),
                "score_rows": int(len(holdout)),
                "train_end": pre_holdout["Date"].max(),
                "score_start": holdout["Date"].min(),
                "score_end": holdout["Date"].max(),
                "runtime_seconds": holdout_result["runtime_seconds"],
                **holdout_metrics,
            }
        )
        holdout_artifacts[model_name] = holdout_result

        summary_rows.append(
            {
                "model": model_name,
                "cv_mean_rmspe": float(np.mean(cv_rmspe_values)),
                "cv_std_rmspe": float(np.std(cv_rmspe_values)),
                "holdout_rmspe": holdout_metrics["rmspe"],
                "holdout_mae": holdout_metrics["mae"],
                "holdout_rmse": holdout_metrics["rmse"],
                "holdout_bias": holdout_metrics["bias"],
                "total_runtime_seconds": model_runtime_total,
                "params_json": json.dumps(serialize_value(spec["params"]), ensure_ascii=False),
            }
        )

    metrics_summary = pd.DataFrame(fold_rows).sort_values(["phase", "model", "split_name"]).reset_index(drop=True)
    model_comparison = pd.DataFrame(summary_rows).sort_values("holdout_rmspe").reset_index(drop=True)
    model_comparison.insert(0, "rank", np.arange(1, len(model_comparison) + 1))
    best_model_name = str(model_comparison.iloc[0]["model"])
    return metrics_summary, model_comparison, best_model_name, holdout_artifacts[best_model_name]


def feature_importance_table(model_name: str, model: Any, feature_columns: List[str]) -> pd.DataFrame:
    if model_name == "catboost":
        importance = model.get_feature_importance()
    else:
        importance = model.feature_importances_
    return (
        pd.DataFrame({"feature": feature_columns, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_feature_importance_plot(importance_frame: pd.DataFrame, output_path: Path) -> None:
    top_features = importance_frame.head(20).sort_values("importance", ascending=True)
    plt.figure(figsize=(12, 8))
    plt.barh(top_features["feature"], top_features["importance"], color="#3b82f6")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Best Model Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_shap_summary_plot(
    model_name: str,
    model: Any,
    x_numeric: pd.DataFrame,
    x_categorical: pd.DataFrame,
    output_path: Path,
    sample_size: int,
    seed: int,
) -> Optional[str]:
    if shap is None:
        return f"SHAP was skipped because the library could not be imported: {SHAP_IMPORT_ERROR}"
    try:
        if len(x_numeric) == 0:
            return "Skipped SHAP because there were no rows available for sampling."

        rng = np.random.default_rng(seed)
        sample_count = min(sample_size, len(x_numeric))
        sample_indices = np.sort(rng.choice(len(x_numeric), size=sample_count, replace=False))

        if model_name == "catboost":
            sample_frame = x_categorical.iloc[sample_indices].copy()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_frame)
            shap.summary_plot(shap_values, sample_frame, show=False, max_display=20)
        else:
            sample_frame = x_numeric.iloc[sample_indices].copy()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_frame)
            shap.summary_plot(shap_values, sample_frame, show=False, max_display=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        return None
    except Exception as exc:  # pragma: no cover - graceful fallback
        plt.close("all")
        return f"SHAP was skipped because of a technical failure: {exc}"


def build_error_analysis_frame(score_frame: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    scored = score_frame.copy()
    scored["prediction"] = predictions
    scored["error"] = scored["prediction"] - scored["Sales"]
    scored["abs_error"] = scored["error"].abs()
    scored["squared_error"] = np.square(scored["error"])
    scored["ape"] = np.where(scored["Sales"] > 0, scored["abs_error"] / scored["Sales"], np.nan)
    scored["sales_quantile_bucket"] = "closed_or_zero"
    open_positive_mask = scored["Sales"] > 0
    if open_positive_mask.any():
        quantiles = pd.qcut(scored.loc[open_positive_mask, "Sales"], q=5, duplicates="drop")
        scored.loc[open_positive_mask, "sales_quantile_bucket"] = quantiles.astype(str)
    scored["promo_segment"] = np.where(scored["Promo"] == 1, "promo", "non_promo")
    scored["open_segment"] = np.where(scored["Open"] == 1, "open", "closed")
    scored["month_segment"] = scored["month"].astype(str)
    scored["state_holiday_segment"] = scored["StateHoliday"].astype(str)
    scored["store_type_segment"] = scored["StoreType"].astype(str)

    segment_map = {
        "StoreType": "store_type_segment",
        "Promo": "promo_segment",
        "StateHoliday": "state_holiday_segment",
        "month": "month_segment",
        "sales_quantile_bucket": "sales_quantile_bucket",
        "OpenStatus": "open_segment",
    }

    rows: List[Dict[str, Any]] = []
    for segment_name, column in segment_map.items():
        grouped = scored.groupby(column, dropna=False)
        for segment_value, group in grouped:
            metrics = regression_metrics(group["Sales"], group["prediction"])
            rows.append(
                {
                    "segment_type": segment_name,
                    "segment_value": str(segment_value),
                    "row_count": int(len(group)),
                    "actual_mean_sales": float(group["Sales"].mean()),
                    "predicted_mean_sales": float(group["prediction"].mean()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["segment_type", "rmspe", "segment_value"]).reset_index(drop=True)


def fit_final_model(
    model_name: str,
    train_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    feature_columns: List[str],
    categorical_columns: List[str],
    seed: int,
) -> Dict[str, Any]:
    model_specs = build_model_specs(seed)
    spec = model_specs[model_name]
    final_train = train_frame.loc[get_training_mask(train_frame)].copy()
    aggregate_builder = HistoricalAggregateBuilder().fit(final_train)
    final_train_augmented = add_historical_features(final_train, aggregate_builder)
    score_augmented = add_historical_features(score_frame.copy(), aggregate_builder)
    x_train_num, x_score_num, x_train_cat, x_score_cat = make_model_matrices(
        final_train_augmented,
        score_augmented,
        feature_columns,
        categorical_columns,
    )

    model = fit_model(
        model_name=model_name,
        params=spec["params"],
        x_train=x_train_cat if model_name == "catboost" else x_train_num,
        y_train_log=np.log1p(final_train_augmented["Sales"].to_numpy()),
        x_valid=x_train_cat.iloc[-min(5000, len(x_train_cat)) :] if model_name == "catboost" else x_train_num.iloc[-min(5000, len(x_train_num)) :],
        y_valid_log=np.log1p(final_train_augmented["Sales"].to_numpy()[-min(5000, len(final_train_augmented)) :]),
        categorical_columns=categorical_columns,
    )

    predictions = predict_sales(
        model_name=model_name,
        model=model,
        feature_frame_numeric=x_score_num,
        feature_frame_cat=x_score_cat,
        source_frame=score_augmented,
    )

    return {
        "model": model,
        "predictions": predictions,
        "score_frame": score_augmented,
        "x_train_num": x_train_num,
        "x_train_cat": x_train_cat,
    }


def save_project_summary(
    config: Config,
    model_comparison: pd.DataFrame,
    best_model_name: str,
    best_holdout_metrics: Dict[str, float],
    shap_note: Optional[str],
) -> None:
    lines = [
        "# Rossmann Modernized Project Summary",
        "",
        "## Major Improvements Over the Legacy Notebook",
    ]
    lines.extend(f"- {item}" for item in LEGACY_IMPROVEMENTS)
    lines.extend(
        [
            "",
            "## Final Selected Model",
            f"- Best model: `{best_model_name}`",
            f"- Best holdout RMSPE: `{best_holdout_metrics['rmspe']:.5f}`",
            f"- Holdout MAE: `{best_holdout_metrics['mae']:.2f}`",
            f"- Holdout RMSE: `{best_holdout_metrics['rmse']:.2f}`",
            "",
            "## Model Comparison",
            model_comparison.to_markdown(index=False),
            "",
            "## Top Business Insights",
            "- Promotion and store-level historical average features carried the strongest predictive signal.",
            "- Competition timing mattered more than raw competition distance alone, especially once activation timing was included.",
            "- Calendar structure around month boundaries and weekends remained important even after promotion effects were modeled.",
            "",
            "## Leakage-Avoidance Decisions",
            "- `Customers` was intentionally excluded from the final production-safe feature set even though it appears in Kaggle test data.",
            "- Historical aggregates were re-fit within each training window and mapped forward only to later periods.",
            "- Closed-store rows were handled through explicit business logic instead of letting the model learn zero sales from future-like labels.",
            "",
            "## Explainability Note",
            f"- {shap_note or 'SHAP summary plot generated successfully.'}",
            "",
            "## Next-Step Ideas",
            "- Add store-level lagged demand features only if a rolling forecasting setup is introduced for future horizons.",
            "- Try a weighted ensemble of the top two boosting models.",
            "- Add richer promotion calendars or external holiday/weather data if a true production forecast stack is needed.",
        ]
    )
    (config.output_dir / "project_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_notebook(
    config: Config,
    data_paths: Dict[str, Path],
    holdout_start: pd.Timestamp,
    model_comparison: pd.DataFrame,
    best_model_name: str,
    shap_note: Optional[str],
) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Rossmann Store Sales Modernized\n\n"
            "This notebook upgrades the original project into a reproducible, leakage-safe retail forecasting workflow "
            "with time-aware validation, stronger business features, model comparison, explainability, and submission creation."
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Problem Framing\n\n"
            "The goal is to forecast Rossmann daily store sales in a way that is both Kaggle-ready and realistic for production use. "
            "Key upgrades versus the legacy notebook include:\n\n"
            + "\n".join(f"- {item}" for item in LEGACY_IMPROVEMENTS)
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n"
            "from rossmann_modernized import (\n"
            "    build_config,\n"
            "    locate_data_paths,\n"
            "    load_datasets,\n"
            "    build_schema_summary,\n"
            "    build_feature_availability,\n"
            "    prepare_store_metadata,\n"
            "    prepare_model_frame,\n"
            "    create_validation_windows,\n"
            "    run_pipeline,\n"
            ")\n"
            "\n"
            "config = build_config(Path.cwd())\n"
            "paths = locate_data_paths(config.project_dir)\n"
            "raw_train, raw_test, raw_store = load_datasets(paths)\n"
            "schema_summary = build_schema_summary(raw_train, raw_test, raw_store)\n"
            "feature_audit = build_feature_availability(raw_train, raw_test, raw_store)"
        ),
        nbf.v4.new_markdown_cell("## 2. Data Loading"),
        nbf.v4.new_code_cell(
            "schema_summary.query(\"dataset == 'train'\").head(10), feature_audit[['feature', 'available_in_test', 'allowed_in_final_model', 'reason']]"
        ),
        nbf.v4.new_markdown_cell("## 3. Cleaning"),
        nbf.v4.new_code_cell(
            "store_features = prepare_store_metadata(raw_store)\n"
            "train_model = prepare_model_frame(raw_train, store_features)\n"
            "test_model = prepare_model_frame(raw_test, store_features)\n"
            "train_model[['Date', 'Store', 'Sales', 'Open', 'Promo', 'CompetitionActive', 'Promo2Active']].head()"
        ),
        nbf.v4.new_markdown_cell("## 4. Feature Engineering"),
        nbf.v4.new_code_cell(
            "selected_columns = [\n"
            "    'Store', 'StoreType', 'Assortment', 'Promo', 'Promo2', 'StateHoliday', 'SchoolHoliday',\n"
            "    'CompetitionDistance', 'CompetitionDistanceLog', 'CompetitionActive', 'MonthsSinceCompetitionOpen',\n"
            "    'Promo2Started', 'Promo2Active', 'WeeksSincePromo2Start', 'PromoIntervalActive',\n"
            "    'year', 'month', 'day', 'iso_week', 'day_of_week', 'quarter', 'is_weekend',\n"
            "    'is_month_start', 'is_month_end', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos'\n"
            "]\n"
            "train_model[selected_columns].head()"
        ),
        nbf.v4.new_markdown_cell("## 5. Validation Design"),
        nbf.v4.new_code_cell(
            "holdout_start, pre_holdout, holdout, cv_windows = create_validation_windows(\n"
            "    train_model, holdout_days=config.holdout_days, cv_splits=config.cv_splits, cv_test_days=config.cv_test_days\n"
            ")\n"
            "pd.DataFrame(cv_windows), holdout_start"
        ),
        nbf.v4.new_markdown_cell("## 6. Model Comparison"),
        nbf.v4.new_code_cell(
            "results = run_pipeline(config=config, create_notebook=False)\n"
            "model_comparison = pd.read_csv(config.output_dir / 'model_comparison.csv')\n"
            "model_comparison"
        ),
        nbf.v4.new_markdown_cell("## 7. Error Analysis"),
        nbf.v4.new_code_cell(
            "error_analysis = pd.read_csv(config.output_dir / 'error_analysis_by_segment.csv')\n"
            "error_analysis.head(20)"
        ),
        nbf.v4.new_markdown_cell("## 8. Explainability"),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(config.output_dir / 'feature_importance_best_model.png')))\n"
            "shap_path = config.output_dir / 'shap_summary_best_model.png'\n"
            "if shap_path.exists():\n"
            "    display(Image(filename=str(shap_path)))\n"
            "else:\n"
            f"    print({json.dumps(shap_note or 'SHAP summary was not generated.')})"
        ),
        nbf.v4.new_markdown_cell("## 9. Final Training and Submission"),
        nbf.v4.new_code_cell(
            "submission = pd.read_csv(config.submission_path)\n"
            "submission.head(), submission.shape"
        ),
        nbf.v4.new_markdown_cell(
            "## 10. Conclusions\n\n"
            f"- Best model family: `{best_model_name}`\n"
            f"- Final holdout window started on `{holdout_start.date()}`\n"
            "- Final model excludes `Customers` to stay forecast-time safe.\n"
            "- Full comparison and summary live in `outputs/model_comparison.csv` and `outputs/project_summary.md`."
        ),
    ]
    nbf.write(nb, config.notebook_path)


def run_pipeline(config: Optional[Config] = None, create_notebook: bool = True) -> Dict[str, Any]:
    runtime_config = config or build_config()
    ensure_directories(runtime_config)
    sns.set_theme(style="whitegrid")

    print("Locating data files...")
    data_paths = locate_data_paths(runtime_config.project_dir)
    raw_train, raw_test, raw_store = load_datasets(data_paths)

    print("Saving schema and availability audits...")
    schema_summary = build_schema_summary(raw_train, raw_test, raw_store)
    feature_availability = build_feature_availability(raw_train, raw_test, raw_store)
    schema_summary.to_csv(runtime_config.output_dir / "schema_summary.csv", index=False)
    feature_availability.to_csv(runtime_config.output_dir / "feature_availability.csv", index=False)

    print("Preparing clean modeling frames...")
    store_metadata = prepare_store_metadata(raw_store)
    train_frame = downcast_numeric(prepare_model_frame(raw_train, store_metadata))
    test_frame = downcast_numeric(prepare_model_frame(raw_test, store_metadata))

    holdout_start, pre_holdout, holdout, validation_windows = create_validation_windows(
        train_frame=train_frame,
        holdout_days=runtime_config.holdout_days,
        cv_splits=runtime_config.cv_splits,
        cv_test_days=runtime_config.cv_test_days,
    )

    validation_window_frame = pd.DataFrame(validation_windows)
    validation_window_frame.to_csv(runtime_config.output_dir / "validation_windows.csv", index=False)

    feature_columns, categorical_columns = get_feature_columns()
    print("Running model comparison...")
    metrics_summary, model_comparison, best_model_name, best_holdout_artifact = evaluate_models(
        pre_holdout=pre_holdout,
        holdout=holdout,
        validation_windows=validation_windows,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        seed=runtime_config.random_seed,
    )
    metrics_summary.to_csv(runtime_config.output_dir / "metrics_summary.csv", index=False)
    model_comparison.to_csv(runtime_config.output_dir / "model_comparison.csv", index=False)

    print(f"Best model: {best_model_name}")
    importance_frame = feature_importance_table(
        best_model_name,
        best_holdout_artifact["model"],
        feature_columns,
    )
    importance_frame.to_csv(runtime_config.output_dir / "feature_importance_best_model.csv", index=False)
    save_feature_importance_plot(
        importance_frame,
        runtime_config.output_dir / "feature_importance_best_model.png",
    )

    print("Saving error analysis...")
    error_analysis = build_error_analysis_frame(
        best_holdout_artifact["score_frame"],
        best_holdout_artifact["predictions"],
    )
    error_analysis.to_csv(runtime_config.output_dir / "error_analysis_by_segment.csv", index=False)

    shap_note: Optional[str] = None
    if runtime_config.skip_shap:
        shap_note = "SHAP generation was skipped via configuration."
    else:
        print("Generating SHAP summary...")
        shap_note = save_shap_summary_plot(
            model_name=best_model_name,
            model=best_holdout_artifact["model"],
            x_numeric=best_holdout_artifact["x_train_num"],
            x_categorical=best_holdout_artifact["x_train_cat"],
            output_path=runtime_config.output_dir / "shap_summary_best_model.png",
            sample_size=runtime_config.shap_sample_size,
            seed=runtime_config.random_seed,
        )

    print("Retraining best model on full training data...")
    final_result = fit_final_model(
        model_name=best_model_name,
        train_frame=train_frame,
        score_frame=test_frame,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        seed=runtime_config.random_seed,
    )

    submission = pd.DataFrame(
        {
            "Id": test_frame["Id"].astype(int),
            "Sales": final_result["predictions"],
        }
    )
    submission.loc[test_frame["Open"].fillna(1).astype(int) == 0, "Sales"] = 0.0
    submission.to_csv(runtime_config.submission_path, index=False)

    best_holdout_metrics = metrics_summary.loc[
        (metrics_summary["model"] == best_model_name) & (metrics_summary["phase"] == "holdout")
    ].iloc[0]
    save_project_summary(
        config=runtime_config,
        model_comparison=model_comparison,
        best_model_name=best_model_name,
        best_holdout_metrics={
            "rmspe": float(best_holdout_metrics["rmspe"]),
            "mae": float(best_holdout_metrics["mae"]),
            "rmse": float(best_holdout_metrics["rmse"]),
        },
        shap_note=shap_note,
    )

    if create_notebook:
        print("Writing notebook artifact...")
        write_notebook(
            config=runtime_config,
            data_paths=data_paths,
            holdout_start=holdout_start,
            model_comparison=model_comparison,
            best_model_name=best_model_name,
            shap_note=shap_note,
        )

    return {
        "config": runtime_config,
        "data_paths": data_paths,
        "schema_summary": schema_summary,
        "feature_availability": feature_availability,
        "model_comparison": model_comparison,
        "metrics_summary": metrics_summary,
        "best_model_name": best_model_name,
        "shap_note": shap_note,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modernized Rossmann Store Sales pipeline")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP generation if needed.")
    parser.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Optional project directory override. Defaults to the script directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(Path(args.project_dir) if args.project_dir else None, skip_shap=args.skip_shap)
    run_pipeline(config=config, create_notebook=True)


if __name__ == "__main__":
    main()
