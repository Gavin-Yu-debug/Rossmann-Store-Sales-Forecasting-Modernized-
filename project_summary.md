# Rossmann Modernized Project Summary

## Major Improvements Over the Legacy Notebook
- Replaced the legacy 70k-row sample with the full training history when fitting the modern pipeline.
- Replaced random train/test splitting with expanding-window validation plus a final 6-week holdout.
- Removed forecast-time-invalid leakage risks from the final model, especially `Customers`.
- Added stronger business features for promotions, competition timing, calendar seasonality, and store metadata.
- Added leakage-safe historical aggregate features that are re-fit inside each time split.
- Compared XGBoost, LightGBM, and CatBoost on the same validation design instead of relying on one model.
- Added error slicing, feature importance, SHAP explainability, artifact saving, and a reproducible script.

## Final Selected Model
- Best model: `catboost`
- Best holdout RMSPE: `0.15284`
- Holdout MAE: `709.18`
- Holdout RMSE: `1176.70`

## Model Comparison
|   rank | model    |   cv_mean_rmspe |   cv_std_rmspe |   holdout_rmspe |   holdout_mae |   holdout_rmse |   holdout_bias |   total_runtime_seconds | params_json                                                                                                                                                                                                                                                                                  |
|-------:|:---------|----------------:|---------------:|----------------:|--------------:|---------------:|---------------:|------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      1 | catboost |        0.168779 |      0.043509  |        0.152845 |       709.178 |        1176.7  |       -338.839 |                490.258  | {"iterations": 600, "learning_rate": 0.05, "depth": 6, "loss_function": "RMSE", "eval_metric": "RMSE", "subsample": 0.8, "l2_leaf_reg": 5.0, "bootstrap_type": "Bernoulli", "random_seed": 42, "allow_writing_files": false, "thread_count": -1, "verbose": false}                           |
|      2 | xgboost  |        0.182019 |      0.0291392 |        0.173659 |       833.987 |        1369.33 |       -403.903 |                 18.3962 | {"n_estimators": 900, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 30, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0, "objective": "reg:squarederror", "tree_method": "hist", "early_stopping_rounds": 100, "random_state": 42, "n_jobs": -1} |
|      3 | lightgbm |        0.157867 |      0.0393331 |        0.17767  |       845.888 |        1406.48 |       -398.001 |                 59.1487 | {"n_estimators": 900, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 40, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.05, "reg_lambda": 0.1, "objective": "regression", "random_state": 42, "n_jobs": -1, "verbosity": -1, "force_col_wise": true}               |

## Top Business Insights
- Promotion and store-level historical average features carried the strongest predictive signal.
- Competition timing mattered more than raw competition distance alone, especially once activation timing was included.
- Calendar structure around month boundaries and weekends remained important even after promotion effects were modeled.

## Leakage-Avoidance Decisions
- `Customers` was intentionally excluded from the final production-safe feature set even though it appears in Kaggle test data.
- Historical aggregates were re-fit within each training window and mapped forward only to later periods.
- Closed-store rows were handled through explicit business logic instead of letting the model learn zero sales from future-like labels.

## Explainability Note
- SHAP summary plot generated successfully.

## Next-Step Ideas
- Add store-level lagged demand features only if a rolling forecasting setup is introduced for future horizons.
- Try a weighted ensemble of the top two boosting models.
- Add richer promotion calendars or external holiday/weather data if a true production forecast stack is needed.