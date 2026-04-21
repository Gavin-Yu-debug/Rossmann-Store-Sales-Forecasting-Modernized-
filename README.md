# Rossmann-Store-Sales-Forecasting-Modernized-
Project Overview

This project is an upgraded version of a classic Rossmann sales forecasting notebook. Instead of simply reproducing the old workflow, I rebuilt the pipeline in a more realistic and reproducible way. The main goal was to make the forecasting process closer to how a real retail prediction task should be handled: using full training data, time-aware validation, leakage-safe feature engineering, model comparison, and error analysis.

Dataset
The project uses the Rossmann Store Sales dataset:
train.csv: 1,001,599 rows
test.csv: 1,115 rows
store.csv: 1,115 rows

The training data covers the period from 2013-01-01 to 2015-07-17, and the test date is 2015-07-31.

What I Improved

Compared with the original notebook, I made several changes:

used the full training dataset instead of only a small sample
replaced random split with expanding-window validation and a final 6-week holdout
removed leakage-prone features such as Customers from the final forecasting model
added stronger calendar, promotion, competition, and store-related features
added leakage-safe historical aggregate features
compared XGBoost, LightGBM, and CatBoost under the same validation setting
added feature importance, SHAP analysis, and segment-level error analysis
generated a final reproducible submission pipeline and saved outputs for later review
Feature Engineering

The main feature engineering work included:

calendar features such as month, quarter, ISO week, day of week, weekend flag, month start/end, and cyclic encodings
store/business features such as StoreType, Assortment, Promo, Promo2, StateHoliday, SchoolHoliday, and CompetitionDistance
competition timing features
promotion cycle features
historical aggregate features such as store average sales, median sales, store-by-day-of-week average, store-by-promo average, and store-type average sales
Validation Strategy

One of the main improvements was the validation design. Since this is a time-series retail forecasting problem, I did not use random train/test split. Instead, I used:

expanding-window time-series validation on earlier periods
a final 6-week holdout as the main model selection benchmark

The main evaluation metric was RMSPE, and I also checked MAE, RMSE, and bias. Closed-store predictions were explicitly set to zero.

Model Comparison

I benchmarked three tree-based models:
XGBoost
LightGBM
CatBoost

Among them, CatBoost performed best on the final holdout set and was selected as the final model. Final holdout results were:
CatBoost: RMSPE 0.1528, MAE 709.18, RMSE 1176.70
XGBoost: RMSPE 0.1737, MAE 833.99, RMSE 1369.33
LightGBM: RMSPE 0.1777, MAE 845.89, RMSE 1406.48
Key Findings

Some useful findings from the project:

promotion-related historical demand was the strongest signal
store-by-day-of-week historical demand was also highly important
competition timing mattered more than raw competition distance
the model tended to underpredict high-sales segments
StoreType b was one of the hardest segments to forecast accurately
Limitations

This project still has some limitations:

Customers was excluded from the final model for production realism
lag and rolling demand features were not added in the final version because I prioritized a stable one-shot prediction pipeline
the final models still showed negative bias on the holdout period
no external features such as weather or local events were available
Future Work

Possible next steps:

add lag and rolling features in a proper sequential forecasting setup
build a weighted ensemble of top models
add external data such as weather or local event signals
reduce underprediction in high-sales periods through calibration or loss reweighting
