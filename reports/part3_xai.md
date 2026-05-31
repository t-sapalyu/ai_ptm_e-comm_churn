# Part 3 - Explainable AI (SHAP)

This section covers the XAI work on top of the Part 2 churn pipeline. It
uses SHAP to show why the model predicts churn.

## 1. Setup

- Library: `shap==0.46.0` (added to `requirements.txt`).
- Model explained: the registered XGBoost pipeline
  (`models:/churn_classifier/3`, promoted to Production in Part 2;
  test F1 = 0.854, ROC-AUC = 0.967 on de-duplicated data).
- Code: `src/churn/explain/shap_explain.py`
  (run with `python -m churn.explain.shap_explain` or `make explain`).
- Outputs: `outputs/xai/`.

## 2. Method

The Part 2 model is a scikit-learn Pipeline (ColumnTransformer preprocessor
plus classifier). `shap.TreeExplainer` only accepts a tree model, so we:

1. transform the 18 input columns with the fitted preprocessor (34 features),
2. build the TreeExplainer on the classifier step, and
3. plot the values against the original (un-scaled) feature values, so a
   waterfall reads `Tenure = 1` instead of `Tenure = -0.73`.

We explain the full test split (762 customers). Logistic regression is
skipped because TreeExplainer does not support it; the loader picks a tree
model (xgboost first, random_forest as fallback).

## 3. Generated plots (requirement -> file)

| PDF requirement | File |
| --- | --- |
| Explanation for a specific point | `01_single_point_bar.png`, `04_waterfall.png`, `05_force_plot.html` |
| Explanation for all points at once | `02_global_bar.png`, `07_beeswarm.png` |
| Summary plot for each class | `03a_summary_class0.png`, `03b_summary_class1.png` |
| Mean SHAP plot | `06_mean_shap.png` |
| Beeswarm plot | `07_beeswarm.png` |
| Dependence plots | `08_dependence_top1.png`, `09_dependence_top2.png` |

## 4. Key findings

Top churn drivers by mean |SHAP| on the test set:

| Rank | Feature | Mean \|SHAP\| | Reading |
| --- | --- | --- | --- |
| 1 | `Tenure` | 2.75 | Short tenure is the strongest churn signal |
| 2 | `Complain` | 1.27 | Complaints raise churn risk |
| 3 | `NumberOfAddress` | 0.95 | |
| 4 | `CashbackAmount` | 0.92 | Lower cashback links to churn |
| 5 | `MaritalStatus_Single` | 0.61 | Single customers churn more |
| 6 | `DaySinceLastOrder` | 0.58 | Longer gaps raise risk |
| 7 | `WarehouseToHome` | 0.56 | Longer delivery distance raises churn |
| 8 | `SatisfactionScore` | 0.54 | |

These match the drivers from Part 1 (tenure, satisfaction, complaints,
recency), so the model is using sensible signals.

Single-customer example (highest-risk row, customer #13): predicted churn
probability 1.000, model margin f(x) = 9.9 vs base value 0.23. The waterfall
puts most of this on short tenure, a logged complaint, and recency.

## 5. Reproduce

```bash
python -m churn.data.make_dataset     # build splits
python -m churn.models.train          # train + register models
python -m churn.models.evaluate       # promote best to Production
python -m churn.explain.shap_explain  # outputs to outputs/xai/
```
