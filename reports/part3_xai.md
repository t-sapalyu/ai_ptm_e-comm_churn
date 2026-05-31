# Part 3 — Explainable AI (SHAP)

This section documents the XAI work added on top of the Part 2 churn
pipeline. It explains **why** the production model predicts churn, using
SHAP (SHapley Additive exPlanations).

## 1. Setup

- Library: `shap==0.46.0` (added to `requirements.txt`).
- Model explained: the registered **XGBoost** pipeline
  (`models:/churn_classifier/3`, the version promoted to *Production* in
  Part 2, test F1 = 0.940).
- Code: `src/churn/explain/shap_explain.py`
  (run with `python -m churn.explain.shap_explain` or `make explain`).
- Outputs: `outputs/xai/` (committed; deliberately **not** under the
  git-ignored `reports/` HTML path).

## 2. Method — explaining a Pipeline, not a bare model

The Part 2 model is a scikit-learn `Pipeline` of a `ColumnTransformer`
preprocessor (median imputation + scaling + one-hot encoding) followed by
the classifier. `shap.TreeExplainer` only accepts a tree model, so we:

1. transform the raw 18 input columns with the fitted `preprocessor`,
   producing **34 engineered features**,
2. build the `TreeExplainer` on the bare `classifier` step, labelling the
   Shapley values with `preprocessor.get_feature_names_out()`, and
3. plot the values against the **original** (un-scaled) feature values, so
   a waterfall reads `Tenure = 1` rather than `Tenure = -0.73`.

The full held-out **test** split (846 customers) is explained. Logistic
regression is intentionally excluded — `TreeExplainer` does not support it;
the loader picks a tree family (xgboost preferred, random_forest fallback,
both paths handled).

## 3. Generated visualisations (requirement → file)

| PDF requirement | File |
| --- | --- |
| Explanation for a specific point | `01_single_point_bar.png`, `04_waterfall.png`, `05_force_plot.html` |
| Explanation for all points at once | `02_global_bar.png`, `07_beeswarm.png` |
| Summary plot for each class | `03a_summary_class0.png`, `03b_summary_class1.png` |
| Mean SHAP plot | `06_mean_shap.png` |
| Beeswarm plot | `07_beeswarm.png` |
| Dependence plots | `08_dependence_top1.png`, `09_dependence_top2.png` |

## 4. Key findings

Top churn drivers by mean |SHAP| on the test sample:

| Rank | Feature | Mean \|SHAP\| | Reading |
| --- | --- | --- | --- |
| 1 | `Tenure` | 2.63 | Short tenure is by far the strongest churn signal |
| 2 | `Complain` | 1.24 | Having complained sharply raises churn risk |
| 3 | `CashbackAmount` | 0.94 | Lower cashback associates with churn |
| 4 | `NumberOfAddress` | 0.93 | |
| 5 | `WarehouseToHome` | 0.70 | Longer delivery distance pushes churn up |
| 6 | `DaySinceLastOrder` | 0.63 | Recency: longer gaps raise risk |
| 7 | `MaritalStatus_Single` | 0.53 | Single customers churn more |
| 8 | `SatisfactionScore` | 0.51 | |
| 9 | `CityTier` | 0.50 | |
| 10 | `PreferedOrderCat_Laptop & Accessory` | 0.41 | |

These align with the Part 1 framing (tenure, satisfaction, complaints and
recency as retention levers), which is a good sanity check that the model
learned business-plausible behaviour rather than spurious correlations.

**Single-customer example (auto-selected highest-risk, customer #80):**
the model margin is f(x) = 10.4 against a base/expected value of 0.24, i.e.
a strong churn score. The waterfall attributes this mainly to a very short
`Tenure = 1`, a logged `Complain = 1`, and a high `NumberOfAddress = 8` —
exactly the levers Part 1 flagged for retention.

## 5. Reproduce

```bash
python -m churn.data.make_dataset     # build splits
python -m churn.models.train          # train + register models
python -m churn.models.evaluate       # promote best to Production
python -m churn.explain.shap_explain  # -> outputs/xai/
```
