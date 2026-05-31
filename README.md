# ai_ptm_e-comm_churn
AI Project Technical Methodologies - Customer Churn Project

# clone the repo
git clone <repo-url>
cd ai_ptm_e-comm_churn

# create the environment (if Windows)
 uv venv --python 3.11 aim_env
source aim_env/Scripts/activate

# install requirements.txt
uv pip install -r requirements.txt
pip install -e .

# Place the dataset (use copy, drag-drop, or Explorer)
copy <path-to>\E_Commerce_Dataset.xlsx data\raw\

# Build splits, train all three models, evaluate, serve
# (if running from visual studio code)
python -m churn.data.make_dataset
python -m churn.models.train
python -m churn.models.evaluate

# to browse the runs in the MLflow UI (http://127.0.0.1:5000)
mlflow ui 

# deploying to a local server (in case we want to integrate predictions into an application or demo client for the local MLflow serving endpoint)
mlflow models serve -m "models:/churn_classifier/Production" -p 5001 --no-conda

# if want to predict with new data
python -m churn.models.predict --input customers.csv --output preds.csv
# if want to rollback to any archived model version 
python -m churn.models.rollback --list
python -m churn.models.rollback --version 3

# After serving, in a second terminal, make inference request to test serving
python -m churn.inference.request

# Part 3 - Explainable AI (SHAP)
# Explain the registered tree model with SHAP. TreeExplainer is run on the
# classifier step of the pipeline over the preprocessed features. Defaults to
# the xgboost version; falls back to random_forest. logreg is not explainable
# with TreeExplainer.
python -m churn.explain.shap_explain --model xgboost --sample-size 400

# All figures are written to outputs/xai/ (committed, NOT under reports/):
#   01_single_point_bar.png   single customer, bar
#   02_global_bar.png         all customers, mean|SHAP| bar
#   03a_summary_class0.png    summary plot - no-churn class
#   03b_summary_class1.png    summary plot - churn class
#   04_waterfall.png          single customer, waterfall
#   05_force_plot.html        single customer, interactive force plot
#   06_mean_shap.png          mean|SHAP| feature importance
#   07_beeswarm.png           beeswarm over all customers
#   08_dependence_top1.png    dependence plot, top feature
#   09_dependence_top2.png    dependence plot, 2nd feature