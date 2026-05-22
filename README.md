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