from flask import Flask, render_template, request
import mlflow
import pandas as pd
import time
import dagshub
import warnings
import yaml
import joblib
from sklearn.preprocessing import StandardScaler
import os
warnings.filterwarnings("ignore")

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "ayazr425"
# repo_name = "Heart-Disease-Pred-proj"
# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri('https://dagshub.com/ayazr425/Heart-Disease-Pred-proj.mlflow')
dagshub.init(repo_owner='ayazr425', repo_name='Heart-Disease-Pred-proj', mlflow=True)
# -------------------------------------------------------------------------------------

# -------------------------- FLASK APP -------------------------------------
app = Flask(__name__)

# ----------------------- GET MODEL FROM MLFLOW ----------------------------
model_name = "my_model"

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["staging"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
print(f"Fetching model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

# -------------------- DOWNLOAD SCALER FROM MLFLOW ARTIFACTS ---------------
def download_scaler_from_mlflow(model_name, model_version, artifact_name="scaler.pkl"):
    client = mlflow.MlflowClient()
    run_id = client.get_model_version(name=model_name, version=model_version).run_id
    scaler_local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_name)
    return joblib.load(scaler_local_path)

scaler = download_scaler_from_mlflow(model_name, model_version)

# ------------------- LOAD COLUMNS TO SCALE FROM PARAMS.YAML --------------
def load_params(path="params.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

params = load_params()
columns_to_scale = params['feature_engineering']['columns_to_scale']

# --------------------------- ROUTES ---------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form
        input_data = [
            float(form_data['age']),
            int(form_data['sex']),
            int(form_data['cp']),
            float(form_data['trestbps']),
            float(form_data['chol']),
            int(form_data['fbs']),
            int(form_data['restecg']),
            float(form_data['thalach']),
            int(form_data['exang']),
            float(form_data['oldpeak']),
            int(form_data['slope']),
            int(form_data['ca']),
            int(form_data['thal'])
        ]

        columns = [
            'Age', 'Sex', 'Chest pain type', 'Blood Pressure', 'Cholesterol',
            'Sugre Blood Value', 'EKG results', 'Max heart rate', 'Exercise angina',
            'Segment Depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
        ]

        input_df = pd.DataFrame([input_data], columns=columns)

        # Use downloaded scaler from MLflow
        input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])

        prediction = model.predict(input_df)[0]

        return render_template("index.html", result=int(prediction))

    except Exception as e:
        print(f"Prediction Error: {e}")
        return render_template("index.html", result="Error: Invalid input")

# --------------------------- START APP ------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
