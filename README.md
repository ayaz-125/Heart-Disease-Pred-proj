# 🫀 Heart Disease Prediction - End-to-End MLOps Project

This is a complete end-to-end MLOps project for heart disease prediction using tools like DVC, MLflow, DagsHub, Flask, and AWS S3,AWS CodePipeline. It demonstrates a production-ready ML pipeline with experiment tracking, model versioning, and cloud integration.

---

## 📁 Project Setup

### Step 1: Initialize Project

```Terminal
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
conda create -n heartpro python=3.10 -y
conda activate heartpro
```

### Step 2: Setup Folder Structure

```Terminal
pip install cookiecutter
cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
```

- Rename `src.models` to `src.model`
- Rename `data/` folder to `data_fol/`
- Add `data_fol/` to `.gitignore`

```Terminal
git add .
git commit -m "Initial setup"
git push
```

---

## 🚀 MLflow Integration with DagsHub

### Step 3: Connect GitHub to DagsHub

- Go to: https://dagshub.com/dashboard
- Create new repo → Connect GitHub repo
- Copy the MLflow tracking code snippet

### Step 4: Install MLflow and DagsHub SDK

```Terminal
pip install dagshub 
pip install mlflow 
pip insyall ipykernel
pip install -r requirements.txt
```

### Step 5: Run Experiment Notebook

```Terminal
jupyter notebook
# Run experiment notebook and track via MLflow
```

```Terminal
git add .
git commit -m "Added experiment logs"
git push
```

---

## 📦 DVC Pipeline Setup

### Step 6: Initialize DVC

```Terminal
dvc init
mkdir local_s3
dvc remote add -d mylocal local_s3
```

### Step 7: Add Pipeline Code (inside src/)

- `logger.py`
- `data_ingestion.py`
- `data_preprocessing.py`
- `feature_engineering.py`
- `model_building.py`
- `model_evaluation.py`
- `register_model.py`

Add the following files to root:
- `dvc.yaml`
- `params.yaml`

```Terminal
dvc repro
dvc status
git add .
git commit -m "DVC pipeline added"
git push
```

---

## ☁️ AWS S3 Integration for DVC

### Step 8: Setup AWS

- Create IAM user with programmatic access
- Create S3 bucket (e.g., `heart-proj-buck`)

### Step 9: Install AWS Tools

```Terminal
pip install "dvc[s3]" awscli
```

### Step 10: ✅ Environment Variable Setup

```Terminal
$env:AWS_ACCESS_KEY_ID="your_access_key_id"
$env:AWS_SECRET_ACCESS_KEY="your_secret_access_key"
$env:BUCKET_NAME="your_bucket_name"
```

### Step 11: Connect S3 as DVC Remote

```Terminal
dvc remote add -d myremote s3://your_bucket_name
dvc push
```

---

## 🌐 Flask App for Model Serving

### Step 12: Create Flask App Directory

```Terminal
mkdir flaskapp
cd flaskapp
# Add Flask app files like app.py, templates/,pro_requirements.txt,params.yaml, etc.
```

### Step 13: Run Flask App

```Terminal
pip install flask
python app.py
```

---

## 🔐 DagsHub Token Setup

### Step 14: Generate Token

- Go to DagsHub > Repo > Settings > Access Tokens
- Click **Generate New Token**
- Name: `CAPSTONE_TEST`
- Copy the token

### Step 15: Set Token in Terminal

```Terminal
$env:CAPSTONE_TEST="your_token_here"
```

---

<!-- ## ✅ Environment Variable Setup

```powershell
$env:AWS_ACCESS_KEY_ID="your_access_key_id"
$env:AWS_SECRET_ACCESS_KEY="your_secret_access_key"
$env:BUCKET_NAME="your_bucket_name" -->


---

## 📌 .gitignore Recommendations

```gitignore
data_fol/
local_s3/
__pycache__/
*.log
*.env
mlruns/
```

---

## ✅ Final Checklist

- [x] Project structure initialized
- [x] MLflow tracking via DagsHub
- [x] DVC pipeline created and reproducible
- [x] AWS S3 remote configured
- [x] Flask app deployed for predictions
- [x] Secure environment variables and token set

---



## ⚙️ CI/CD with AWS

### Plan:

- Create a pipeline that:

- Pulls code from GitHub

- Builds with CodeBuild

- Deploys to EC2 with CodeDeploy

#### AWS Services Used:

- IAM - Create roles for EC2 and CodeDeploy

- EC2 - Launch instance, install required packages

- S3 - Stores build artifacts

- CodeBuild - Builds app from repo using buildspec.yaml

- CodeDeploy - Deploys to EC2 instance

- CodePipeline - Orchestrates CI/CD process

#### Before AWS Console Work

- Copy logger.py and params.yaml into flaskapp/

- Git add, commit, and push

#### AWS Console Steps:

- IAM Setup

- Create roles for EC2 and CodeDeploy with relevant policies

#### EC2 Setup:

- Launch Ubuntu 22.04 instance

- Allow HTTP, HTTPS, and port 5000 in security group

- Attach IAM role to EC2

- SSH into EC2 and run bash install.sh script to install dependencies

### CodeDeploy:

- Create Application & Deployment Group

- Attach IAM role and EC2 instance tag

### CodePipeline

- Source Stage: GitHub repo as source

- Build Stage: CodeBuild using buildspec.yaml

- Deploy Stage: CodeDeploy using created app and deployment group

#### 🚤 Deploy and Run

- SSH into EC2
- Navigate to project folder:
- cd /home/ubuntu/flaskapp
- pip3 install -r prod_requirements.txt
- python3 app.py

#### Authorize with DagsHub if prompted

- Access app in browser:
-http://<EC2-public-IP>:5000

#### 📊 Tools & Tech Stack

- Python, Pandas, Scikit-learn

- DVC, MLflow, Flask

- DagsHub, GitHub, AWS S3, EC2, CodeBuild, CodeDeploy, CodePipeline

- Docker (optional), GitHub Actions (optional)

🙌 Acknowledgements

Inspired by industry-level ML workflows and MLOps best practices.
Thanks to open-source projects like DVC, MLflow, and DagsHub.


---

## 🙌 Acknowledgements

Inspired by industry-level ML workflows and MLOps best practices.  
Thanks to open-source projects like [DVC](https://dvc.org), [MLflow](https://mlflow.org), and [DagsHub](https://dagshub.com).

---
