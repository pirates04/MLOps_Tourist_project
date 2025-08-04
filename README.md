
# 🧭 Tourist Spending Prediction Pipeline

An end-to-end Machine Learning pipeline that predicts tourist spending using modern MLOps tools, built by **Shikhar Bhargava**.

This project demonstrates a production-ready workflow from data preprocessing to deployment, with MLflow tracking, DVC-based version control, and containerized deployment using Docker and FastAPI.

---

## 🚀 Features

- 📊 Exploratory Data Analysis (EDA) in Google Colab
- 🎯 Model training with hyperparameter tuning
- 🤖 Comparison of multiple ML algorithms
- 🧪 Experiment tracking using MLflow
- 📈 Evaluation with custom metrics and visualizations
- 📦 Data version control using DVC
- 🔁 CI/CD setup using GitHub Actions
- 🚀 FastAPI-based prediction API
- 🐳 Containerized with Docker for deployment

---

## 🧰 Tech Stack

- **Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn, Matplotlib
- **MLOps Tools:** MLflow, DVC
- **Deployment:** FastAPI, Docker
- **Automation:** GitHub Actions

---

## 📁 Folder Structure

```
MLPipelineProject/
│
├── data/                          # (Optional) DVC-tracked dataset
│   └── tourist_data.csv
│
├── notebooks/
│   └── mlflow_colab.ipynb         # EDA + MLflow tracking in Colab
│
├── scripts/
│   ├── model.py                   # Model architecture/logic
│   ├── train_script.py            # Training and evaluation
│   ├── metrics.py                 # Custom evaluation metrics
│   └── mlflow_model_registry.py   # Register models with MLflow
│
├── deployment/
│   ├── predict_api.py             # FastAPI endpoint
│   └── Dockerfile                 # Docker container config
│
├── .github/workflows/
│   └── github_workflows.yml       # CI/CD pipeline
│
├── dvc.yaml                       # DVC pipeline stages
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview (this file)
└── .gitignore / .dvcignore        # Ignore settings
```

---

## 📊 Model Training and Evaluation

To train and evaluate the models:

```bash
python scripts/train_script.py
```

You can configure or tune hyperparameters directly in the script. The training logs and metrics will be tracked in MLflow.

---

## 🧪 MLflow (Colab Friendly)

To use MLflow tracking on Colab:

1. Open `notebooks/mlflow_colab.ipynb`
2. Set up `mlflow.set_tracking_uri(...)`
3. Run the notebook to log experiments and visualize metrics

---

## 📡 Running the FastAPI App

To start the API server:

```bash
uvicorn deployment.predict_api:app --reload
```

### Example Endpoint:
- `POST /predict`: Send JSON input to receive predicted spending value

---

## 🐳 Docker Deployment

To containerize and run the app:

```bash
docker build -t tourist-predictor .
docker run -p 8000:8000 tourist-predictor
```

Access the API at `http://localhost:8000/docs`

---

## 🔁 CI/CD with GitHub Actions

The file `.github/workflows/github_workflows.yml` includes a sample workflow you can extend for:
- Linting
- Testing
- Docker builds
- Deployments

---

## 🔄 DVC Setup

If using DVC for data tracking:

```bash
dvc init
dvc pull                # Pull tracked data from remote
dvc repro               # Re-run the full pipeline
```

Make sure `tourist_data.csv` is added via `dvc add data/tourist_data.csv`.

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

## ✍️ Author

**Shikhar Bhargava**  
GitHub: https://github.com/pirates04  


---
