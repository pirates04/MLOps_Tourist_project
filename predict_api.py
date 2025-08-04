from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Tourist Spending Prediction API")

# Define request schema
class InputData(BaseModel):
    features: list  # list of feature values in order

# Load model from MLflow model registry
model_name = "TouristSpendingModel"
model_stage = "Production"  # or use "Staging"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

@app.post("/predict")
def predict(data: InputData):
    try:
        input_df = pd.DataFrame([data.features])
        prediction = model.predict(input_df)[0]
        return {"predicted_spending": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))