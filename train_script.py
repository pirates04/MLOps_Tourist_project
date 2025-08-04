import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Tourist_Spending_Experiment")

df = pd.read_csv('tourist_data.csv')
df.fillna(df.median(numeric_only=True), inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('Spending', axis=1)
y = df['Spending']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse, r2 = evaluate_model(y_test, y_pred)
        mlflow.log_param("model_type", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(model, name.replace(" ", "_").lower() + "_model")
        print(f"{name} - RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
gs_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2')
gs_rf.fit(X_train, y_train)

rf_best = gs_rf.best_estimator_
y_pred_rf_best = rf_best.predict(X_test)
rmse_best, r2_best = evaluate_model(y_test, y_pred_rf_best)

with mlflow.start_run(run_name="Random Forest Tuned"):
    mlflow.log_params(gs_rf.best_params_)
    mlflow.log_metric("rmse", rmse_best)
    mlflow.log_metric("r2_score", r2_best)
    mlflow.sklearn.log_model(rf_best, "random_forest_best")
    print(f"Random Forest Tuned - RMSE: {rmse_best:.4f}, R2 Score: {r2_best:.4f}")

joblib.dump(rf_best, 'random_forest_best.pkl')