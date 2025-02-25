import os
from pathlib import Path
import pandas as pd
import joblib
import logging
import json
import mlflow
import bentoml
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# Set project root based on environment
if os.path.exists("/.dockerenv"):
    PROJECT_ROOT = Path("/opt/airflow")  # Docker
else:
    PROJECT_ROOT = Path.cwd()

# Define directories based on PROJECT_ROOT
INPUT_DIR = PROJECT_ROOT / "data/normalized_data"
OUTPUT_DIR = PROJECT_ROOT / "metrics"
MODEL_DIR = PROJECT_ROOT / "models/saved_models"


def evaluate_model(X_test, y_test, model):
    logging.info("Making predictions on the test set...")
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='binary')
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "mean_squared_error": mse,
        "r2_score": r2
    }
    
    logging.info(f"Evaluation metrics: {metrics}")
    
    mlflow.set_experiment("Weather_prediction")
    with mlflow.start_run():
        mlflow.log_params({"model_type": "RandomForest"})
        mlflow.log_metrics(metrics)
    
    return metrics, predictions


def save_metrics(metrics, filename=OUTPUT_DIR / "scores.json"):
    try:
        filename.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving metrics to {filename}...")
        with open(filename, "w") as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to save metrics: {e}")


def save_predictions(predictions, filename=OUTPUT_DIR / "predictions.csv"):
    try:
        filename.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving predictions to {filename}...")
        pd.DataFrame(predictions, columns=["Predicted_RainTomorrow"]).to_csv(filename, index=False)
    except Exception as e:
        logging.error(f"Failed to save predictions: {e}")


def load_previous_model_from_bentoml(X_test, y_test):
    try:
        previous_model = bentoml.sklearn.load_model("weather_rf_model:latest")
        if previous_model:
            logging.info("Evaluating previous model...")
            metrics, _ = evaluate_model(X_test, y_test, previous_model)
            return metrics["f1_score"]
    except Exception as e:
        logging.error(f"Error loading model: {e}")
    return None


def should_update_model(current_metrics, previous_metrics):
    return current_metrics["f1_score"] >= previous_metrics


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    try:
        logging.info("Loading test data...")
        X_test = pd.read_csv(INPUT_DIR / "X_test_scaled.csv")
        y_test = pd.read_csv(INPUT_DIR / "y_test.csv").values.ravel()
        
        model_path = MODEL_DIR / "trained_model.pkl"
        if not model_path.exists():
            logging.error(f"Model not found at {model_path}. Exiting.")
            return

        logging.info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        
        metrics, predictions = evaluate_model(X_test, y_test, model)
        save_metrics(metrics)
        save_predictions(predictions)
        
        previous_metrics = load_previous_model_from_bentoml(X_test, y_test)
        if previous_metrics is not None:
            if should_update_model(metrics, previous_metrics):
                logging.info("Updating BentoML model...")
                bentoml.sklearn.save_model("weather_rf_model", model)
            else:
                logging.info("Current model does not improve. No update.")
        else:
            logging.info("No previous model found. Saving current model.")
            bentoml.sklearn.save_model("weather_rf_model", model)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()
