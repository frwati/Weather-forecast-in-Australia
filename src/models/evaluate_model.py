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
    """
    Load the previous model from BentoML and evaluate its F1 score.
    
    Parameters:
        X_test (DataFrame): Test features.
        y_test (Series): Test target.
    
    Returns:
        f1_previous (float): F1 score of the previous model.
    """
    try:
        # Load the previous model stored in BentoML
        previous_model = bentoml.sklearn.load_model("weather_rf_model:latest")
        
        # If the model is loaded successfully, evaluate it
        if previous_model is not None:
            logging.info("Evaluating previous model from BentoML...")
            metrics, _ = evaluate_model(X_test, y_test, previous_model)
            return metrics["f1_score"]
        else:
            logging.error("No previous model found in BentoML.")
            return None
    
    except Exception as e:
        logging.error(f"Error loading model from BentoML: {e}")
        return None
    

def should_update_model(current_metrics, previous_metrics):
    """
    Compare the F1 score of the current model and previous model to decide whether to update.
    
    Parameters:
        current_metrics (dict): Metrics of the current model.
        previous_metrics (float): F1 score of the previous model.
    
    Returns:
        bool: True if the current model should be updated; False otherwise.
    """
    f1_current = current_metrics["f1_score"]
    f1_previous = previous_metrics
    return f1_current >= f1_previous


def main(input_dir="data/normalized_data", output_dir="metrics", model_dir="models"):
    """
    Main function to evaluate a trained model and save metrics and predictions.
    
    Parameters:
        input_dir (str): Directory containing test data.
        output_dir (str): Directory to save evaluation metrics and predictions.
        model_dir (str): Directory containing the trained model.
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    try:
        # Load the test data
        logging.info("Loading test data...")
        X_test = pd.read_csv(os.path.join(input_dir, "X_test_scaled.csv"))
        y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv")).values.ravel()

        # Load the trained model
        model_path = os.path.join(model_dir, "trained_model.pkl")
        if not os.path.exists(model_path):
            logging.error(f"Trained model file not found at {model_path}. Exiting.")
            return

        logging.info(f"Loading trained model from {model_path}...")
        model = joblib.load(model_path)

        # Evaluate the current model and get metrics and predictions
        metrics, predictions = evaluate_model(X_test, y_test, model)
       
        # Save metrics and predictions
        save_metrics(metrics, os.path.join(output_dir, "scores.json"))
        save_predictions(predictions, os.path.join(output_dir, "predictions.csv"))
        
        # Evaluate the previous model from BentoML and decide if it should be updated
        previous_metrics = load_previous_model_from_bentoml(X_test, y_test)
        if previous_metrics is not None:
            if should_update_model(metrics, previous_metrics):
                logging.info("Current model performs better. Updating BentoML model.")
                bentoml.sklearn.save_model("weather_rf_model", model)
            else:
                logging.info("Current model does not perform better. No update to BentoML model.")
        else:
            logging.info("No previous model in BentoML. Saving the current model.")
            bentoml.sklearn.save_model("weather_rf_model", model)

    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")


if __name__ == "__main__":
    main()
