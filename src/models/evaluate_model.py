import pandas as pd
import joblib
import logging
import os
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import json


def evaluate_model(X_test, y_test, model):
    """
    Evaluate the model using test data and calculate performance metrics.
    
    Parameters:
        X_test (DataFrame): Test features.
        y_test (Series): Test target.
        model (sklearn model): Trained model.
    
    Returns:
        metrics (dict): Model evaluation metrics (Accuracy, F1, MSE, R2).
        predictions (DataFrame): Model predictions on test set.
    """
    logging.info("Making predictions on the test set...")
    predictions = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='binary')  # Binary classification assumption
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "mean_squared_error": mse,
        "r2_score": r2
    }
    
    logging.info(f"Evaluation metrics calculated: Accuracy={accuracy}, F1={f1}, MSE={mse}, R2={r2}")
    
    return metrics, predictions


def save_metrics(metrics, filename="metrics/scores.json"):
    """
    Save model evaluation metrics to a JSON file.
    
    Parameters:
        metrics (dict): Model evaluation metrics.
        filename (str): Path to save the metrics.
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logging.info(f"Saving metrics to {filename}...")
        with open(filename, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info("Metrics saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save metrics: {e}")


def save_predictions(predictions, filename="metrics/predictions.csv"):
    """
    Save model predictions to a CSV file.
    
    Parameters:
        predictions (DataFrame): Model predictions on the test set.
        filename (str): Path to save the predictions.
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logging.info(f"Saving predictions to {filename}...")
        pd.DataFrame(predictions, columns=["Predicted_RainTomorrow"]).to_csv(filename, index=False)
        logging.info("Predictions saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save predictions: {e}")


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

        # Evaluate the model and get metrics and predictions
        metrics, predictions = evaluate_model(X_test, y_test, model)

        # Save metrics and predictions
        save_metrics(metrics, os.path.join(output_dir, "scores.json"))
        save_predictions(predictions, os.path.join(output_dir, "predictions.csv"))

    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")


if __name__ == "__main__":
    main()
