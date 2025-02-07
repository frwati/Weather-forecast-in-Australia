import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import logging
import joblib
import bentoml


def train_model(X_train, y_train, best_params=None):
    """
    Trains a RandomForestClassifier using optional best hyperparameters.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series or np.array): Training target data.
        best_params (dict, optional): Best hyperparameters for the RandomForestClassifier. Defaults to None.

    Returns:
        RandomForestClassifier: Trained RandomForestClassifier model.
    """
    # Initialize the model with or without best_params
    if best_params:
        model = RandomForestClassifier(**best_params)
    else:
        model = RandomForestClassifier()

    logging.info("Training RandomForestClassifier model...")
    model.fit(X_train, y_train)
    
    return model


def save_trained_model(model, output_dir="models"):
    """
    Saves the trained model to a specified file using BentoML.

    Args:
        model (RandomForestClassifier): Trained model.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model locally for DVC tracking
    model_path = os.path.join(output_dir, "trained_model.pkl")
    joblib.dump(model, model_path)
    logging.info(f"Model saved locally at {model_path}")


def main(input_dir="data/normalized_data", output_dir="models"):
    """
    Main function to train and save a RandomForestClassifier model.

    Args:
        input_dir (str): Path to the directory containing preprocessed training data.
        output_dir (str): Path to save the trained model.
    """
    # Set up logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Load the training data
    logging.info("Loading training data...")
    X_train = pd.read_csv(os.path.join(input_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).values.ravel()

    # Load the best parameters for GridSearchCV
    best_params_file = os.path.join(output_dir, "best_params.pkl")
    if os.path.exists(best_params_file):
        logging.info(f"Loading best parameters from {best_params_file}...")
        best_params = joblib.load(best_params_file)
    else:
        logging.warning("Best parameters file not found. Proceeding with default parameters.")
        best_params = None

    # Train the model using the best parameters
    model = train_model(X_train, y_train, best_params=best_params)

    # Save the model in BentoML & DVC
    save_trained_model(model, output_dir)


if __name__ == "__main__":
    main()
