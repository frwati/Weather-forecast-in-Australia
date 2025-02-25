import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import logging
import joblib
import bentoml



# Detect if running inside a Docker container
if os.path.exists("/.dockerenv"):
    PROJECT_ROOT = "/opt/airflow"  # Adjust based on your Docker setup
else:
    PROJECT_ROOT = os.getcwd()

logging.info(f"Running inside: {PROJECT_ROOT}")

def train_model(X_train, y_train, best_params=None):
    """
    Trains a RandomForestClassifier using optional best hyperparameters.
    """
    model = RandomForestClassifier(**best_params) if best_params else RandomForestClassifier()
    logging.info("Training RandomForestClassifier model...")
    model.fit(X_train, y_train)
    return model

def save_trained_model(model, filename):
    """
    Saves the trained model to a specified file using joblib and BentoML.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Saving with joblib
    joblib.dump(model, filename)
    logging.info(f"Model saved at {filename}")
    
    # Saving with BentoML
    bentoml.sklearn.save_model('weather_rf_model', model)
    logging.info(f"Model saved in BentoML store with name 'weather_rf_model'.")

def main():
    """
    Main function to train and save a RandomForestClassifier model.
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    # Define adaptable paths
    input_dir = os.path.join(PROJECT_ROOT, "data/normalized_data")
    input_dir_grid = os.path.join(PROJECT_ROOT, "models/best_parameters")
    output_dir_save = os.path.join(PROJECT_ROOT, "models/saved_models")
    
    # Load training data
    logging.info("Loading training data...")
    X_train = pd.read_csv(os.path.join(input_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).values.ravel()
    
    # Load best parameters if available
    best_params_file = os.path.join(input_dir_grid, "best_params.pkl")
    best_params = joblib.load(best_params_file) if os.path.exists(best_params_file) else None
    if best_params:
        logging.info(f"Using best parameters: {best_params}")
    else:
        logging.warning("Best parameters file not found. Proceeding with default parameters.")
    
    # Train model
    model = train_model(X_train, y_train, best_params)
    
    # Save model using both joblib and BentoML
    save_trained_model(model, os.path.join(output_dir_save, "trained_model.pkl"))

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


<<<<<<< HEAD
def save_trained_model(model, output_dir="models"):
=======
<<<<<<< HEAD
def save_trained_model(model, output_dir="models"):
=======
def save_trained_model(model, filename="../../models/saved_models/trained_model.pkl"):
>>>>>>> origin/main
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
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


def main(input_dir="../../data/normalized_data", output_dir_save="../../models/saved_models" ,input_dir_grid="../../models/best_parameters"):
    """
    Main function to train and save a RandomForestClassifier model.

    Args:
        input_dir (str): Path to the directory containing preprocessed training data.
        input_dir_grid (str): Path to the best Parameters from Grid
        output_dir_save (str): Path to save the trained model.

    """
    # Set up logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Load the training data
    logging.info("Loading training data...")
    X_train = pd.read_csv(os.path.join(input_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).values.ravel()

    # Load the best parameters for GridSearchCV
    best_params_file = os.path.join(input_dir_grid, "best_params.pkl")
    if os.path.exists(best_params_file):
        logging.info(f"Loading best parameters from {best_params_file}...")
        best_params = joblib.load(best_params_file)
    else:
        logging.warning("Best parameters file not found. Proceeding with default parameters.")
        best_params = None

    # Train the model using the best parameters
    model = train_model(X_train, y_train, best_params=best_params)

<<<<<<< HEAD
    # Save the trained model
    save_trained_model(model, os.path.join(output_dir_save, "trained_model.pkl"))

=======
<<<<<<< HEAD
    # Save the model in BentoML & DVC
    save_trained_model(model, output_dir)
=======
    # Save the trained model
    save_trained_model(model, os.path.join(output_dir_save, "trained_model.pkl"))
>>>>>>> origin/main
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495


if __name__ == "__main__":
    main()
