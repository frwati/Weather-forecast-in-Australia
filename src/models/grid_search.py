import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import joblib
import logging
import os

<<<<<<< HEAD
# Determine project root dynamically
if os.path.exists("/.dockerenv"):
    PROJECT_ROOT = "/opt/airflow"  # Docker environment path
else:
    PROJECT_ROOT = os.getcwd()  # Local environment path

logging.info(f"Running inside: {PROJECT_ROOT}")

def grid_search(X_train, y_train, param_file):
    """
    Perform GridSearchCV to find the best hyperparameters for the RandomForest model.
    If best parameters already exist, load them instead of rerunning GridSearchCV.
    """
    if os.path.exists(param_file):
        logging.info(f"Loading existing best parameters from {param_file}...")
        return joblib.load(param_file)

=======

def grid_search(X_train, y_train):
    """
    Perform GridSearchCV to find the best hyperparameters for the RandomForest model.
    
    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    
    Returns:
        dict: Best hyperparameters found by GridSearchCV.
    """
    # Log SMOTE application
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    logging.info("Applying SMOTE to handle class imbalance...")
    X_train_smote, y_train_smote = SMOTE().fit_resample(X_train, y_train)
    logging.info(f"SMOTE applied: {X_train_smote.shape[0]} samples after resampling.")

<<<<<<< HEAD
    model = RandomForestClassifier(random_state=42)
=======
    # Initialize model
    model = RandomForestClassifier(random_state=42)

    # Define the parameter grid
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 2],
    }

<<<<<<< HEAD
=======
    # Log grid search initialization
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    logging.info("Initializing GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        verbose=2,
        n_jobs=-1,
        scoring='accuracy',
    )

<<<<<<< HEAD
    logging.info("Performing GridSearchCV...")
    grid_search.fit(X_train_smote, y_train_smote)

    best_params = grid_search.best_params_
    logging.info(f"GridSearchCV completed. Best parameters: {best_params}")

    # Save best parameters
    os.makedirs(os.path.dirname(param_file), exist_ok=True)
    joblib.dump(best_params, param_file)
    logging.info(f"Best parameters saved to {param_file}")

    return best_params

def main():
    """
    Main function to perform hyperparameter tuning and save the best parameters.
=======
    # Perform grid search
    logging.info("Performing GridSearchCV...")
    grid_search.fit(X_train_smote, y_train_smote)

    # Log results
    best_params = grid_search.best_params_
    logging.info(f"GridSearchCV completed. Best parameters: {best_params}")
    return best_params


def save_best_params(best_params, filename="models/best_params.pkl"):
    """
    Save the best hyperparameters to a .pkl file.
    
    Parameters:
        best_params (dict): Best hyperparameters found by GridSearchCV.
        filename (str): Path to save the best parameters.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save parameters
    logging.info(f"Saving best parameters to {filename}...")
    joblib.dump(best_params, filename)
    logging.info("Best parameters saved successfully.")


def main(input_dir="../../data/normalized_data/", output_dir="../../models/best_parameters"):
    """
    Main function to perform hyperparameter tuning and save the best parameters.
    
    Parameters:
        input_dir (str): Path to the directory containing normalized data.
        output_dir (str): Path to save the model and parameters.
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

<<<<<<< HEAD
    input_dir = os.path.join(PROJECT_ROOT, "data/normalized_data")
    output_dir = os.path.join(PROJECT_ROOT, "models/best_parameters")
    param_file = os.path.join(output_dir, "best_params.pkl")

=======
    # Load training data
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    logging.info("Loading training data...")
    X_train = pd.read_csv(os.path.join(input_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).values.ravel()

<<<<<<< HEAD
    logging.info(f"Training data loaded: X_train shape {X_train.shape}, y_train shape {y_train.shape}.")

    best_parameters = grid_search(X_train, y_train, param_file)
    logging.info(f"Best parameters used: {best_parameters}")
=======
    # Log data shape
    logging.info(f"Training data loaded: X_train shape {X_train.shape}, y_train shape {y_train.shape}.")

    # Perform GridSearchCV
    best_parameters = grid_search(X_train, y_train)

    # Save the best parameters
    save_best_params(best_parameters, os.path.join(output_dir, "best_params.pkl"))

>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495

if __name__ == "__main__":
    main()
