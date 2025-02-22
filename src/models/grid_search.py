import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import joblib
import logging
import os


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
    logging.info("Applying SMOTE to handle class imbalance...")
    X_train_smote, y_train_smote = SMOTE().fit_resample(X_train, y_train)
    logging.info(f"SMOTE applied: {X_train_smote.shape[0]} samples after resampling.")

    # Initialize model
    model = RandomForestClassifier(random_state=42)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 2],
    }

    # Log grid search initialization
    logging.info("Initializing GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        verbose=2,
        n_jobs=-1,
        scoring='accuracy',
    )

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


def main(input_dir="../../data/normalized_data/", output_dir="models/best_parameters"):
    """
    Main function to perform hyperparameter tuning and save the best parameters.
    
    Parameters:
        input_dir (str): Path to the directory containing normalized data.
        output_dir (str): Path to save the model and parameters.
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Load training data
    logging.info("Loading training data...")
    X_train = pd.read_csv(os.path.join(input_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).values.ravel()

    # Log data shape
    logging.info(f"Training data loaded: X_train shape {X_train.shape}, y_train shape {y_train.shape}.")

    # Perform GridSearchCV
    best_parameters = grid_search(X_train, y_train)

    # Save the best parameters
    save_best_params(best_parameters, os.path.join(output_dir, "best_params.pkl"))


if __name__ == "__main__":
    main()
