import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import joblib
import logging
import os

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

    logging.info("Applying SMOTE to handle class imbalance...")
    X_train_smote, y_train_smote = SMOTE().fit_resample(X_train, y_train)
    logging.info(f"SMOTE applied: {X_train_smote.shape[0]} samples after resampling.")

    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 2],
    }

    logging.info("Initializing GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        verbose=2,
        n_jobs=-1,
        scoring='accuracy',
    )

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
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    input_dir = os.path.join(PROJECT_ROOT, "data/normalized_data")
    output_dir = os.path.join(PROJECT_ROOT, "models/best_parameters")
    param_file = os.path.join(output_dir, "best_params.pkl")

    logging.info("Loading training data...")
    X_train = pd.read_csv(os.path.join(input_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).values.ravel()

    logging.info(f"Training data loaded: X_train shape {X_train.shape}, y_train shape {y_train.shape}.")

    best_parameters = grid_search(X_train, y_train, param_file)
    logging.info(f"Best parameters used: {best_parameters}")

if __name__ == "__main__":
    main()
