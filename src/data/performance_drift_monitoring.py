import pandas as pd
import logging
import joblib
import os
import numpy as np
import json
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.ui.workspace import Workspace

# Detect if running inside Airflow's Docker container
if os.path.exists("/.dockerenv"):
    project_root = "/opt/airflow"  # Path inside Docker
    logging.info(f"Running inside Docker at: {project_root}...")
else:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # Local path
    logging.info(f"Running locally at: {project_root}...")

# Define constants
SCALER_DIR = os.path.join(project_root, 'preprocessing/scalers')
LE_DIR = os.path.join(project_root, 'preprocessing/label_encoders')
current_data_path = os.path.join(project_root, "data/raw_data/current.csv")
reference_data_path = os.path.join(project_root, "data/raw_data/reference.csv")
model_drift_output_path = os.path.join(project_root, "metrics/performance_drift/model_drift.json")
MODEL_PATH = os.path.join(project_root, 'models/saved_models/trained_model.pkl')

def load_preprocessors(scaler_dir=SCALER_DIR, le_dir=LE_DIR):
    """
    Load saved scalers and label encoders from disk.
    """
    scalers = {}
    label_encoders = {}
    
    # Load scalers
    if os.path.exists(scaler_dir):
        for scaler_file in os.listdir(scaler_dir):
            if scaler_file.endswith("_scaler.pkl"):
                column = scaler_file.split("_")[0]
                scalers[column] = joblib.load(os.path.join(scaler_dir, scaler_file))
    
    # Load label encoders
    if os.path.exists(le_dir):
        for le_file in os.listdir(le_dir):
            if le_file.endswith("_le.pkl"):
                column = le_file.split("_")[0]
                label_encoders[column] = joblib.load(os.path.join(le_dir, le_file))
    
    logging.info("Preprocessors loaded successfully.")
    return scalers, label_encoders

def fill_missing_temps(df, date_column='Date', column=None):
    """
    Fills missing values in the specified column by taking the median value
    of the same day (ignoring the year) across all years.
    """
    if column is None:
        raise ValueError("You must specify the column to fill missing values for.")
    
    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Extract day and month to group by
    df['day_month'] = df[date_column].dt.strftime('%m-%d')

    # Compute the median temp for each day-month combination
    medians = df.groupby('day_month')[column].median()

    # Fill NaN values in the temp column with the median value
    df.loc[:, column] = df.apply(
        lambda row: medians[row['day_month']] if pd.isna(row[column]) else row[column],
        axis=1
    )

    # Drop the helper day_month column
    df.drop(columns=['day_month'], inplace=True)

    return df

def preprocess_data(input_file, scalers, label_encoders):
    """
    Preprocesses the data using the provided scalers and label encoders.
    """
    # Load raw data
    logging.info("Loading raw data...")
    df = pd.read_csv(input_file)

    # Drop the identified columns
    df = df.drop(columns=['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'])

    # Remove rows with no RainTomorrow data available and convert the str into int values
    df = df.dropna(subset=['RainTomorrow'])
    df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

    # Fill NaN values in RainToday to 'No' and replace the str into int values
    df['RainToday'] = df['RainToday'].fillna('No')
    df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})

    # Ensure Date is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # 'coerce' will turn invalid dates into NaT

    # Check if there are any invalid date values
    if df['Date'].isnull().any():
        logging.warning("Some rows have invalid or missing dates. These will be dropped.")
        df = df.dropna(subset=['Date'])

    # Create time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    # Using "IsWeekend" to indicate if the day is a weekend (Saturday/Sunday)
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    # Filling missing values in specific columns
    columns_to_fillup = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
                         'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
    for column in columns_to_fillup:
        if column in df.columns:
            df = fill_missing_temps(df, 'Date', column=column)
        else:
            logging.info(f"Column '{column}' not found in DataFrame. Skipping.")

    # Removing rows containing missing values 
    df = df.dropna()

    # Apply scalers to numerical columns
    scale_columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 
                     'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 
                     'Temp9am', 'Temp3pm']

    for column in scale_columns:
        if column in scalers:
            df[column] = scalers[column].transform(df[[column]])
        else:
            logging.warning(f"Scaler for column '{column}' not found. Skipping normalization for this column.")

    # Apply label encoders to categorical columns
    encode_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']

    for column in encode_columns:
        if column in label_encoders:
            df[column] = label_encoders[column].transform(df[column])
        else:
            logging.warning(f"Label encoder for column '{column}' not found. Skipping encoding for this column.")
            
    # Return features and target from downstream use
    X = df.drop(['RainTomorrow', 'Date'], axis=1)
    y = df['RainTomorrow']
    return X, y



def save_model_drift_metrics(metrics, output_path):
    """
    Save the model drift metrics to a JSON file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the metrics to the file
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def add_report_to_workspace(workspace_name, project_name, project_description, report):
    """
    Adds a report to an existing or new project in a workspace.
    """
    # Create or get workspace
    workspace = Workspace.create(workspace_name)

    # Check if project already exists
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    # Create a new project if it doesn't exist
    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description

    # Add report to the project
    workspace.add_report(project.id, report)
    logging.info(f"New report added to project {project_name}")


def main():
   
    # Load the saved scalers and label encoders
    scalers, label_encoders = load_preprocessors()

    # Preprocess the current data
    X_current, y_current = preprocess_data(current_data_path, scalers, label_encoders)
    X_reference, y_reference = preprocess_data(reference_data_path, scalers, label_encoders)
      
    # Load the deployed model and make predictions
    model = joblib.load(MODEL_PATH)  # Load the model
    
    # Make predictions and add it to the X_current and X_reference
    X_current['prediction'] = model.predict(X_current)
    X_reference['prediction'] = model.predict(X_reference)

    # Add target variable to X_current and X_reference
    X_current['target'] = y_current  
    X_reference['target'] = y_reference

    # Define column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = "target"
    column_mapping.prediction = "prediction"
    
    # Generate the classification report
    classification_report = Report(metrics=[ClassificationPreset()])
    classification_report.run(reference_data=X_reference, current_data=X_current, column_mapping=column_mapping)

    # Extract the classification metrics from the report
    metrics = classification_report.as_dict()

    # Iterate through the list of metrics to find 'ClassificationQualityMetric'
    classification_quality_metric = None
    for metric in metrics['metrics']:
        if metric['metric'] == 'ClassificationQualityMetric':
            classification_quality_metric = metric['result']
            break

    if classification_quality_metric is not None:
        current_metrics = classification_quality_metric['current']
        reference_metrics = classification_quality_metric['reference']

        # Extract the classification metrics
        accuracy = current_metrics.get('accuracy', None)
        f1_score = current_metrics.get('f1', None)
        precision = current_metrics.get('precision', None)
        recall = current_metrics.get('recall', None)

        # Print the extracted metrics
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1_score}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

        # Define thresholds for retraining
        THRESHOLDS = {
            "accuracy": 0.02,
            "f1_score": 0.02,
            "precision": 0.02,
            "recall": 0.02,
        }

        # Compute performance drift
        performance_drop = {
            "accuracy": abs(accuracy - reference_metrics['accuracy']),
            "f1_score": abs(f1_score - reference_metrics['f1']),
            "precision": abs(precision - reference_metrics['precision']),
            "recall": abs(recall - reference_metrics['recall']),
        }
        # Prepare the drift metrics for exporting
        model_drift_metrics = {
            "current_metrics": current_metrics,
            "reference_metrics": reference_metrics,
            "performance_drop": performance_drop,
            "thresholds": THRESHOLDS,
        }

        # Print comparison logs
        print("\n **Performance Drift Comparison:**")
        for metric, drop in performance_drop.items():
            previous_value = reference_metrics.get(metric, None)
            current_value = current_metrics.get(metric, None)
    
            if previous_value is not None and current_value is not None:
                print(f" {metric}: Previous = {previous_value:.4f}, Current = {current_value:.4f}, Drop = {drop:.4f}")
            else:
                print(f" {metric}: Missing in either reference or current metrics.")
        
        # Check if retraining is needed
        retrain_needed = any(drop > THRESHOLDS[metric] for metric, drop in performance_drop.items())
         # Export the model drift metrics to a JSON file

        save_model_drift_metrics(model_drift_metrics, model_drift_output_path)

        if retrain_needed:
            print("\n**Performance Drift Detected! Retraining Required.**")
        else:
            print("\n**No Significant Performance Drift. Model is stable.**")

        # Add the classification report to the workspace
        WORKSPACE_NAME = "Weather-Classification"
        PROJECT_NAME = "Weather Forecast in Australia"
        PROJECT_DESCRIPTION = "This project focuses on building a weather forecasting system for Australia using machine learning models."

        workspace = Workspace.create(WORKSPACE_NAME)
        add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, classification_report)
    


        return retrain_needed
    else:
        print("Error: 'ClassificationQualityMetric' not found in metrics")

if __name__ == "__main__":
    retrain_needed = main()
    exit(1 if retrain_needed else 0)