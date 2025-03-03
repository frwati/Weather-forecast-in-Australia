import pandas as pd
import logging
import joblib
import os
import numpy as np
import json
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
from prometheus_client import start_http_server, Gauge
import time

# Define Prometheus metric objects
accuracy_gauge = Gauge('model_accuracy', 'Model accuracy', ['model'])
f1_score_gauge = Gauge('model_f1_score', 'F1 score of the model', ['model'])
precision_gauge = Gauge('model_precision', 'Model precision', ['model'])
recall_gauge = Gauge('model_recall', 'Model recall', ['model'])


# Define constants
SCALER_DIR = 'preprocessing/scalers'
LE_DIR = 'preprocessing/label_encoders'

def start_prometheus_server(port=8000):
    """
    Starts a Prometheus metrics server to expose drift scores and performance drift.
    """
    start_http_server(port)
    logging.info(f"Prometheus metrics exposed on port {port}")

def expose_performance_drift(performance_drop, model_name="weather_rf_model"):
    """
    Expose the performance drift metrics (accuracy, f1_score, precision, recall) to prometheus.
    """
    accuracy_gauge.labels(model=model_name).set(performance_drop['accuracy'])
    f1_score_gauge.labels(model=model_name).set(performance_drop['f1_score'])
    precision_gauge.labels(model=model_name).set(performance_drop['precision'])
    recall_gauge.labels(model=model_name).set(performance_drop['recall'])

def load_preprocessors(scaler_dir=SCALER_DIR, le_dir=LE_DIR):
    """
    Load saved scalers and label encoders from disk.
    """
    scalers = {}
    label_encoders = {}

    # Load scalers
    for scaler_file in os.listdir(scaler_dir):
        if scaler_file.endswith("_scaler.pkl"):
            column = scaler_file.split("_")[0]
            scalers[column] = joblib.load(os.path.join(scaler_dir, scaler_file))
    
    # Load label encoders
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

    
def main():
    # Paths
    current_data_path = "data/raw_data/current.csv"
    reference_data_path = "data/raw_data/reference.csv"
    scaler_dir = 'preprocessing/scalers'
    le_dir = 'preprocessing/label_encoders'
    
    # Load the saved scalers and label encoders
    scalers, label_encoders = load_preprocessors(scaler_dir, le_dir)
    
    # Preprocess the current data
    X_current, y_current = preprocess_data(current_data_path, scalers, label_encoders)
    X_reference, y_reference = preprocess_data(reference_data_path, scalers, label_encoders)
      
     # Load the deployed model and make predictions
    model_path = 'models/trained_model.pkl'  # Path to the saved model
    model = joblib.load(model_path)  # Load the model 
    
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
    
    classification_report = Report(metrics=[ClassificationPreset()])
    classification_report.run(reference_data=X_reference, current_data=X_current, column_mapping=column_mapping)
      
    # Extract the classification metrics from the report
    metrics = classification_report.as_dict()

    # Calculate performance drift metrics
    classification_quality_metric = next(
	(metric['result'] for metric in metrics['metrics'] if metric['metrics'] == 'ClassificationQualityMetric'), None)

    if classification_quality_metric:
	current_metrics = classification_quality_metric['current']
        reference_metrics = classification_quality_metric['reference']

        # Now safely access the classification metrics of current dataset
        accuracy = current_metrics.get('accuracy', None)
        f1_score = current_metrics.get('f1', None)
        precision = current_metrics.get('precision', None)
        recall = current_metrics.get('recall', None)

        # Print the extracted metrics
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1_score}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

        # Define thresholds for retraining (set as needed)
        THRESHOLDS = {
            "accuracy": 0.02,  # Retrain if Accuracy drops more than 0.02
            "f1_score": 0.02,  # Retrain if F1 score drops more than 0.02
            "precision": 0.02,  # Retrain if Precision drops more than 0.02
            "recall": 0.02,  # Retrain if Recall drops more than 0.02
        }

        # Compute performance drift based on the metrics
        performance_drop = {
            "accuracy": abs(accuracy - reference_metrics['accuracy']),
            "f1_score": abs(f1_score - reference_metrics['f1']),
            "precision": abs(precision - reference_metrics['precision']),
            "recall": abs(recall - reference_metrics['recall']),
        }

        # Expose performance drift to prometheus
	expose_performance_drift(performance_drop)

	# Print comparison logs
        print("\n **Performance Drift Comparison:**")
        for metric, drop in performance_drop.items():
            # Use .get() method to avoid KeyError if metric is missing
            previous_value = reference_metrics.get(metric, None)
            current_value = current_metrics.get(metric, None)
    
        if previous_value is not None and current_value is not None:
            print(f" {metric}: Previous = {previous_value:.4f}, Current = {current_value:.4f}, Drop = {drop:.4f}")
        else:
            print(f" {metric}: Missing in either reference or current metrics.")
        

        # Check if retraining is needed
        retrain_needed = any(drop > THRESHOLDS[metric] for metric, drop in performance_drop.items())

        if retrain_needed:
            logging.info("Performance Drift Detected! Retraining Required.")
        else:
            logging.info("No Significant Performance Drift. Model is stable.")

        return retrain_needed
    else:
        logging.error("Classification Quality Metric not found in the report")
	return False

if __name__ == "__main__":
    retrain_needed = main()
    exit(1 if retrain_needed else 0)

