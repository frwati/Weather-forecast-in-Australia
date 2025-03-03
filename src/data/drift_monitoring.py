import pandas as pd
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os
import logging
from pathlib import Path
from prometheus_client import start_http_server, Gauge
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set project root based on environment
if os.path.exists("/.dockerenv"):
    PROJECT_ROOT = Path("/opt/airflow")  # Docker
else:
    PROJECT_ROOT = Path.cwd()

# Define Prometheus metric objects
drift_score_gauge = Gauge('data_drift_score', 'The current data drift score', ['model'])

def start_prometheus_server(port=8000):
    """
    Starts a Prometheus metrics server to expose drift score.
    """
    start_http_server(port)
    logging.info(f"Prometheus metrics exposed on port {port}")

def expose_data_drift(drift_score):
    """
    Expose the data drift score to Prometheus.
    """
    if drift_score is not None:
        drift_score_gauge.labels(model='Weather_rf_model').set(drift_score)
        logging.info(f"Exposed data drift score: {drift_score}")

def detect_data_drift(reference_path, current_path, output_json="metrics/data_drift.json", output_html="reports/data_drift.html"):
    """
    Detects data drift between reference and current datasets.
    Saves results in JSON (for pipeline) and HTML (for visualization)
    
    Args:
        reference_path (_type_): _description_
        current_path (_type_): _description_
        output_json (str, optional): _description_. Defaults to "metrics/data_drift.json".
        output_html (str, optional): _description_. Defaults to "reports/data_drift.html".
    """
    reference_path = PROJECT_ROOT / reference_path
    current_path = PROJECT_ROOT / current_path
    output_json = PROJECT_ROOT / output_json
    output_html = PROJECT_ROOT / output_html

    # Check if file exists
    if not reference_path.exists():
        logging.error(f"Reference dataset {reference_path} is missing")
        return
    
    if not current_path.exists():
        logging.error(f"Current dataset {current_path} is missing")
        return
    
    try:
        # Load datasets
        logging.info("Loading datasets for drift detection")
        reference_data = pd.read_csv(reference_path)
        current_data = pd.read_csv(current_path)
        
        # Ensure both datasets have the same structure
        if not all(reference_data.columns == current_data.columns):
            logging.error(f"Reference and current datasets have different columns")
            return
        
    except Exception as e:
        logging.error(f"Error while loading datasets: {e}")
        return
    
    # Run Evidently drift Analysis
    try:
        logging.info("Running Evidently drift detection")
        data_drift_report = Report(metrics=[DataDriftPreset()])
        data_drift_report.run(reference_data=reference_data, current_data=current_data)
    
        # Save JSON (for pipeline processing)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        drift_results = data_drift_report.as_dict()
        with open(output_json, "w") as f:
            json.dump(drift_results, f, indent=4)
        logging.info(f"Drift result saved to: {output_json}")
    
        # Save HTML (for visualization) **FIX: Convert Path to str**
        output_html.parent.mkdir(parents=True, exist_ok=True)
        data_drift_report.save_html(str(output_html))  # <-- FIXED
        logging.info(f"Drift report saved to: {output_html}")
    
        # Return drift score for monitoring
        drift_score = drift_results["metrics"][0]["result"]["drift_share"]
        return drift_score
    
    except Exception as e:
        logging.error(f"Error during drift analysis: {e}")
        return

def should_retrain(drift_score, drift_threshold=0.5):
    """
    Decide whether retraining is needed based on the drift score.
    Args:
        drift_score (float): The drift score calculated from the data drift report
        drift_threshold (float): The threshold beyond which retraining is triggered
    Returns:
        bool: True if retraining is needed, False otherwise
    """
    if drift_score is not None:
        logging.info(f"Drift score: {drift_score}")
        if drift_score > drift_threshold:
            logging.info("Significant drift detected! Triggering model retraining.")
            return True
        else:
            logging.info("No significant drift. Model remains unchanged.")
            return False
    else:
        logging.error("Drift score is None. Cannot determine retraining necessity.")
        return False
    
def main():
    # Start Prometheus server
    start_prometheus_server(port=8000)

    reference_path = "data/raw_data/reference.csv"
    current_path = "data/raw_data/current.csv"

    # Detect drift
    drift_score = detect_data_drift(reference_path, current_path)
    expose_data_drift(drift_score)

    # Decide whether retraining is needed
    retrain_needed = should_retrain(drift_score)
    
    if retrain_needed:
        # Trigger retraining in your pipeline here
        logging.info("Trigger retraining process.")
    else:
        logging.info("No retraining needed.")

        
if __name__ == "__main__":
    main()
