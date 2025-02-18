import pandas as pd
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os
import logging

#Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    #Check if file exists
    if not os.path.exists(reference_path):
        logging.error(f"Reference dataset {reference_path} is missing")
        return
    
    if not os.path.exists(current_path):
        logging.error(f"Current dataset {current_path} is missing")
        return
    
    try:
        #Load datasets
        logging.info("Loading datasets for drift detection")
        reference_data = pd.read_csv(reference_path)
        current_data = pd.read_csv(current_path)
        
        #Ensure both datasets have the same structure
        if not all(reference_data.columns == current_data.columns):
            logging.error(f"Reference and current datasets have different columns")
            return
        
    except Exception as e:
        logging.error(f"Error while loading datasets: {e}")
        return
    
    #Run Evidently drift Analysis
    try:
        logging.info("Running Evidently drift detection")
        data_drift_report = Report(metrics=[DataDriftPreset()])
        data_drift_report.run(reference_data=reference_data, current_data=current_data)
    
        #Save JSON (for pipeline processing)
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        drift_results = data_drift_report.as_dict()
        with open(output_json, "w") as f:
            json.dump(drift_results, f, indent=4)
        logging.info(f"Drift result saved to: {output_json}")
    
        #Save HTML (for visualization)
        os.makedirs(os.path.dirname(output_html), exist_ok=True)
        data_drift_report.save_html(output_html)
        logging.info(f"Drift report saved to: {output_html}")
    
        # Return drift score for monitoring
        drift_score = drift_results["metrics"][0]["result"]["drift_share"]
        return drift_score
    
    except Exception as e:
        logging.error(f"Error during drift analysis: {e}")
        return

if __name__ == "__main__":
    reference_path = "data/raw_data/reference.csv"
    current_path = "data/raw_data/current.csv"
    
    # Detect drift   
    drift_score = detect_data_drift(reference_path, current_path)
    
    if drift_score is not None:
        logging.info(f"Drift score: {drift_score}")
        
        #Trigger retraining if drift > 0.5
        drift_threshold = 0.5
        if drift_score > drift_threshold:
            logging.info("Significant drift detected! Triggering model retraining.")
        else:
            logging.info("No Significant drift. Model remains unchanged")
    else:
        logging.error(f"Drift detection failed")
