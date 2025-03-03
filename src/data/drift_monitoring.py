import pandas as pd
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os
import logging
from pathlib import Path
from evidently.ui.workspace import Workspace

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set project root based on environment
if os.path.exists("/.dockerenv"):
    PROJECT_ROOT = Path("/opt/airflow")  # Docker
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    #PROJECT_ROOT = Path.cwd()

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
    #reference_path = os.path.join(PROJECT_ROOT, reference_path)
    #current_path = os.path.join(PROJECT_ROOT, current_path)
    output_json = os.path.join(PROJECT_ROOT, output_json)
    output_html = os.path.join(PROJECT_ROOT, output_html)

    reference_path = Path(reference_path)
    current_path = Path(current_path)
    output_json = Path(output_json)
    output_html = Path(output_html)



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
        return data_drift_report, drift_score
    
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

if __name__ == "__main__":
    WORKSPACE_NAME = "Weather-Classification"
    PROJECT_NAME = "Weather Forecast in Australia"
    PROJECT_DESCRIPTION = "This project focuses on building a weather forecasting system for Australia using machine learning models. The pipeline includes data preparation,drift monitoring, model training, hyperparameter tuning, and evaluation."

    reference_path = os.path.join(PROJECT_ROOT, "data/raw_data/reference.csv")
    current_path = os.path.join(PROJECT_ROOT, "data/raw_data/current.csv")
    # Detect drift   
    data_drift_report,drift_score = detect_data_drift(reference_path, current_path)

    add_report_to_workspace(WORKSPACE_NAME, PROJECT_NAME, PROJECT_DESCRIPTION, data_drift_report)


    # Decide whether retraining is needed
    retrain_needed = should_retrain(drift_score)
    
    if retrain_needed:
        logging.info("Trigger retraining process.")
    else:
        logging.info("No retraining needed.")