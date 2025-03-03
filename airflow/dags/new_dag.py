from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
import subprocess
import json
import os
# Function to run commands
def run_command(command):
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        raise  # Reraise the exception to mark the task as failed
# Define the DAG
dag = DAG(
    'dvc_pipeline_V2',
    description='DVC pipeline for weather forecasting',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
)
# Stage 1: Data Ingestion
ingestion = PythonOperator(
    task_id='ingestion',
    python_callable=run_command,
    op_args=['python /opt/airflow/src/data/data_ingestion.py'],  # Updated path
    dag=dag,
)
def check_reference_file():
    """
    Check if reference.csv exists
    """
    reference_path = "data/raw_data/reference.csv"
    if os.path.exists(reference_path):
        return "data_drift_monitoring"
    else:
        return "split_data"
check_reference_task = BranchPythonOperator(
    task_id="Check_reference_file",
    python_callable = check_reference_file,
    dag=dag
)
# Stage 2: Data Drift Monitoring
data_drift_task = BashOperator(
    task_id="data_drift_monitoring",
    bash_command="python /opt/airflow/src/data/drift_monitoring.py",
    dag=dag
)
def check_data_drift():
    """Check drift score and determine next step."""
    drift_score_path = "/opt/airflow/metrics/data_drift.json"
    try:
        with open(drift_score_path, "r") as f:
            drift_results = json.load(f)
        drift_score = drift_results["metrics"][0]["result"]["drift_share"]
        if drift_score > 0.5:
            return "split_data"
        else:
            return "performance_drift_monitoring"
    except Exception as e:
        print(f"Error checking drift: {e}")
        return "performance_drift_monitoring"
check_drift_task = BranchPythonOperator(
    task_id="check_drift",
    python_callable=check_data_drift,
    dag=dag,
)
# Step 3: Performance Drift Monitoring
performance_drift_task = BashOperator(
    task_id="performance_drift_monitoring",
    bash_command="python /opt/airflow/src/data/performance_drift_monitoring.py",
    dag=dag
)
def check_model_performance():
    """Check performance metrics and decide whether to retrain."""
    model_drift_path = "/opt/airflow//metrics/performance_drift/model_drift.json"
    try:
        with open(model_drift_path, "r") as f:
            model_drift = json.load(f)
        if all(value == 0.0 for value in model_drift.values()):
            return "split_data"
        else:
            return "stop_pipeline"
    except Exception as e:
        print(f"Error checking model drift: {e}")
        return "stop_pipeline"
check_performance_task = BranchPythonOperator(
    task_id="check_model_performance",
    python_callable=check_model_performance,
    dag=dag,
)
stop_pipeline_task = BashOperator(
    task_id="stop_pipeline",
    bash_command="echo 'Model training not required. Stopping pipeline.'",
    dag=dag,
)
# Stage 4: Splitting Data
splitting = PythonOperator(
    task_id='splitting',
    python_callable=run_command,
    op_args=['python /opt/airflow/src/data/split_data.py'],  # Updated path
    dag=dag,
)
# Stage 5: Normalize Data
normalize_data = PythonOperator(
    task_id='normalize_data',
    python_callable=run_command,
    op_args=['python /opt/airflow/src/data/normalize_data.py'],  # Updated path
    dag=dag,
)
# Stage 6: Grid Search
grid_search = PythonOperator(
    task_id='grid_search',
    python_callable=run_command,
    op_args=['python /opt/airflow/src/models/grid_search.py'],  # Updated path
    dag=dag,
)
# Stage 7: Train Model
train_model = PythonOperator(
    task_id='train_model',
    python_callable=run_command,
    op_args=['python /opt/airflow/src/models/train_model.py'],  # Updated path
    dag=dag,
)
# Stage 8: Evaluate Model
evaluate_model = PythonOperator(
    task_id='evaluate_model',
    python_callable=run_command,
    op_args=['python /opt/airflow/src/models/evaluate_model.py'],  # Updated path
    dag=dag,
)
# Set dependencies between tasks (ordering)
ingestion >> check_reference_task
check_reference_task >> [data_drift_task, splitting]
# If reference.csv exists, check drift score
data_drift_task >> check_drift_task
check_drift_task >> [performance_drift_task, splitting]
# If drift score <= 0.5 -> Check model performance drift
performance_drift_task >> check_performance_task
check_performance_task >> [stop_pipeline_task, splitting]
# If drift score > 0.5 OR model performance dropped → Run full training pipeline
splitting >> normalize_data >> grid_search >> train_model >> evaluate_model