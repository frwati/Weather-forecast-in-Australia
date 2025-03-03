from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import subprocess

# Function to run commands
def run_command(command):
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        raise  # Reraise the exception to mark the task as failed

# Define the DAG
dag = DAG(
    'dvc_pipeline_V1',
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

# Stage 2: Splitting Data
splitting = PythonOperator(
    task_id='splitting',
    python_callable=run_command,
    op_args=['python /opt/airflow/src/data/split_data.py'],  # Updated path
    dag=dag,
)

# Stage 3: Normalize Data
normalize_data = PythonOperator(
    task_id='normalize_data',
    python_callable=run_command,
    op_args=['python /opt/airflow/src/data/normalize_data.py'],  # Updated path
    dag=dag,
)

# Stage 4: Grid Search
grid_search = PythonOperator(
    task_id='grid_search',
    python_callable=run_command,
    op_args=['python /opt/airflow/src/models/grid_search.py'],  # Updated path
    dag=dag,
)

# Stage 5: Train Model
train_model = PythonOperator(
    task_id='train_model',
    python_callable=run_command,
    op_args=['python /opt/airflow/src/models/train_model.py'],  # Updated path
    dag=dag,
)

# Stage 6: Evaluate Model
evaluate_model = PythonOperator(
    task_id='evaluate_model',
    python_callable=run_command,
    op_args=['python /opt/airflow/src/models/evaluate_model.py'],  # Updated path
    dag=dag,
)

# Stage 7: Data Drift Detection (new stage)
data_drift_detection = PythonOperator(
    task_id='data_drift_detection',
    python_callable=run_command,
    op_args=['python /opt/airflow/src/data/drift_monitoring.py'],  # Updated path
    dag=dag,
)

# Set dependencies between tasks (ordering)
ingestion >> splitting >> normalize_data >> grid_search >> train_model >> evaluate_model >> data_drift_detection
