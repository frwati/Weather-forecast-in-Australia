# Project Name: Weather Forecast in Australia

## Description
This project focuses on building a weather forecasting system for Australia using machine learning models. The pipeline includes data preparation,drift monitoring, model training, hyperparameter tuning, and evaluation.

## Directory Structure
```
your_project/
│── data/
│   └── raw_data/                  # Directory containing raw data files retrieved during data ingestion
│   │    ├── current.csv           # New data retrieved by the data ingestion process (current/latest data)
│   │    └── reference.csv         # Previous data retrieved by the data ingestion process (historical data)
│   └── split_data/                # Directory storing split datasets for model training and testing
│   │    ├── X_train.csv           # Features for the training set (input variables)
│   │    ├── X_test.csv            # Features for the test set (input variables)
│   │    ├── y_train.csv           # Labels for the training set (target variables)
│   │    └── y_test.csv            # Labels for the test set (target variables)
│   └── normalized_data/           # Directory for storing preprocessed/normalized data
│        ├── normalized_X_train.csv # Normalized features for the training set
│        └── normalized_X_test.csv  # Normalized features for the test set
|
│── metrics/                       # Step 2 and 7: Drift Monitoring and Model Evaluation
│   ├── data_drift/                # Directory to store data drift reports
│   │    └── data_drift_report.json # Data drift report (stored as a JSON file)
│   ├── performance_drift/         # Directory to store model performance drift reports
│   │    └── performance_drift_report.json # Model performance drift report (stored as a JSON file)
|   ├── prediction.csv            # Stores prediction values in csv format
|   ├── scores.json               # Stores training model score in JSON file
| 
│── mlruns/                       # MLflow tracking experiments
| 
│── mlruns/                       # Step 7: Model Evaluation
│   ├── <unique_id>/              # Unique tracking server or experiment group ID
│   └── <experiment_id>/          # Experiment ID under which runs are grouped
│       ├── metrics/              # Metrics directory for all runs within this experiment
│       │   ├── accuracy          # Stores accuracy metric for this run
│       │   ├── f1_score          # Stores F1 score metric for this run
│       │   ├── mean_squared_error # Stores MSE (Mean Squared Error) metric for this run
│       │   └── r2_score          # Stores R² (R-squared) score metric for this run
│       ├── tags/                 # Contains tags associated with the experiment
│       ├── params/               # Parameters for the experiment (like model settings)
│       ├── artifacts/            # Artifacts like trained models, images, or files
│       └── meta.yaml             # Metadata about the experiment (name, ID, etc.)
|
│── models/                       # Step 5 and 6: Grid Search and Train model
│   ├── best_param.pkl            # Best hyperparameters for training the model (pickle format)
│   └── trained_model.pkl         # Final trained model (pickle format)
|
│── preprocessing/                # Step 3: Data Preprocessing
│   ├── label_encoders/           # Stores label encoders for converting categorical variables to numeric values
│   │    └── encoder_name.pkl     # Label encoder for a specific categorical feature (e.g., "weather_condition")
│   └── scalers/                  # Stores scalers for normalizing or standardizing numerical features
│        └── scaler_name.pkl      # Scaler used to transform numerical features (MinMaxScaler)
|
│── reports/                      # Step 2: Drift Monitoring
│   ├── data_drift/               # Directory to store data drift reports (html format)
│   └── performance_drift/        # Directory to store performance drift reports (html format)
|
│── src/                                       # Main source code directory for the weather prediction pipeline
│   ├── data/                                  # Data-related processing scripts
│   │    ├── data_ingestion.py                 # Ingests new data (e.g., retrieves latest weather data from external sources)
│   │    ├── drift_monitoring.py               # Monitors data drift (detects changes in input data distribution over time)
│   │    ├── performance_drift_monitoring.py   # Monitors model performance drift (detects changes in model performance over time)
│   │    ├── split.py                          # Splits the dataset into training and testing sets (avoiding data leakage)
│   │    └── normalize.py                      # Normalizes data by scaling features to improve model training
│   ├── models/                                # Machine learning model-related scripts
│   │    ├── grid_search.py                    # Performs hyperparameter tuning using Grid Search to optimize model parameters
│   │    ├── train_model.py                    # Trains the model using the training dataset (e.g., RandomForest, XGBoost, etc.)
│   │    └── evaluate.py                       # Evaluates the model performance using metrics (accuracy, F1 score, mse, r2_score)
│   └── service.py                             # API service for weather prediction using Bentoml, with JWT authentication security
|
│── .github/                    # Directory for GitHub Actions workflows
│   └── workflow/               # Contains GitHub Actions workflow files for CI/CD 
│        └── python-ci.yml      # GitHub Actions workflow for Python Continuous Integration (CI)
|
│── airflow/                    # Main Airflow directory
│   ├── dags/                   # DAGs directory (contains workflows)
│   |    └── new_dag.py         # DAG file
│   ├── logs/                   # Logs directory (stores execution logs)
│   ├── README.md               # Documentation for the airflow project
│   └── dcoker-compose.yaml     # Docker Compose file for running Airflow
|
│── Dockerfile.template         # Template for Dockerfile (used for building images)
│── bentofile.yaml              # BentoML configuration file
│── dvc.yaml                    # Data Version Control (DVC) pipeline
│── dvc.lock                    # DVC lock file (records exact versions of data)
│── .gitignore                  # Files to be ignored by Git
│── .dvcignore                  # Files to be ignored by DVC   
│── README.md                   # Documentation for the project
│── requirements.txt            # Python dependencies
```

## Steps:
### 1. Data Ingestion (src/data/data_ingestion.py)
This script handles the ingestion of new weather data, whether by scraping, pulling data from APIs, or reading files. It retrieves the latest weather data and prepares it for further processing.
### Example usage: 
```python src/data/data_ingestion.py```

### 2. Drift Monitoring 
### I. Data Drift Monitoring(src/data/drift_monitoring.py)
This file is responsible for detecting data drift. Data drift refers to the change in the statistical properties of the input data over time, which can affect the model's accuracy. The script monitors these changes to ensure the model remains valid.
### Example usage: 
```
python src/data/drift_monitoring.py
```

### II. Model Drift Monitoring (src/data/performance_drift_monitoring.py)
This script tracks performance drift, which is the degradation of the model’s performance over time. It uses performance metrics (e.g., accuracy, F1 score) to monitor if the model's predictions are becoming less reliable, signaling a need for retraining.
### Example usage: 
```
python src/data/performance_drift_monitoring.py
```

### 3. Data Splitting (src/data/split.py)
This script is responsible for splitting the dataset into training and testing sets. It ensures that data leakage is prevented and the model is trained on one part of the data and tested on another.
### Example usage: 
```
python src/data/split.py
```


### 4. Data Normalization (src/data/normalize.py)
This script normalizes the dataset. Normalization scales the features so that they all lie within the same range, improving model convergence during training.
### Example usage: 
```
python src/data/normalize.py
```


### 5. Grid Search for Hyperparameter Tuning (src/models/grid_search.py)
This script performs hyperparameter tuning using Grid Search to find the optimal set of parameters for the model. It helps in improving the model's performance by systematically testing combinations of different hyperparameters.
### Example usage: 
```
python src/models/grid_search.py
```


### 6. Model Training (src/models/train_model.py)
This script trains the model using the training data. It involve using machine learning algorithms RandomForestClassifier.
### Example usage:
```
python src/models/train_model.py
```


### 7. Model Evaluation (src/models/evaluate.py)
After the model is trained, this script evaluates its performance on the test data, using metrics such as accuracy, precision, recall, or mean squared error. Also, rename current.csv file to reference.csv file for future update in dataset.
### Example usage: 
```
python src/models/evaluate.py
```

### 8. API service (src/service.py)
This file contains the Bentoml API service, which wraps the trained weather prediction model into an API for real-time predictions. It integrates JWT authentication to secure the service, ensuring only authorized users can access the weather prediction functionality.
### Example usage: 
```
src/service.py
```

# Airflow Docker Setup  

## 🚀 Overview  
This setup containerizes **Apache Airflow** using **Docker Compose** (version 3.8).  
- Uses **CeleryExecutor** to support distributed task execution.  
- Integrated with **PostgreSQL (Database)** and **Redis (Message Broker)**.  

---

## 📦 Docker Setup  

### Prerequisites  
Ensure you have the following installed before proceeding:  
- **Docker** (latest version)  
- **Docker Compose**  
- **Python 3.8+**  
- **pip & virtualenv**  

---

  
### Setup Instructions (airflow/docker-compose.yaml)
1. ***Start Airflow with Docker Compose***
   ```sh
   docker-compose up -d
   ```
   ⏳ This process may take around 5 minutes to complete.
   
2. ***Check Running Containers***
   ```sh
   docker ps
   ```
   
3. ***Access the Airflow Webserver Container***
   ```sh
   docker exec -it <CONTAINER_ID> /bin/bash
   ```
   Replace <CONTAINER_ID> with the actual ID of the Airflow Webserver container from the previous step.
   
4. ***Remove the multipart package***
  ```sh
  pip uninstall python-multipart
  ```
  
5. ***Access Airflow & Trigger the DAG***
- Open your browser and go to: http://localhost:8081
- Log in and navigate to the DAGs tab.
- Find dvc_pipeline, then trigger it.
