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
