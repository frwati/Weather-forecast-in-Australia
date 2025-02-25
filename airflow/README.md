Setup Instructions

1-Start Airflow with Docker Compose
docker-compose up -d
⏳ This process may take around 5 minutes to complete.

2-Check Running Containers
docker ps

3-Access the Airflow Webserver Container
docker exec -it <CONTAINER_ID> /bin/bash
Replace <CONTAINER_ID> with the actual ID of the Airflow Webserver container from the previous step.

4-Remove the multipart package
pip uninstall python-multipart
Access Airflow & Trigger the DAG

5-Open your browser and go to: http://localhost:8081
Log in and navigate to the DAGs tab.
Find dvc_pipeline, then trigger it.