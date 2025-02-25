#!/bin/bash

# Upgrade pip
echo "🔄 Upgrading pip..."
python -m pip install --upgrade pip

# Ensure 'python-multipart' is installed
echo "🔧 Installing required packages..."
pip uninstall -y multipart && pip install python-multipart

# Initialize the database
#echo "📌 Initializing the Airflow database..."
#airflow db migrate

# Create the Airflow admin user if not already created
if ! airflow users list | grep -q "airflow"; then
    echo "👤 Creating admin user..."
    airflow users create \
        --username airflow \
        --password airflow \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@admin.com
fi

# Start the requested command
exec "$@"
