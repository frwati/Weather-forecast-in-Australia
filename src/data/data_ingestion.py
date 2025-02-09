import kagglehub
import shutil
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler('data_ingestion.log')  
    ]
)

def download_and_process_data(dataset_name, raw_data_folder):
    """
    Download and process the dataset:
    1. Download the dataset using kagglehub.
    2. Search for the first CSV file.
    3. Move and rename the CSV to `raw.csv` in the `raw_data` folder.
    """
    
    # Ensure the raw_data folder exists
    os.makedirs(raw_data_folder, exist_ok=True)
    
    # Log dataset download process
    logging.info(f"Starting download of dataset: {dataset_name}...")

    try:
        # Download the dataset using kagglehub
        download_path = kagglehub.dataset_download(dataset_name)
        logging.info(f"Downloaded dataset to: {download_path}")
    except Exception as e:
        logging.error(f"Failed to download the dataset: {e}")
        raise RuntimeError(f"Error downloading dataset: {e}")
    
    # Recursively search for the first CSV file in the downloaded folder
    logging.info("Searching for CSV file in the downloaded folder...")
    
    csv_file = None
    for dirpath, dirnames, filenames in os.walk(download_path):
        for file in filenames:
            if file.endswith('.csv'):
                csv_file = os.path.join(dirpath, file)
                break  # Stop once we find the first CSV file
        if csv_file:
            break

    if csv_file:
        # If CSV file is found, move it to the raw_data folder and rename it
        destination_file_path = os.path.join(raw_data_folder, 'raw.csv')
        
        # If raw.csv exists, remove it before moving the new file
        if os.path.exists(destination_file_path):
            logging.info(f"Replacing existing raw.csv with the new file.")
            os.remove(destination_file_path)
        
        # Move the CSV file and rename it to raw.csv
        shutil.move(csv_file, destination_file_path)
        logging.info(f"Moved and renamed {csv_file} to {destination_file_path}")
    else:
        logging.error("No CSV file found in the downloaded dataset.")
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")
    
    # Print final path where the dataset is stored
    logging.info(f"Dataset is now stored in: {destination_file_path}")
    return destination_file_path


if __name__ == "__main__":
    # Define dataset and raw_data folder path
    dataset_name = "jsphyg/weather-dataset-rattle-package"  # You can change this as needed
    raw_data_folder = os.path.join('../../data', 'raw_data')  # Adjust path as needed
    
    # Call the function to download and process the data
    try:
        download_and_process_data(dataset_name, raw_data_folder)
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
        exit(1)