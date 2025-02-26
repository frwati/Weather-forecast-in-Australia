import kagglehub
import shutil
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def download_and_process_data(dataset_name, raw_data_folder):
    """
    Download and process the dataset:
    1. Download the dataset using kagglehub.
    2. Search for the first CSV file.
    3. Copy and rename the CSV to `raw.csv` in the `raw_data` folder.
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
    logging.info("Searching for CSV file in the downloaded dataset folder...")

    csv_file = None
    for dirpath, _, filenames in os.walk(download_path):
        for file in filenames:
            if file.endswith(".csv"):
                csv_file = os.path.join(dirpath, file)
                logging.info(f"Found CSV file: {csv_file}")
                break  # Stop once we find the first CSV file
        if csv_file:
            break
    
    if not csv_file:
        logging.error("No CSV file found in the downloaded dataset.")
        raise FileNotFoundError("No CSV file found in the dataset.")
    
    #Save the latest dataset as 'current.csv'
    current_file_path = os.path.join(raw_data_folder, "current.csv")
 
    # Copy the newly downloaded dataset as current.csv
    shutil.copy(csv_file, current_file_path)
    logging.info(f"Saved new dataset as current.csv: {current_file_path}")

    return current_file_path
    
    
def main():
    # Define absolute path for project directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    raw_data_folder = os.path.join(project_root, "data", "raw_data")

    dataset_name = "jsphyg/weather-dataset-rattle-package"  # Modify as needed

    try:
        cur_path = download_and_process_data(dataset_name, raw_data_folder)
        logging.info(f"Data ingestion complete. Saved to: {cur_path}")
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
