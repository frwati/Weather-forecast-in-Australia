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

    if csv_file:
        # Define the destination file path
        destination_file_path = os.path.join(raw_data_folder, "raw.csv")

        # If raw.csv exists, remove it before copying the new file
        if os.path.exists(destination_file_path):
            logging.info("Replacing existing raw.csv with the new file.")
            os.remove(destination_file_path)

        # Copy the CSV file and rename it to raw.csv
        try:
            shutil.copy(csv_file, destination_file_path)
            logging.info(f"Copied {csv_file} to {destination_file_path}")
        except Exception as e:
            logging.error(f"Error copying file: {e}")
            raise e
    else:
        logging.error("No CSV file found in the downloaded dataset.")
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")

    # Print final path where the dataset is stored
    logging.info(f"Dataset is now stored in: {destination_file_path}")
    return destination_file_path


def main():
    # Define absolute path for project directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    raw_data_folder = os.path.join(project_root, "data", "raw_data")

    dataset_name = "jsphyg/weather-dataset-rattle-package"  # Modify as needed

    try:
        download_and_process_data(dataset_name, raw_data_folder)
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
