import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import logging
import joblib

def save_preprocessors(scaler, le, scaler_path='preprocessing/scaler.pkl', le_path='preprocessing/le.pkl'):
    """
    Save the scaler and label encoder to disk.
    """
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)
    logging.info("Scaler and label encoder saved successfully.")


def normalize_data(input_dir, output_dir):
    """
    Normalizes numerical features and encodes categorical features for training and testing datasets.
    
    Parameters:
        input_dir (str): Path to the directory containing raw split data.
        output_dir (str): Path to save the normalized datasets.
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load the training and testing data
    logging.info("Loading training and testing datasets...")
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv"))
    
    # Check for missing values
    if X_train.isnull().sum().any() or X_test.isnull().sum().any():
        raise ValueError("Input datasets contain missing values. Please preprocess the data before normalization.")
    
    # Columns for scaling
    logging.info("Normalizing the datasets...")
    scale_columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am',
                     'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
                     'Pressure3pm', 'Temp9am', 'Temp3pm']
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit the scaler on the training data and transform both train and test datasets
    for column in scale_columns:
        if column in X_train.columns and column in X_test.columns:
            X_train[[column]] = scaler.fit_transform(X_train[[column]])
            X_test[[column]] = scaler.transform(X_test[[column]])
        else:
            logging.info(f"Column '{column}' not found in DataFrame. Skipping.")
    
    # Columns for encoding
    encode_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
    
    # Initialize label encoder
    le = LabelEncoder()
    
    # Encode categorical columns
    for column in encode_columns:
        # Check for unseen categories in the test set
        X_train[column] = le.fit_transform(X_train[column])
        X_test[column] = le.transform(X_test[column])
    
    # Save the normalized datasets
    logging.info("Saving normalized datasets...")
    X_train.to_csv(os.path.join(output_dir, "X_train_scaled.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test_scaled.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    # Save the scaler and label encoder
    logging.info("Saving scaler and label encoder...")
    save_preprocessors(scaler, le)
    logging.info(f"Normalized datasets and preprocessors saved to {output_dir}")


def main(input_dir="data/split_data", output_dir="data/normalized_data"):
    """
    Main function to normalize data and save the output.
    
    Parameters: 
        input_dir (str): Path to the directory containing raw processed data.
        output_dir (str): Path to save the normalized data.
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    normalize_data(input_dir, output_dir)
    

if __name__ == "__main__":
    main()
