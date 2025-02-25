import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import logging
import joblib

def save_preprocessors(scalers, label_encoders, scaler_dir='preprocessing/scalers', le_dir='preprocessing/label_encoders'):
<<<<<<< HEAD
    os.makedirs(scaler_dir, exist_ok=True)
    os.makedirs(le_dir, exist_ok=True)
    
    for column, scaler in scalers.items():
        joblib.dump(scaler, os.path.join(scaler_dir, f"{column}_scaler.pkl"))
    
=======
    """
    Save the scalers and label encoders to disk.
    """
    os.makedirs(scaler_dir, exist_ok=True)
    os.makedirs(le_dir, exist_ok=True)
    
    # Save each scaler
    for column, scaler in scalers.items():
        joblib.dump(scaler, os.path.join(scaler_dir, f"{column}_scaler.pkl"))
    
    # Save each label encoder
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    for column, le in label_encoders.items():
        joblib.dump(le, os.path.join(le_dir, f"{column}_le.pkl"))
    
    logging.info("Scalers and label encoders saved successfully.")

def normalize_data(input_dir, output_dir):
<<<<<<< HEAD
    os.makedirs(output_dir, exist_ok=True)
    
=======
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
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    logging.info("Loading training and testing datasets...")
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv"))
    
<<<<<<< HEAD
    if X_train.isnull().sum().any() or X_test.isnull().sum().any():
        raise ValueError("Input datasets contain missing values. Please preprocess the data before normalization.")
    
=======
    # Check for missing values
    if X_train.isnull().sum().any() or X_test.isnull().sum().any():
        raise ValueError("Input datasets contain missing values. Please preprocess the data before normalization.")
    
    # Columns for scaling
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    logging.info("Normalizing the datasets...")
    scale_columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am',
                     'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
                     'Pressure3pm', 'Temp9am', 'Temp3pm']
<<<<<<< HEAD
    scalers = {}
    
    label_encoders = {}
    
    for column in scale_columns:
        if column in X_train.columns and column in X_test.columns:
            scaler = StandardScaler()
            X_train[column] = scaler.fit_transform(X_train[[column]])
            X_test[column] = scaler.transform(X_test[[column]])
=======
    
    # Initialize scalers dictionary
    scalers = {}
    
    # Initialize label encoders dictionary
    label_encoders = {}
    
    # Fit scaler for each column and store them
    for column in scale_columns:
        if column in X_train.columns and column in X_test.columns:
            scaler = StandardScaler()
            X_train.loc[:, column] = scaler.fit_transform(X_train[[column]])
            X_test.loc[:, column] = scaler.transform(X_test[[column]])
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
            scalers[column] = scaler
        else:
            logging.info(f"Column '{column}' not found in DataFrame. Skipping.")
    
<<<<<<< HEAD
    encode_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
    
=======
    # Columns for encoding
    encode_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
    
    # Encode categorical columns and save each label encoder
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    for column in encode_columns:
        if column in X_train.columns and column in X_test.columns:
            le = LabelEncoder()
            X_train[column] = le.fit_transform(X_train[column])
<<<<<<< HEAD
=======
            # For unseen categories in test, use .transform() safely or assign a default
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
            X_test[column] = X_test[column].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            label_encoders[column] = le
        else:
            logging.info(f"Column '{column}' not found in DataFrame. Skipping.")
    
<<<<<<< HEAD
=======
    # Save the normalized datasets
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    logging.info("Saving normalized datasets...")
    X_train.to_csv(os.path.join(output_dir, "X_train_scaled.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test_scaled.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
<<<<<<< HEAD
=======
    # Save the scalers and label encoders
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    logging.info("Saving scalers and label encoders...")
    save_preprocessors(scalers, label_encoders)
    logging.info(f"Normalized datasets and preprocessors saved to {output_dir}")

<<<<<<< HEAD
def get_project_root():
    return "/opt/airflow" if os.path.exists("/.dockerenv") else os.getcwd()

def main():
    project_root = get_project_root()
    input_dir = os.path.join(project_root, "data/split_data")
    output_dir = os.path.join(project_root, "data/normalized_data")
    
=======

def main(input_dir="../../data/split_data", output_dir="../../data/normalized_data"):
    """
    Main function to normalize data and save the output.
    
    Parameters: 
        input_dir (str): Path to the directory containing raw processed data.
        output_dir (str): Path to save the normalized data.
    """
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    normalize_data(input_dir, output_dir)
<<<<<<< HEAD
=======
    
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495

if __name__ == "__main__":
    main()
