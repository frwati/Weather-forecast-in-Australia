import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging

if os.path.exists("/.dockerenv"):
    PROJECT_ROOT = "/opt/airflow"  
else:
    PROJECT_ROOT = os.getcwd()  

logging.info(f"Running inside: {PROJECT_ROOT}")

def fill_missing_temps(df, date_column='Date', column=None):
    """
    Fill missing values based on the median values of the same day across different years.
    """
    if column is None:
        raise ValueError("Column to fill missing values must be specified.")

    df[date_column] = pd.to_datetime(df[date_column])
    df['day_month'] = df[date_column].dt.strftime('%m-%d')

    medians = df.groupby('day_month')[column].median()

    df[column] = df.apply(lambda row: medians[row['day_month']] if pd.isna(row[column]) else row[column], axis=1)
    df.drop(columns=['day_month'], inplace=True)

    return df

def clean_data(df):
    """
    Clean weather data by:
    - Dropping irrelevant columns.
    - Converting categorical values to numerical.
    - Adding temporal features (day, month, year, etc.).
    """
    df.drop(columns=['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], errors='ignore', inplace=True)

    df.dropna(subset=['RainTomorrow'], inplace=True)
    df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})
    df['RainToday'] = df['RainToday'].fillna('No').map({'No': 0, 'Yes': 1})

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if df['Date'].isnull().any():
        logging.warning("Invalid dates detected and will be removed.")
        df.dropna(subset=['Date'], inplace=True)

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    columns_to_fill = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
                       'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
    
    for column in columns_to_fill:
        if column in df.columns:
            df = fill_missing_temps(df, 'Date', column)
        else:
            logging.info(f"Column '{column}' not found, skipping.")

    df.dropna(inplace=True)

    return df

def split_data(input_path, output_dir, test_size=0.2, random_state=42):
    """
    Split weather data into training and testing sets and store them.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    df = clean_data(df)

    X = df.drop(['RainTomorrow', 'Date'], axis=1)
    y = df['RainTomorrow']

    logging.info("Splitting data into training and testing sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    logging.info("Saving split data")
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

def main():
    """
    Main function to process and split the dataset.
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    input_file = os.path.join(PROJECT_ROOT, "data/raw_data/current.csv")
    output_dir = os.path.join(PROJECT_ROOT, "data/split_data")

    split_data(input_file, output_dir)

if __name__ == "__main__":
    main()
