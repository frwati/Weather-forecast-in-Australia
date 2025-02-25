import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging

<<<<<<< HEAD
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
=======

def fill_missing_temps(df, date_column='Date', column=None):
    """
    Fills missing values in the specified column by taking the median value
    of the same day (ignoring the year) across all years.

    Parameters:
    df (pd.DataFrame): The input DataFrame with a date column and a temperature column.
    date_column (str): The name of the date column.
    column (str): The name of the column to fill missing values for.

    Returns:
    pd.DataFrame: DataFrame with missing values in the specified column filled.
    """
    if column is None:
        raise ValueError("You must specify the column to fill missing values for.")

    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Extract day and month to group by
    df['day_month'] = df[date_column].dt.strftime('%m-%d')

    # Compute the median temp for each day-month combination
    medians = df.groupby('day_month')[column].median()

    # Fill NaN values in the temp column with the median value
    df.loc[:, column] = df.apply(
        lambda row: medians[row['day_month']] if pd.isna(row[column]) else row[column],
        axis=1
    )

    # Drop the helper day_month column
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    df.drop(columns=['day_month'], inplace=True)

    return df

<<<<<<< HEAD
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

=======

def clean_data(df):
    """
    Cleans the input weather dataset by:
    - Removing rows with no RainTomorrow data and converting it to binary format.
    - Filling missing values in relevant columns.
    - Dropping columns with more than 30% missing values.
    - Encoding categorical data and adding time-based features.
    """
    # Drop the identified columns
    df = df.drop(columns=['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'])
    
    # Remove rows with no RainTomorrow data available and convert the str into int values
    df = df.dropna(subset=['RainTomorrow'])
    df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

    # Fill nan values in RainToday to 'No' and replace the str into int values
    df['RainToday'] = df['RainToday'].fillna('No')
    df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})

    # Ensure Date is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # 'coerce' will turn invalid dates into NaT

    # Check if there are any invalid date values
    if df['Date'].isnull().any():
        logging.warning("Some rows have invalid or missing dates. These will be dropped.")
        df = df.dropna(subset=['Date'])

    # Create time-based features
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
<<<<<<< HEAD
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
=======
    
    # Using "IsWeekend" to indicate if the day is a weekend (Saturday/Sunday)
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
 
    # Filling missing values in specific columns
    columns_to_fillup = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
                         'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 
                         ]
    for column in columns_to_fillup:
        if column in df.columns:
            df = fill_missing_temps(df, 'Date', column=column)
        else:
            logging.info(f"Column '{column}' not found in DataFrame. Skipping.")
            
    # Removing rows containing missing values 
    df = df.dropna()

    return df


def split_data(input_dir, output_dir, test_size=0.2, random_state=42):
    """
    Splits the weather dataset into train and test sets and saves them.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    logging.info(f"Loading dataset from {input_dir}")
    df = pd.read_csv(input_dir)

    # Clean data
    df = clean_data(df)

    # Feature and target variable
    X = df.drop(['RainTomorrow', 'Date'], axis=1)
    y = df['RainTomorrow']

    # Split the train and test dataset
    logging.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Save the split datasets
    logging.info("Saving split datasets")
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

<<<<<<< HEAD
def main():
    """
    Main function to process and split the dataset.
=======

def main(input_file="data/raw_data/current.csv", output_dir="data/split_data"):
    """
    Main function to handle data splitting.

    Parameters:
        input_file (str): Path to the input raw data file.
        output_dir (str): Path to save the preprocessed file.
>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

<<<<<<< HEAD
    input_file = os.path.join(PROJECT_ROOT, "data/raw_data/current.csv")
    output_dir = os.path.join(PROJECT_ROOT, "data/split_data")

    split_data(input_file, output_dir)

=======
    split_data(input_file, output_dir)


>>>>>>> 60a6f7160f0a0c68122143a5e5ef1ebf3b6e2495
if __name__ == "__main__":
    main()
