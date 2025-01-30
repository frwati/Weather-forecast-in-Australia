from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load the saved scalers and label encoders
scalers = {}
label_encoders = {}

# Load the scaler for each numerical column
scale_columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am',
                 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
                 'Pressure3pm', 'Temp9am', 'Temp3pm']
for column in scale_columns:
    scalers[column] = joblib.load(f"preprocessing/scalers/{column}_scaler.pkl")

# Load the label encoder for each categorical column
encode_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
for column in encode_columns:
    label_encoders[column] = joblib.load(f"preprocessing/label_encoders/{column}_le.pkl")

# Define a FastAPI app
app = FastAPI()

# Define input model for prediction
class WeatherData(BaseModel):
    Date: str
    Location: str
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    WindGustDir: str
    WindGustSpeed: float    
    WindDir9am: str
    WindDir3pm: str
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    Temp9am: float
    Temp3pm: float
    RainToday: str

# Preprocessing function for incoming data
def preprocess_input_data(weather_data):
    """
    Preprocesses the incoming weather data before feeding it to the model.
    """
    # Convert input data to DataFrame
    df = pd.DataFrame([weather_data.dict()])
    
    # Convert 'RainToday' to number (binary: 0 for 'No', 1 for 'Yes')
    df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})

    # Convert 'Date' to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    # Drop the 'Date' column (since it's no longer needed) and other unnecessary columns
    df = df.drop(columns=['Date'])
    
    # Scale the numerical features
    for column in scale_columns:
        if column in df.columns:
            df[column] = scalers[column].transform(df[[column]])

    # Encode the categorical columns
    for column in encode_columns:
        if column in df.columns:
            df[column] = label_encoders[column].transform(df[column])

    return df

@app.post("/predict")
async def predict(weather_data: WeatherData):
    """
    Predict whether it will rain tomorrow based on the input weather data.
    """
    try:
        # Preprocess the incoming weather data
        preprocessed_data = preprocess_input_data(weather_data)

        # Load your trained model here (assuming it's saved as 'model.pkl')
        model = joblib.load('models/trained_model.pkl')

        # Make prediction
        prediction = model.predict(preprocessed_data)
        
        # Map prediction to 'Yes' or 'No'
        result = "Yes" if prediction[0] == 1 else "No"
        
        return {"RainTomorrow": result}

    except Exception as e:
        print(f"Error during prediction: {e}")  
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/")
def root():
    return {"message": "Welcome to the Weather Prediction API!"}
