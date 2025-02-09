from pydantic import BaseModel
import joblib
import pandas as pd
import bentoml
from bentoml.io import JSON 
from starlette.responses import JSONResponse
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from datetime import datetime, timedelta
import os

# Secret key and algorithm for JWT authentication
JWT_SECRET_KEY = "weather-report"
JWT_ALGORITHM = "HS256"

# User credentials for authentication
USERS = {
    "user123" : "password123",
    "user456" : "password456"
}

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/predict-weather":
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})
            
            try:
                token = token.split()[1]
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid Token"})
            
            request.state.user = payload.get("sub")
        
        response = await call_next(request)
        return response


# Load the saved scalers and label encoders
scalers = {}
label_encoders = {}

# Define the base directory for loading the preprocessing files (you can modify this as needed)
preprocessing_dir = "preprocessing"

# Load the scaler for each numerical column
scale_columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am',
                 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
                 'Pressure3pm', 'Temp9am', 'Temp3pm']
for column in scale_columns:
    scaler_path = os.path.join(preprocessing_dir, 'scalers', f"{column}_scaler.pkl")
    scalers[column] = joblib.load(scaler_path)

# Load the label encoder for each categorical column
encode_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
for column in encode_columns:
    encoder_path = os.path.join(preprocessing_dir, 'label_encoders', f"{column}_le.pkl")
    label_encoders[column] = joblib.load(encoder_path)


# Define input model for prediction
class InputModel(BaseModel):
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
def preprocess_input_data(input_model:InputModel):
    """
    Preprocesses the incoming weather data before feeding it to the model.
    """
    # Convert input data to DataFrame
    df = pd.DataFrame([input_model.dict()])
    
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

weather_rf_runner = bentoml.sklearn.get("weather_rf_model:latest").to_runner()

weather_service = bentoml.Service("weather_service", runners=[weather_rf_runner])

weather_service.add_asgi_middleware(JWTAuthMiddleware)

# Create an API endpoint for the service
@weather_service.api(input=JSON(), output=JSON(), route="/login")
def login(credentials: dict) -> dict:
    username = credentials.get("username")
    password = credentials.get("password")
    
    if username in USERS and USERS[username] == password:
        token = create_jwt_token(username)
        return {"token": token}
    else:
        return {"detail": "Invalid credentials"}
    
# Create an API endpoint for the service
@weather_service.api(
    input=JSON(pydantic_model=InputModel), 
    output=JSON(),
    route="/predict-weather")

async def classify(input_data: InputModel, ctx: bentoml.Context) -> dict:
    """
    Predict whether it will rain tomorrow based on the input weather data.
    """
    request = ctx.request
    user = request.state.user if hasattr(request.state, 'user') else None
    
    preprocessed_data = preprocess_input_data(input_data)
           
    # Convert the preprocessed data to a NumPy array (model typically accepts this format)
    input_array = preprocessed_data.values

    # Run the model to get a prediction
    result = await weather_rf_runner.predict.async_run(input_array.reshape(1, -1))
    
    return {
        "prediction": result.tolist()
    }
    
# Function to create a JWT token
def create_jwt_token(user_id: str):
    """Generate a JWT token for authentication."""
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {
        "sub": user_id,
        "exp": expiration
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


