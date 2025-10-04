"""
FastAPI application for Census Income Prediction.
Provides REST API endpoints for salary prediction based on demographic data.
"""

import pickle
import os
import pandas as pd
import numpy as np
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Import our ML functions
from starter.ml.data import process_data
from starter.ml.model import inference

# Create FastAPI app
app = FastAPI(
    title="Census Income Prediction API",
    description="Predict whether a person makes over $50K based on demographic features",
    version="1.0.0"
)

# Load model and encoders at startup
model = None
encoder = None
lb = None

def load_model():
    """Load the trained model and encoders."""
    global model, encoder, lb
    
    # Try different paths for model files
    possible_model_paths = [
        "model/model.pkl",  # Local development
        "../model/model.pkl",  # Alternative local path
        "/app/starter/model/model.pkl",  # Heroku absolute path
    ]
    
    possible_encoder_paths = [
        "model/encoder.pkl",
        "../model/encoder.pkl", 
        "/app/starter/model/encoder.pkl",
    ]
    
    possible_lb_paths = [
        "model/labelizer.pkl",
        "../model/labelizer.pkl",
        "/app/starter/model/labelizer.pkl", 
    ]
    
    # Try to load model files
    model_loaded = False
    for model_path, encoder_path, lb_path in zip(possible_model_paths, possible_encoder_paths, possible_lb_paths):
        try:
            if os.path.exists(model_path) and os.path.exists(encoder_path) and os.path.exists(lb_path):
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                
                with open(encoder_path, "rb") as f:
                    encoder = pickle.load(f)
                    
                with open(lb_path, "rb") as f:
                    lb = pickle.load(f)
                
                model_loaded = True
                print(f"Model loaded successfully from {model_path}")
                break
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            continue
    
    if not model_loaded:
        print("WARNING: Could not load pre-trained model. Training new model...")
        # Train model on-demand for Heroku
        from starter.ml.model import train_model
        from starter.ml.data import process_data
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        # Create minimal training data
        demo_data = {
            'age': [39, 50, 38, 53, 28] * 50,
            'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 'Private'] * 50,
            'fnlgt': [77516, 83311, 215646, 234721, 338409] * 50,
            'education': ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Bachelors'] * 50,
            'education-num': [13, 13, 9, 7, 13] * 50,
            'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-civ-spouse', 'Married-civ-spouse'] * 50,
            'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Handlers-cleaners', 'Prof-specialty'] * 50,
            'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Husband', 'Wife'] * 50,
            'race': ['White', 'White', 'White', 'Black', 'Black'] * 50,
            'sex': ['Male', 'Male', 'Male', 'Male', 'Female'] * 50,
            'capital-gain': [2174, 0, 0, 0, 0] * 50,
            'capital-loss': [0, 0, 0, 0, 0] * 50,
            'hours-per-week': [40, 13, 40, 40, 40] * 50,
            'native-country': ['United-States'] * 250,
            'salary': ['<=50K', '<=50K', '<=50K', '<=50K', '>50K'] * 50  # Add some >50K samples
        }
        data = pd.DataFrame(demo_data)
        
        # Train-test split
        train, test = train_test_split(data, test_size=0.20, random_state=42)
        
        # Define categorical features
        cat_features = [
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country",
        ]
        
        # Process training data
        X_train, y_train, encoder, lb = process_data(
            train, categorical_features=cat_features, label="salary", training=True
        )
        
        # Train model
        model = train_model(X_train, y_train)
        print("Model trained successfully with demo data")

# Load model on startup
load_model()

# Define Pydantic model for input data
class CensusData(BaseModel):
    """
    Input data model for census prediction.
    
    Note: Uses Field(alias=...) to handle column names with hyphens
    since Python variables can't contain hyphens.
    """
    age: int = Field(..., description="Age in years", example=39)
    workclass: Literal[
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
        "Local-gov", "State-gov", "Without-pay", "Never-worked", "?"
    ] = Field(..., description="Work class", example="State-gov")
    fnlgt: int = Field(..., description="Final weight", example=77516)
    education: Literal[
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
        "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
        "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool", "?"
    ] = Field(..., description="Education level", example="Bachelors")
    education_num: int = Field(..., alias="education-num", description="Education number", example=13)
    marital_status: Literal[
        "Married-civ-spouse", "Divorced", "Never-married", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ] = Field(..., alias="marital-status", description="Marital status", example="Never-married")
    occupation: Literal[
        "Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Prof-specialty",
        "Other-service", "Sales", "Craft-repair", "Transport-moving",
        "Farming-fishing", "Machine-op-inspct", "Tech-support", "Protective-serv",
        "Armed-Forces", "Priv-house-serv", "?"
    ] = Field(..., description="Occupation", example="Adm-clerical")
    relationship: Literal[
        "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
    ] = Field(..., description="Relationship", example="Not-in-family")
    race: Literal[
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
    ] = Field(..., description="Race", example="White")
    sex: Literal["Female", "Male"] = Field(..., description="Sex", example="Male")
    capital_gain: int = Field(..., alias="capital-gain", description="Capital gains", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", description="Capital losses", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", description="Hours per week", example=40)
    native_country: Literal[
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
        "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
        "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
        "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
        "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
        "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
        "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands", "?"
    ] = Field(..., alias="native-country", description="Native country", example="United-States")
    
    class Config:
        """Pydantic configuration."""
        # Allow population by field name or alias
        allow_population_by_field_name = True
        # Example data for automatic documentation
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White", 
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: Literal["<=50K", ">50K"] = Field(..., description="Salary prediction")


@app.get("/", response_model=dict)
def welcome():
    """
    Welcome endpoint - returns a greeting message.
    
    Returns:
        dict: Welcome message
    """
    return {
        "message": "Welcome to the Census Income Prediction API!",
        "description": "Use POST /predict to get salary predictions based on demographic data",
        "docs": "Visit /docs for interactive API documentation"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CensusData):
    """
    Predict salary category based on demographic features.
    
    Args:
        data: Census demographic data
        
    Returns:
        PredictionResponse: Salary prediction (<=50K or >50K)
    """
    # Convert Pydantic model to DataFrame
    # Handle the alias fields properly
    input_dict = {
        "age": data.age,
        "workclass": data.workclass,
        "fnlgt": data.fnlgt,
        "education": data.education,
        "education-num": data.education_num,
        "marital-status": data.marital_status,
        "occupation": data.occupation,
        "relationship": data.relationship,
        "race": data.race,
        "sex": data.sex,
        "capital-gain": data.capital_gain,
        "capital-loss": data.capital_loss,
        "hours-per-week": data.hours_per_week,
        "native-country": data.native_country
    }
    
    # Create DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Define categorical features (same as training)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship", 
        "race",
        "sex",
        "native-country",
    ]
    
    # Process the data (inference mode)
    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # Make prediction
    prediction = inference(model, X)
    
    # Convert prediction back to label
    prediction_label = lb.inverse_transform(prediction)[0]
    
    return PredictionResponse(prediction=prediction_label)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
