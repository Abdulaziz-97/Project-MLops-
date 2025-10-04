# Script to train machine learning model.

import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

def main():
    """Main function to train and save the model."""
    
    # Add code to load in the data.
    print("Loading data...")
    data = pd.read_csv("../data/census_clean.csv")
    print(f"Data loaded: {data.shape}")
    
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    print(f"Train size: {train.shape}, Test size: {test.shape}")
    
    # Define categorical features
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
    
    # Process training data
    print("Processing training data...")
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    print(f"Training features shape: {X_train.shape}")
    
    # Process the test data with the process_data function.
    print("Processing test data...")
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", 
        training=False, encoder=encoder, lb=lb
    )
    print(f"Test features shape: {X_test.shape}")
    
    # Train and save a model.
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate on test set
    print("Evaluating model...")
    test_predictions = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, test_predictions)
    
    print(f"Model Performance:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {fbeta:.4f}")
    
    # Create model directory if it doesn't exist
    os.makedirs("../model", exist_ok=True)
    
    # Save the model and encoders
    print("Saving model and encoders...")
    with open("../model/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open("../model/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
        
    with open("../model/labelizer.pkl", "wb") as f:
        pickle.dump(lb, f)
    
    print("Model training complete! Files saved:")
    print("  - ../model/model.pkl")
    print("  - ../model/encoder.pkl") 
    print("  - ../model/labelizer.pkl")

if __name__ == "__main__":
    main()
