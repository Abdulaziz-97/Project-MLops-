#!/usr/bin/env python3
"""
Script to compute model performance metrics on data slices.
Analyzes performance across different categorical feature values.
"""

import pandas as pd
import pickle
import sys
import os
from sklearn.model_selection import train_test_split

# Add starter directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'starter'))

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference


def compute_slice_metrics(model, X, y, feature_name, encoder, lb, raw_df, cat_features):
    """
    Compute model performance metrics on slices of data.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model
    X : np.array
        Processed features
    y : np.array
        True labels
    feature_name : str
        Name of categorical feature to slice on
    encoder : OneHotEncoder
        Fitted encoder
    lb : LabelBinarizer
        Fitted label binarizer
    raw_df : pd.DataFrame
        Raw dataframe with original feature values
    cat_features : list
        List of categorical feature names
        
    Returns
    -------
    dict
        Dictionary mapping feature values to their metrics
    """
    results = {}
    unique_values = raw_df[feature_name].unique()
    
    print(f"\nComputing metrics for feature: {feature_name}")
    print("=" * 70)
    print(f"{'Value':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Count':<8}")
    print("-" * 70)
    
    for value in sorted(unique_values):
        # Filter rows for this feature value
        mask = raw_df[feature_name] == value
        slice_df = raw_df[mask].copy()
        
        if len(slice_df) == 0:
            continue
        
        # Process the slice
        try:
            X_slice, y_slice, _, _ = process_data(
                slice_df,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb
            )
            
            # Skip if not enough data or only one class
            if len(y_slice) < 2 or len(set(y_slice)) < 2:
                print(f"{str(value):<30} {'N/A':<12} {'N/A':<12} {'N/A':<12} {len(y_slice):<8}")
                continue
            
            # Get predictions
            preds = inference(model, X_slice)
            
            # Compute metrics
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            
            # Store results
            results[value] = {
                'precision': precision,
                'recall': recall,
                'fbeta': fbeta,
                'count': len(y_slice)
            }
            
            # Print results
            print(f"{str(value):<30} {precision:<12.4f} {recall:<12.4f} {fbeta:<12.4f} {len(y_slice):<8}")
            
        except Exception as e:
            print(f"{str(value):<30} Error: {e}")
            continue
    
    return results


def main():
    """Main function to run slice analysis."""
    print("\n" + "="*70)
    print("MODEL PERFORMANCE ON DATA SLICES")
    print("="*70)
    
    # Load model and artifacts
    print("\nLoading model and artifacts...")
    try:
        with open("starter/model/model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("starter/model/encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        with open("starter/model/labelizer.pkl", "rb") as f:
            lb = pickle.load(f)
        print("✓ Model artifacts loaded successfully")
    except FileNotFoundError as e:
        print(f"Error: Could not find model files. Train model first.")
        print(f"Details: {e}")
        return 1
    
    # Load data
    print("\nLoading data...")
    try:
        data = pd.read_csv("starter/data/census_clean.csv")
        print(f"✓ Data loaded: {data.shape}")
    except FileNotFoundError:
        print("Error: census_clean.csv not found")
        return 1
    
    # Split data (same as training)
    _, test = train_test_split(data, test_size=0.20, random_state=42)
    
    # Define categorical features
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    
    # Process test data
    print("\nProcessing test data...")
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # Compute overall metrics
    print("\nComputing overall metrics...")
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    
    print(f"\nOverall Performance:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {fbeta:.4f}")
    print(f"  Test size: {len(y_test)}")
    
    # Open output file
    with open("slice_output.txt", "w") as f:
        f.write("Model Performance on Data Slices\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Overall Performance:\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n")
        f.write(f"  F1-Score:  {fbeta:.4f}\n")
        f.write(f"  Test size: {len(y_test)}\n\n")
        
        # Compute and write slice metrics for each categorical feature
        for feature in cat_features:
            print(f"\n{'='*70}")
            f.write(f"\n{'='*70}\n")
            f.write(f"Feature: {feature}\n")
            f.write("=" * 70 + "\n")
            f.write(f"{'Value':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Count':<8}\n")
            f.write("-" * 70 + "\n")
            
            results = compute_slice_metrics(
                model, X_test, y_test, feature, encoder, lb, test, cat_features
            )
            
            # Write to file
            for value, metrics in results.items():
                f.write(
                    f"{str(value):<30} "
                    f"{metrics['precision']:<12.4f} "
                    f"{metrics['recall']:<12.4f} "
                    f"{metrics['fbeta']:<12.4f} "
                    f"{metrics['count']:<8}\n"
                )
    
    print(f"\n{'='*70}")
    print("✓ Slice analysis complete!")
    print("✓ Results saved to slice_output.txt")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

