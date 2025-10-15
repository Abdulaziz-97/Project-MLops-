"""
Unit tests for ML model functions.
Tests are deterministic with fixed random seeds.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import sys
import os

# Add starter directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'starter'))

from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference


@pytest.fixture
def sample_data():
    """Create deterministic sample data for testing."""
    return pd.DataFrame({
        'age': [39, 50, 38, 53, 28],
        'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 'Private'],
        'fnlgt': [77516, 83311, 215646, 234721, 338409],
        'education': ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Bachelors'],
        'education-num': [13, 13, 9, 7, 13],
        'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-civ-spouse', 'Married-civ-spouse'],
        'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Handlers-cleaners', 'Prof-specialty'],
        'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Husband', 'Wife'],
        'race': ['White', 'White', 'White', 'Black', 'Black'],
        'sex': ['Male', 'Male', 'Male', 'Male', 'Female'],
        'capital-gain': [2174, 0, 0, 0, 0],
        'capital-loss': [0, 0, 0, 0, 0],
        'hours-per-week': [40, 13, 40, 40, 40],
        'native-country': ['United-States', 'United-States', 'United-States', 'United-States', 'Cuba'],
        'salary': ['<=50K', '<=50K', '<=50K', '<=50K', '>50K']
    })


@pytest.fixture
def cat_features():
    """Categorical features for testing."""
    return [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]


def test_train_model_returns_estimator(cat_features, sample_data):
    """
    Test that train_model returns a fitted RandomForestClassifier.
    Uses deterministic data to ensure reproducibility.
    """
    # Process data
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    
    # Train model
    model = train_model(X, y)
    
    # Assertions
    assert isinstance(model, RandomForestClassifier), "Model should be RandomForestClassifier"
    assert hasattr(model, 'predict'), "Model should have predict method"
    assert hasattr(model, 'n_features_in_'), "Model should be fitted"
    assert model.n_features_in_ == X.shape[1], "Model should know correct number of features"


def test_inference_output_shape():
    """
    Test that inference returns predictions with correct shape.
    Uses small synthetic matrix with fixed seed.
    """
    # Create deterministic synthetic data
    np.random.seed(42)
    X = np.random.random((50, 10))
    y = np.random.randint(0, 2, 50)
    
    # Train model
    model = train_model(X, y)
    
    # Run inference
    predictions = inference(model, X)
    
    # Assertions
    assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
    assert predictions.shape[0] == X.shape[0], f"Should have {X.shape[0]} predictions, got {predictions.shape[0]}"
    assert len(predictions.shape) == 1, "Predictions should be 1-dimensional"
    assert all(pred in [0, 1] for pred in predictions), "Predictions should be binary (0 or 1)"


def test_compute_model_metrics_types():
    """
    Test that compute_model_metrics returns correct types and values.
    Uses known y_true/y_pred to test exact metric calculation.
    """
    # Known test case with deterministic values
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    
    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    # Type assertions
    assert isinstance(precision, (float, np.floating)), "Precision should be float"
    assert isinstance(recall, (float, np.floating)), "Recall should be float"
    assert isinstance(fbeta, (float, np.floating)), "F-beta should be float"
    
    # Range assertions
    assert 0 <= precision <= 1, f"Precision should be [0,1], got {precision}"
    assert 0 <= recall <= 1, f"Recall should be [0,1], got {recall}"
    assert 0 <= fbeta <= 1, f"F-beta should be [0,1], got {fbeta}"
    
    # Exact value assertions for this specific case
    # TP=2, FP=1, FN=1, TN=1
    # Precision = TP/(TP+FP) = 2/3 ≈ 0.6667
    # Recall = TP/(TP+FN) = 2/3 ≈ 0.6667
    assert abs(precision - 0.6667) < 0.01, f"Expected precision ~0.667, got {precision}"
    assert abs(recall - 0.6667) < 0.01, f"Expected recall ~0.667, got {recall}"


def test_process_data_training(sample_data, cat_features):
    """Test that process_data correctly processes training data."""
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    
    # Type assertions
    assert isinstance(X, np.ndarray), "X should be numpy array"
    assert isinstance(y, np.ndarray), "y should be numpy array"
    assert isinstance(encoder, OneHotEncoder), "encoder should be OneHotEncoder"
    assert isinstance(lb, LabelBinarizer), "lb should be LabelBinarizer"
    
    # Shape assertions
    assert X.shape[0] == len(sample_data), "X should have same number of rows as input"
    assert y.shape[0] == len(sample_data), "y should have same number of rows as input"
    assert len(y.shape) == 1, "y should be 1-dimensional"
    
    # Data quality assertions
    assert not np.isnan(X).any(), "X should not contain NaN values"
    assert not np.isnan(y).any(), "y should not contain NaN values"


def test_process_data_inference(sample_data, cat_features):
    """Test that process_data correctly processes inference data."""
    # Get training encoders first
    _, _, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    
    # Test inference mode
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # Assertions
    assert isinstance(X, np.ndarray), "X should be numpy array"
    assert isinstance(y, np.ndarray), "y should be numpy array"
    assert X.shape[0] == len(sample_data), "X should have same number of rows as input"


def test_model_deterministic_with_seed():
    """Test that model training is deterministic with fixed random seed."""
    np.random.seed(42)
    X = np.random.random((100, 10))
    y = np.random.randint(0, 2, 100)
    
    # Train two models with same seed
    model1 = train_model(X, y)
    model2 = train_model(X, y)
    
    # Predictions should be identical
    preds1 = inference(model1, X[:10])
    preds2 = inference(model2, X[:10])
    
    np.testing.assert_array_equal(preds1, preds2, "Models with same seed should give same predictions")

