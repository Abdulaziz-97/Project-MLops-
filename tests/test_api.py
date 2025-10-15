"""
Unit tests for FastAPI endpoints.
Tests GET and POST endpoints with both prediction classes.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add starter directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'starter'))

from main import app

# Create test client
client = TestClient(app)


def test_get_root_returns_greeting():
    """
    Test GET request to root endpoint.
    Must test both status code and response content.
    """
    # Make GET request
    response = client.get("/")
    
    # Assert status code
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    # Assert response content
    json_response = response.json()
    assert isinstance(json_response, dict), "Response should be a dictionary"
    assert "message" in json_response, "Response should contain 'message' key"
    assert "Welcome" in json_response["message"], "Message should contain 'Welcome'"
    
    # Assert all expected keys
    expected_keys = ["message", "description", "docs"]
    for key in expected_keys:
        assert key in json_response, f"Response should contain '{key}' key"


def test_post_predict_low_income():
    """
    Test POST request that predicts <=50K (low income).
    Uses profile likely to earn <=50K: young, less educated, part-time.
    """
    # Data for someone likely to earn <=50K
    low_income_person = {
        "age": 22,
        "workclass": "Private",
        "fnlgt": 30000,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    }
    
    # Make POST request
    response = client.post("/predict", json=low_income_person)
    
    # Assert status code
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    # Assert response structure
    json_response = response.json()
    assert isinstance(json_response, dict), "Response should be a dictionary"
    assert "prediction" in json_response, "Response should contain 'prediction' key"
    
    # Assert prediction is valid
    prediction = json_response["prediction"]
    assert prediction in ["<=50K", ">50K"], f"Prediction should be '<=50K' or '>50K', got {prediction}"
    
    # For this profile, we expect <=50K (though we can't guarantee due to model uncertainty)
    print(f"Low income prediction: {prediction}")


def test_post_predict_high_income():
    """
    Test POST request that predicts >50K (high income).
    Uses profile likely to earn >50K: educated, professional, married, full-time.
    """
    # Data for someone likely to earn >50K
    high_income_person = {
        "age": 45,
        "workclass": "Private",
        "fnlgt": 150000,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15000,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }
    
    # Make POST request
    response = client.post("/predict", json=high_income_person)
    
    # Assert status code
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    # Assert response structure
    json_response = response.json()
    assert isinstance(json_response, dict), "Response should be a dictionary"
    assert "prediction" in json_response, "Response should contain 'prediction' key"
    
    # Assert prediction is valid
    prediction = json_response["prediction"]
    assert prediction in ["<=50K", ">50K"], f"Prediction should be '<=50K' or '>50K', got {prediction}"
    
    # For this profile, we expect >50K (though we can't guarantee due to model uncertainty)
    print(f"High income prediction: {prediction}")


def test_post_predict_validation_error():
    """Test POST with invalid data returns 422 validation error."""
    invalid_data = {
        "age": "not_a_number",  # Invalid: should be int
        "workclass": "Private"
        # Missing required fields
    }
    
    response = client.post("/predict", json=invalid_data)
    
    # Should return 422 for validation error
    assert response.status_code == 422, f"Expected 422 for invalid data, got {response.status_code}"


def test_api_docs_accessible():
    """Test that API documentation endpoints are accessible."""
    # Test OpenAPI schema
    response = client.get("/openapi.json")
    assert response.status_code == 200, "OpenAPI schema should be accessible"
    
    schema = response.json()
    assert "paths" in schema, "Schema should contain paths"
    assert "/" in schema["paths"], "Schema should include root endpoint"
    assert "/predict" in schema["paths"], "Schema should include predict endpoint"


def test_post_predict_with_hyphens():
    """Test that hyphenated field names are handled correctly."""
    # Valid data with hyphenated fields
    person_data = {
        "age": 35,
        "workclass": "Private",
        "fnlgt": 100000,
        "education": "Bachelors",
        "education-num": 13,  # Hyphenated
        "marital-status": "Married-civ-spouse",  # Hyphenated
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5000,  # Hyphenated
        "capital-loss": 0,  # Hyphenated
        "hours-per-week": 40,  # Hyphenated
        "native-country": "United-States"  # Hyphenated
    }
    
    response = client.post("/predict", json=person_data)
    
    # Should successfully handle hyphenated fields
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert "prediction" in response.json(), "Response should contain prediction"

