#!/usr/bin/env python3
"""
Script to query the live deployed API.
Tests both GET and POST endpoints and displays results.
Usage: python scripts/query_live.py
"""

import requests
import os
import sys
import json

# Get API URL from environment or use default
LIVE_API_URL = os.environ.get("LIVE_API_URL", "http://localhost:8000")

def test_get_endpoint():
    """Test the GET / endpoint."""
    print(f"\n{'='*60}")
    print("Testing GET / endpoint")
    print(f"{'='*60}")
    
    try:
        response = requests.get(f"{LIVE_API_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_post_predict():
    """Test the POST /predict endpoint with sample data."""
    print(f"\n{'='*60}")
    print("Testing POST /predict endpoint")
    print(f"{'='*60}")
    
    # Sample person data - likely to earn >50K
    sample_data = {
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
    
    print(f"\nSending data:")
    print(json.dumps(sample_data, indent=2))
    
    try:
        response = requests.post(
            f"{LIVE_API_URL}/predict",
            json=sample_data
        )
        print(f"\nStatus Code: {response.status_code}")
        print(f"Prediction: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_post_predict_low_income():
    """Test POST /predict with low income profile."""
    print(f"\n{'='*60}")
    print("Testing POST /predict with low income profile")
    print(f"{'='*60}")
    
    # Sample person data - likely to earn <=50K
    sample_data = {
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
    
    print(f"\nSending data:")
    print(json.dumps(sample_data, indent=2))
    
    try:
        response = requests.post(
            f"{LIVE_API_URL}/predict",
            json=sample_data
        )
        print(f"\nStatus Code: {response.status_code}")
        print(f"Prediction: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all API tests."""
    print(f"\n{'#'*60}")
    print(f"# Testing Live API: {LIVE_API_URL}")
    print(f"{'#'*60}")
    
    results = {
        "GET /": test_get_endpoint(),
        "POST /predict (high income)": test_post_predict(),
        "POST /predict (low income)": test_post_predict_low_income()
    }
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:40s} {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'All tests passed!' if all_passed else 'Some tests failed.'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

