"""
File to test the api
"""

import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_welcome_message():
    """
    Test case to check root of the api
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome!"


def test_less_than_50():
    """
    Test case to check for predictions less than 50
    """
    data = {
        "workclass": "state_gov",
        "education": "bachelors",
        "marital_status": "never_married",
        "occupation": "adm_clerical",
        "relationship": "not_in_family",
        "race": "white",
        "sex": "male",
        "native_country": "united_states",
        "age": 39,
        "fnlgt": 77516,
        "education_num": 13,
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
    }

    response = client.post("/predict", data=json.dumps(data))
    assert response.status_code == 200
    assert response.json() == {'pred': '<=50K'}


def test_greater_than_50():
    """
    Test case to check for predictions greater than 50
    """
    data = {"age": 52,
            "workclass": "Self-emp-inc",
            "fnlgt": 287927,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Wife",
            "race": "White",
            "sex": "Female",
            "capital_gain": 15024,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
            }

    response = client.post("/predict", data=json.dumps(data))
    assert response.status_code == 200
    assert response.json() == {'pred': '>50K'}


def test_post_empty_data():
    """
    Test case to check empty data
    """
    data = {}

    response = client.post("/predict", data=data)
    assert response.status_code == 422
