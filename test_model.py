"""
File to test the model
"""

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
import joblib
from ml.data import process_data
from ml.model import train_model

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


@pytest.fixture(name='cleaned_data')
def cleaned_data():
    """
    This is a fixture for loading cleaned data and will be used by other tests.

    Yields:
        pd.Dataframe : Cleaned data 
    """
    yield pd.read_csv('data/cleaned_census_income.csv')


def test_model():
    """
    Test case to check if the correct model is loaded
    """
    model = joblib.load("model/trained_model.joblib")
    assert isinstance(model, RandomForestClassifier)


def test_cleaned_data(cleaned_data):
    """
    Test case to check if the cleaned data is loaded properly

    Args:
        cleaned_data (pd.Dataframe): Cleaned Data from the fixture
    """
    assert cleaned_data.shape[0] > 0 and cleaned_data.shape[1] > 0


def test_ml_training(cleaned_data):
    """
    Test case to check after the cleaned data is loaded
    model is trained propely or not.

    Args:
        cleaned_data (pd.Dataframe): Cleaned Data from the fixture
    """
    X_train, y_train, encoder, lb = process_data(
        cleaned_data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X_train, y_train)
    assert model is not None
    assert encoder is not None
    assert lb is not None
