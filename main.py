"""
Webser App Request code
"""
import os
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from ml.data import process_data
from ml.model import inference

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class Data(BaseModel):
    """
    Pyndatic model to be consumed by rest api
    """
    # Using the first row of census.csv as sample
    age: int = Field(None, example=39)
    workclass: str = Field(None, example='State-gov')
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example='Never-married')
    occupation: str = Field(None, example='Adm-clerical')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='White')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=2174)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=40)
    native_country: str = Field(None, example='United-States')


model_object = joblib.load("model/trained_model.joblib")
encoder = joblib.load("model/encoder.joblib")
lb = joblib.load("model/lb.joblib")

app = FastAPI()


@app.get("/")
async def root() -> str:
    """
    Root API endpoint

    Returns
    -------
    str
        A welcome message.
    """
    return "Welcome!"


@app.post('/predict')
async def predict(data: Data):
    """
    API to return prediction from model

    Returns:
        json: Prediction from the model
    """
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

    sample = {key.replace('_', '-'): [value]
              for key, value in data.__dict__.items()}
    input_data = pd.DataFrame.from_dict(sample)
    X, _, _, _ = process_data(
        input_data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    output = inference(model=model_object, X=X)[0]
    str_out = '<=50K' if output == 0 else '>50K'
    return {"pred": str_out}


if __name__ == "__main__":
    config = uvicorn.Config(
        "main:app", host="0.0.0.0", reload=True, port=8080, log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()
