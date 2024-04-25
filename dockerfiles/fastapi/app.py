import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks,Query
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated

def load_model(model_name: str, alias: str):
    """
    Load a trained model and associated data dictionary.

    This function attempts to load a trained model specified by its name and alias. If the model is not found in the
    MLflow registry, it loads the default model from a file. Additionally, it loads information about the ETL pipeline
    from an S3 bucket. If the data dictionary is not found in the S3 bucket, it loads it from a local file.

    :param model_name: The name of the model.
    :param alias: The alias of the model version.
    :return: A tuple containing the loaded model, its version, and the data dictionary.
    """

    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        # If there is no registry in MLflow, open the default model
        file_ml = open('/app/files/model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open('/app/files/data.json', 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()

    return model_ml, version_model_ml, data_dictionary


def check_model():
    """
    Check for updates in the model and update if necessary.

    The function checks the model registry to see if the version of the champion model has changed. If the version
    has changed, it updates the model and the data dictionary accordingly.

    :return: None
    """

    global model
    global data_dict
    global version_model

    try:
        model_name = "mnist_784"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        # Check in the model registry if the version of the champion has changed
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        # If the versions are not the same
        if new_version_model != version_model:
            # Load the new model and update version and data dictionary
            model, version_model, data_dict = load_model(model_name, alias)

    except:
        # If an error occurs during the process, pass silently
        pass


class ModelInput(BaseModel):

    img: list = Field(
        description="Imágen MNIST 28x28 pixeles",
        img=[]
    )


class Item(BaseModel):
    img: list = []


class ModelOutput(BaseModel):


    int_output: int = Field(
        description="Output of the model. Number from 0 to 9",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": 0,
                }
            ]
        }
    }


# Load the model before start
model, version_model, data_dict = load_model("mnist_784", "champion")

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Root endpoint of the mnist_784 model API.
    """
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the TP AdM II Carlos Rodriguez"}))


@app.get("/print_image/")
def read_img(item: Item):
    return JSONResponse(content=jsonable_encoder(item.img))

@app.post("/predict_image/")
def predict(img: Item):
    """
    Endpoint for predicting digit drawn in image.
    """
    
    prediction = model.predict(np.array(img.img).reshape(1, -1))[0]

    return JSONResponse(content=jsonable_encoder(prediction))


@app.post("/predict/", response_model=ModelOutput)
def predict(
    input_img_array: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    """
    Endpoint for predicting heart disease.

    This endpoint receives features related to a patient's health and predicts whether the patient has heart disease
    or not using a trained model. It returns the prediction result in both integer and string formats.
    """
    print('input_img_array')
    print(input_img_array)
    
    prediction = model.predict(np.array(input_img_array).reshape(1, -1))[0]

    # Check if the model has changed asynchronously
    background_tasks.add_task(check_model)

    # Return the prediction result
    return ModelOutput(int_output=bool(prediction))
