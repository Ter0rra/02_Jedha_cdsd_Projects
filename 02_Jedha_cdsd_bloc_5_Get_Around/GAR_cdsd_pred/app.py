import uvicorn
import pandas as pd 
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import joblib

description = """
Welcome to car price API. This app is made for you to understand how FastAPI works! Try it out 🕹️

## Endpoints summary

* `/preview` a few rows of your dataset
* `/predict`: **POST** request that display a car price prediction

## Preview

This is a preview endpoint where you can see a few rows of your dataset

* `/preview` a few rows of your dataset


## Machine Learning

This is a Machine Learning endpoint that predict rental car price per day from different variables given.

* `/predict` to predict rental car price per day with multiples variables input


Check out documentation below 👇 for more information on each endpoint. 
"""

tags_metadata = [
    {
        "name": "Preview",
        "description": "Simple endpoints to try out!",
    },
    {
        "name": "Machine Learning",
        "description": """Prediction of rental car price per day with **POST** request on multiple variables."""
    }
]

app = FastAPI(
    title=" Car prices API",
    description=description,
    version="0.1",
    openapi_tags=tags_metadata
)

# Read data 
df = pd.DataFrame({"some": ["data"]})

# Log model from mlflow 
loaded_model = joblib.load('/home/user/app/modele_GAR.joblib') #lien vers le fichier job lib dans le conteneur

class PredictionFeatures(BaseModel):
    model_key: object
    mileage: float
    engine_power: float
    fuel: object
    paint_color: object
    car_type: object
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

#### SOME CODE ####
###################

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Car Rental Price Prediction API! 🚗",
        "status": "running",
        "model_loaded": loaded_model is not None,
        "endpoints": {
            "documentation": "/docs",
            "preview": "/preview?rows=10",
            "predict": "/predict",
            "health": "/health"
        },
        "version": "1.0.0"
    }

@app.get("/preview", tags=["Preview"])
async def random_employees(rows: int = 10):
    """
    Get a sample of your whole dataset.
    You can specify how many rows you want by specifying a value for `rows`, default is `10`
    """
    df = pd.read_csv("DATA/get_around_pricing_project.csv")
    df = df.drop('Unnamed: 0', axis=1)
    sample = df.sample(rows)
    return sample.to_dict(orient='records')



@app.post("/predict", tags=["Machine Learning"])
async def predict(predictionFeatures: PredictionFeatures):
    """
    To predict rental car price per day with multiple variables :
    - Click on 'Try it out'
    - Change variables (Please respect data type, see 'Variables & data type', an example is provided)
    - Click on 'Execute' 
    - You will see down below a prediction of price base on your values.

        
    Variables & data type:
    - model_key                    (object)
    - mileage                       (int64)
    - engine_power                  (int64)
    - fuel                         (object)
    - paint_color                  (object)
    - car_type                     (object)
    - private_parking_available      (bool)
    - has_gps                        (bool)
    - has_air_conditioning           (bool)
    - automatic_car                  (bool)
    - has_getaround_connect          (bool)
    - has_speed_regulator            (bool)
    - winter_tires                   (bool)

    Prediction is rental_price_per_day (int64)

    Exemple to use:

    {
    "model_key": " Renault",
    "mileage": 109839,
    "engine_power": 135,
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "sedan",
    "private_parking_available": true,
    "has_gps": true,
    "has_air_conditioning": false,
    "automatic_car": false,
    "has_getaround_connect": true,
    "has_speed_regulator": false,
    "winter_tires": true
    }

    rental_price_per_day :   152 (real)    //    137.34634 (potential prediction)
    """
    # Read data 
    variables = pd.DataFrame({"model_key": [predictionFeatures.model_key],
                              "mileage": [predictionFeatures.mileage],
                              "engine_power": [predictionFeatures.engine_power],
                              "fuel": [predictionFeatures.fuel],
                              "paint_color": [predictionFeatures.paint_color],
                              "car_type": [predictionFeatures.car_type],
                              "private_parking_available": [predictionFeatures.private_parking_available],
                              "has_gps": [predictionFeatures.has_gps],
                              "has_air_conditioning": [predictionFeatures.has_air_conditioning],
                              "automatic_car": [predictionFeatures.automatic_car],
                              "has_getaround_connect": [predictionFeatures.has_getaround_connect],
                              "has_speed_regulator": [predictionFeatures.has_speed_regulator],
                              "winter_tires": [predictionFeatures.winter_tires]})

    # Load model as a PyFuncModel.
    prediction = loaded_model.predict(variables)

    # Format response
    response = {"prediction": prediction.tolist()[0]}
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)