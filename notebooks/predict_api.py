'''
File: predict.py
Project: src
File Created: Monday, 17th August 2020 2:32:13 pm
Author: Sparsh Dutta (sparsh.dtt@gmail.com)
-----
Last Modified: Tuesday, 18th August 2020 11:57:09 pm
Modified By: Sparsh Dutta (sparsh.dtt@gmail.com>)
-----
Copyright 2020 Sparsh Dutta
'''
import sys
sys.path.append('..')

from letcon.src.utils.utils import load_artifacts
from letcon.src.logger.logger import logger
from pydantic import BaseModel
from fastapi import FastAPI

import uvicorn

# Initializing Logger
logger_function = logger(name='predict_api')

# Loading saved data and model object from pickle file
data_pipeline, trained_model = load_artifacts(data_filename='data_object', 
                                              model_filename='model_object')

# Initializing the fastapi client
app = FastAPI(debug=True)

# Using Pydantic BaseModel class for automatic data validation
class Data(BaseModel):
    pH: float
    chlorides: float
    volatile_acidity: float
    citric_acid: float
    alcohol: float
    total_sulfur_dioxide: float
    density: float
    residual_sugar: float
    fixed_acidity: float
    sulphates : float
    free_sulfur_dioxide : float


@app.post("/predict")
def predict(data: Data):
    try:
        processed_data = data_pipeline.do_preprocessing(x_data=data.dict())
        prediction = trained_model.predict(x_data=processed_data)
        return {"prediction": float(prediction[0])}   
    except:
        logger_function.error('Critical Error Occurred...')
        return {"prediction" : "error"}
        
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)