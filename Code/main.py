

from typing import Union
from fastapi import FastAPI
import dataPipeline  as dp
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/classify") 
def classify(): 
    return  { "Emotions" : dp.classify()}

# @app.get('/loadModel')
# def load_model():
#     return dp.load_model()

