

from typing import Union
from fastapi import FastAPI
import dataPipeline  as dp
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/summary") 
def summary(): 
    dp.splitter()
    res  = dp.classify()
    dp.upload_res()
    return   res

@app.post("/upload_result_to_postgres")
def upload_to_pq(): 
    return  dp.upload_res()

