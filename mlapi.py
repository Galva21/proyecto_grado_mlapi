# obtencion del servidor machine learning
from lib2to3.pytree import Base
from pyexpat import model
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import joblib
import pandas as pd

app = FastAPI()

class ScoringItem(BaseModel):
    tamaniogrande:int#"3",
    espacioDisponible:int #"1",
    estadoEconomico:int#"3",
    tiempoDisponible:int#"3"

with open('model.pkl','rb') as f:
    model = joblib.load(f)

@app.post('/')
async def scoring_endpoint(item:ScoringItem):
    print("VALUES: ",item.dict().values())
    print("KEYS: ",item.dict().keys())
    df = pd.DataFrame([item.dict().values()], columns = item.dict().keys())
    yhat = model.predict(df)
    return {"prediction":int(yhat)}