from typing import Dict
from fastapi import FastAPI
from pydantic import BaseModel

from model import Model

app = FastAPI()
model = None


@app.on_event('startup')
def startup_event():
    global model
    model = Model()


class SentimentRequest(BaseModel):
    texts: list[str]


class SentimentResponse(BaseModel):
    sentiments: list[str]
    confidences: list[float]
    probabilities: Dict[str, list[float]]


@app.post('/predict', response_model=SentimentResponse)
def predict(request: SentimentRequest):
    response = model.predict(request.texts)
    return SentimentResponse(
        sentiments=response['sentiments'],
        confidences=response['confidences'],
        probabilities=response['probabilities']
    )
