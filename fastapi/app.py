from fastapi import FastAPI
from predict import get_predictions

app=FastAPI()

@app.post('/predict')
async def get_sentiment(given_text: str):
    output_value=get_predictions(given_text)
    return {'toxic_level':output_value}


