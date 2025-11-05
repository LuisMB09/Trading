# app.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import mlflow
from fastapi import File, UploadFile
import numpy as np
import io
from PIL import Image
from pydantic import BaseModel


app = FastAPI()

class FinancialData(BaseModel):
    open: float
    high: float
    low: float
    close: float

labels = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

model_name = 'cifar_net'
model_version = 'latest'

model = mlflow.tensorflow.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

@app.get("/")
def hello():
    return {"message": "Hello, World!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        image = image.resize((32, 32))
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        predict_idx = int(np.argmax(prediction, axis=1)[0])
        return {
            "predicted_idx": predict_idx,
            "label": labels[predict_idx]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/financial-predict")
def fin_predict(data: FinancialData):
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=9999)
    