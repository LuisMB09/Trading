# app.py
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def hello():
    return {"message": "Hello, World!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=9999)
    