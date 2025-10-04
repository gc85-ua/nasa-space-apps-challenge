from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/models")
async def list_models():
    return ["randomforest", "xboost"]
