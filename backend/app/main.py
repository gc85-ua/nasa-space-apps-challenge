from fastapi import FastAPI
from app.routes.ml_route import router as ml_router
app = FastAPI()
app.include_router(ml_router)
@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/models")
async def list_models():
    return ["randomforest", "xboost"]
