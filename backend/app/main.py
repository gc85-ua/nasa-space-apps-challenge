from fastapi import FastAPI, Response
from app.routes.ml_route import router as ml_router
# allow CORS for all origins (for development purposes)
from fastapi.middleware.cors import CORSMiddleware
import json
app = FastAPI(
    title="Exoplanet Prediction API",
    description="API for predicting exoplanets using a trained Random Forest model.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ml_router)
@app.get("/")
async def read_root():
    return Response(content=json.dumps({"message": "Welcome to the Exoplanet prediction API"},ensure_ascii=False), media_type="application/json", status_code=200)

@app.get("/models")
async def list_models():
    with open("random_forest_model_metrics.json", "r") as f:
        metrics = json.load(f)
    model_data = {
        "model_name": "Random Forest Classifier",
        "metrics": metrics,
        "endpoint": {
            "uri": "/predict",
            "method": "POST"
        }
    }
    return Response(content=json.dumps(model_data, ensure_ascii=False), media_type="application/json", status_code=200)
