from fastapi import APIRouter, Query

import joblib


model = joblib.load("./routes/random_forest_model.joblib")

router = APIRouter()

@router.get("/predict")
async def predict(
    feature1: float = Query(..., description="Feature 1"),
    feature2: float = Query(..., description="Feature 2"),
    feature3: float = Query(..., description="Feature 3"),
):
    features = [[feature1, feature2, feature3]]
    try:
        prediction = model.predict(features)
        return {"prediction": prediction[0]}
    except Exception as e:
        return {"error": str(e)}

"""
get request example:
http://localhost:8000/predict?feature1=1.0&feature2=2.0&feature3=3.0
"""