from fastapi import APIRouter, Query, Response, Body
from pydantic import BaseModel

import joblib,json
import numpy as np


model = joblib.load("./routes/random_forest_model.joblib")

router = APIRouter()

class PredictionRequest(BaseModel):
    orbit_period_days: float
    stellar_radius_sun: float
    stellar_effective_temp_kelvin: float
    planet_equilibrium_temp_kelvin: float
    stellar_surface_gravity_log10_cm_s2: float

@router.post("/predict",
             summary="Predicts exoplanet candidates",
             description="Predicts whether a planet is an exoplanet candidate using the trained Random Forest model. The input features are: orbit_period_days, stellar_radius_sun, stellar_effective_temp_kelvin, planet_equilibrium_temp_kelvin, stellar_surface_gravity_log10_cm_s2.",
             response_description="Prediction result indicating whether the planet is a CANDIDATE or NOT A CANDIDATE.",
             )
async def predict(body: PredictionRequest):
    orbit_period_days = body.orbit_period_days
    stellar_radius_sun = body.stellar_radius_sun
    stellar_effective_temp_kelvin = body.stellar_effective_temp_kelvin
    planet_equilibrium_temp_kelvin = body.planet_equilibrium_temp_kelvin
    stellar_surface_gravity_log10_cm_s2 = body.stellar_surface_gravity_log10_cm_s2
    if None in [orbit_period_days, stellar_radius_sun, stellar_effective_temp_kelvin, planet_equilibrium_temp_kelvin, stellar_surface_gravity_log10_cm_s2]:
        return Response(content=json.dumps({"error": "Missing one or more required features in the request body."}), status_code=400, media_type="application/json")
    
    features = [[orbit_period_days, stellar_radius_sun, stellar_effective_temp_kelvin, planet_equilibrium_temp_kelvin, stellar_surface_gravity_log10_cm_s2]]
    features = np.array(features)
    try:
        prediction = model.predict(features)
        return Response(content=json.dumps({"prediction": "CANDIDATE" if prediction[0]==1 else "NOT A CANDIDATE"}), status_code=200, media_type="application/json")
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}), status_code=500, media_type="application/json")

    """
    curl example:
    curl -X 'POST' 'http://localhost:8000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"orbit_period_days": 10, "stellar_radius_sun": 1, "stellar_effective_temp_kelvin": 5000,"planet_equilibrium_temp_kelvin": 300,"stellar_surface_gravity_log10_cm_s2": 4.5}'
    """