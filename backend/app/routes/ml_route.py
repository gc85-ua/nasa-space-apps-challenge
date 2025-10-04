from fastapi import APIRouter, Query, Response

import joblib,json
import numpy as np


model = joblib.load("./routes/random_forest_model.joblib")

router = APIRouter()

@router.post("/predict")
async def predict(body: dict):
    orbit_period_days = body.get("orbit_period_days")
    stellar_radius_sun = body.get("stellar_radius_sun")
    stellar_effective_temp_kelvin = body.get("stellar_effective_temp_kelvin")
    planet_equilibrium_temp_kelvin = body.get("planet_equilibrium_temp_kelvin")
    stellar_surface_gravity_log10_cm_s2 = body.get("stellar_surface_gravity_log10_cm_s2")
    if None in [orbit_period_days, stellar_radius_sun, stellar_effective_temp_kelvin, planet_equilibrium_temp_kelvin, stellar_surface_gravity_log10_cm_s2]:
        return Response(content=json.dumps({"error": "Missing one or more required features in the request body."}), status_code=400)
    
    features = [[orbit_period_days, stellar_radius_sun, stellar_effective_temp_kelvin, planet_equilibrium_temp_kelvin, stellar_surface_gravity_log10_cm_s2]]
    features = np.array(features)
    try:
        prediction = model.predict(features)
        return Response(content=json.dumps({"prediction": "CANDIDATE" if prediction[0]==1 else "NOT A CANDIDATE"}), status_code=200)
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}), status_code=500)

    """
    curl example:
    curl -X 'POST' 'http://localhost:8000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"orbit_period_days": 10, "stellar_radius_sun": 1, "stellar_effective_temp_kelvin": 5000,"planet_equilibrium_temp_kelvin": 300,"stellar_surface_gravity_log10_cm_s2": 4.5}'
    """