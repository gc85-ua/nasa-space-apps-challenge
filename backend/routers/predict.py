from fastapi import APIRouter
from pydantic import BaseModel
from services.model_service import predict_planet

router = APIRouter()

# Schema para recibir datos del frontend
class PlanetData(BaseModel):
    pl_rade: float
    pl_orbper: float
    st_teff: float

@router.post("/predict")
def predict(data: PlanetData):
    # Llama a la función del servicio que tiene el modelo
    return predict_planet(data.pl_rade, data.pl_orbper, data.st_teff)
