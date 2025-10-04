from pydantic import BaseModel

class PlanetData(BaseModel):
    pl_rade: float
    pl_orbper: float
    st_teff: float
