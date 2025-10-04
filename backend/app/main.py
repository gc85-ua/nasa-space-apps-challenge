from fastapi import FastAPI
from app.routes.ml_route import router as ml_router
# allow CORS for all origins (for development purposes)
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
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
    return {"Hello": "World"}

@app.get("/models")
async def list_models():
    return ["randomforest", "xboost"]
