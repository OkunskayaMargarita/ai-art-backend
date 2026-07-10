from fastapi import FastAPI
from api.generation import router as generation_router

app = FastAPI()

app.include_router(generation_router)


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "AI Art Backend is running"
    }