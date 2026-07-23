from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.generation import router as generation_router
from api.profiles import router as profiles_router
from api.models import router as models_router
from api.characters import router as characters_router
from config.settings import OUTPUT_DIR

app = FastAPI(
    title="AI Art Backend",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generation_router)
app.include_router(profiles_router)
app.include_router(models_router)
app.include_router(characters_router)

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs",
)


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "AI Art Backend is running",
    }