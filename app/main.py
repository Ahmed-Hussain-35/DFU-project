from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from .models import init_db
from .routers import auth_router, visits_router

app = FastAPI(title="DFU Clinical Platform")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8010,http://127.0.0.1:8010").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])

app.include_router(auth_router.router)
app.include_router(visits_router.router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
