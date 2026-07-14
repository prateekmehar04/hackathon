"""
FastAPI entry point for the ECG Arrhythmia Classification service.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import classify, samples, health, validation

app = FastAPI(
    title="ECG Arrhythmia Classifier",
    description="Classifies cardiac arrhythmias from ECG signals with SHAP-based explanations.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(classify.router, prefix="/api")
app.include_router(samples.router, prefix="/api")
app.include_router(validation.router, prefix="/api")
