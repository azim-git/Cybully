import sys
import os

# make src/ importable from api/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from predict import predict


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cyberbullying Classifier API",
    description="Classifies social media posts as normal, offensive, or hatespeech.",
    version="1.0.0"
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# allows the React frontend (localhost:3000) to call this API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Available models ──────────────────────────────────────────────────────────
# maps display name → directory path
AVAILABLE_MODELS = {
    "Baseline":      "models/baseline",
    "Class Weights": "models/v2_class_weights",
    "Lower Learning Rate": "models/v3_lower_lr",
    "Best Model":    "models/v4_more_epochs",
}

# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text:       str
    model_name: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("text must not be empty")
        return v.strip()

    @field_validator("model_name")
    @classmethod
    def model_must_be_valid(cls, v):
        valid = list(AVAILABLE_MODELS.keys())
        if v not in valid:
            raise ValueError(f"model_name must be one of {valid}")
        return v


class ScoresSchema(BaseModel):
    normal:     float
    offensive:  float
    hatespeech: float


class PredictResponse(BaseModel):
    label:      str
    confidence: float
    scores:     ScoresSchema
    model_used: str


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIST_DIR   = os.path.join(ROOT_DIR, "frontend", "dist")
INDEX_HTML = os.path.join(DIST_DIR, "index.html")

# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/models")
def get_models():
    """Return list of available models for the frontend dropdown."""
    return {"models": list(AVAILABLE_MODELS.keys())}


@app.post("/predict", response_model=PredictResponse)
def run_predict(request: PredictRequest):
    """
    Classify a social media post.

    Body:
        text:       the post to classify
        model_name: one of the available model display names
    """
    model_dir = AVAILABLE_MODELS[request.model_name]

    try:
        result = predict(request.text, model_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(
        label=result["label"],
        confidence=result["confidence"],
        scores=ScoresSchema(**result["scores"]),
        model_used=request.model_name
    )

# serve React static files
app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets")

@app.get("/{full_path:path}")
def serve_react(full_path: str):
    return FileResponse(INDEX_HTML)