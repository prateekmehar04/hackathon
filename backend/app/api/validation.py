"""
GET /api/validation  —  returns the saved validation results from training.
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
import json

router = APIRouter(tags=["Validation"])

VALIDATION_PATH = Path(__file__).parent.parent / "models" / "validation_results.json"


@router.get("/validation")
def get_validation():
    """Return per-class F1, confusion matrix, and patient-split results from training."""
    if not VALIDATION_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No validation results found. Run python train.py first."
        )
    return json.loads(VALIDATION_PATH.read_text())
