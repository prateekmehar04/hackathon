"""
/api/samples — list and fetch built-in demo ECG records.
"""
from fastapi import APIRouter, HTTPException

from app.core.samples import list_samples, load_sample

router = APIRouter(tags=["Samples"])


@router.get("/samples")
def get_samples():
    """Return list of available demo records."""
    return list_samples()


@router.get("/sample/{record_id}")
def get_sample(record_id: str):
    """Fetch a single demo ECG record (downloads from PhysioNet on first call)."""
    try:
        return load_sample(record_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load record: {exc}")
