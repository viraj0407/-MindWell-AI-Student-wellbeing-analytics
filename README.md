# Student Mental Health Score Predictor

FastAPI app that serves a scikit-learn Random Forest pipeline and a small web UI.

## Run locally
    pip install -r requirements.txt
    uvicorn main:app --reload
Open http://127.0.0.1:8000 (API docs at /docs).

## Notes
- `scikit-learn` is pinned to 1.6.1 because the .pkl was saved with that version.
- The pipeline expects `Group_Country` (not `Country`), which `main.py` builds for you.

## Deploy (Render / Railway)
Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
