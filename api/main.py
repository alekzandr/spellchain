from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.models import DeckRequest, ScoringResponse, MetricResult, StandardResponse
from api.engine_wrapper import MNSSWrapper
from api.meta_loader import MetaLoader
import os
import json

app = FastAPI(title="Spellchain API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine and meta loader
CARD_DB_PATH = os.path.join("data", "standard_cards.json")
META_DIR = "Meta"
engine = None
meta_loader = None

@app.on_event("startup")
async def startup_event():
    global engine, meta_loader
    if os.path.exists(CARD_DB_PATH):
        engine = MNSSWrapper(CARD_DB_PATH)
        meta_loader = MetaLoader(META_DIR, CARD_DB_PATH)
    else:
        print(f"Warning: Card database not found at {CARD_DB_PATH}")

@app.get("/")
async def root():
    return {"message": "Spellchain API is running"}

@app.post("/score", response_model=ScoringResponse)
async def score_deck(request: DeckRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized. Card database missing?")
    
    # For now, we'll use a dummy/empty baseline if one isn't provided,
    # or handle self-normalization.
    # In a real app, we'd load a persistent baseline for Standard.
    
    # Placeholder baseline (mu=0, sigma=1)
    # Ideally, we should have a pre-computed baseline for Standard.
    from mnss_v2 import RAW_METRICS
    dummy_baseline = {k: {"mu": 0.0, "sigma": 1.0} for k in RAW_METRICS.keys()}
    
    try:
        results = engine.score_deck(request.decklist, dummy_baseline)
        return ScoringResponse(
            deck_name="Uploaded Deck",
            results=MetricResult(
                raw=results['raw'],
                unit=results['unit'],
                mnss=results['mnss']
            ),
            cards=results['cards']
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "db_loaded": engine is not None}

# ---------- Meta API Endpoints ----------

@app.get("/meta/decks")
async def get_meta_decks():
    """Get all analyzed Meta decklists with MNSS scores."""
    if meta_loader is None:
        raise HTTPException(status_code=500, detail="Meta loader not initialized.")
    
    try:
        decks = meta_loader.analyze_all_decks()
        return {"decks": decks, "count": len(decks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/meta/baseline")
async def get_meta_baseline():
    """Get the baseline computed from Meta decklists."""
    if meta_loader is None:
        raise HTTPException(status_code=500, detail="Meta loader not initialized.")
    
    try:
        baseline = meta_loader.build_baseline()
        return {"baseline": baseline}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/meta/summary")
async def get_meta_summary():
    """Get a summary of the current meta."""
    if meta_loader is None:
        raise HTTPException(status_code=500, detail="Meta loader not initialized.")
    
    try:
        summary = meta_loader.get_meta_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/meta/refresh")
async def refresh_meta():
    """Reload Meta folder and recompute analysis."""
    if meta_loader is None:
        raise HTTPException(status_code=500, detail="Meta loader not initialized.")
    
    try:
        result = meta_loader.refresh()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score/with-meta-baseline", response_model=ScoringResponse)
async def score_deck_with_meta_baseline(request: DeckRequest):
    """Score a deck using the Meta baseline for normalization."""
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized. Card database missing?")
    if meta_loader is None:
        raise HTTPException(status_code=500, detail="Meta loader not initialized.")
    
    try:
        baseline = meta_loader.build_baseline()
        results = engine.score_deck(request.decklist, baseline)
        return ScoringResponse(
            deck_name="Uploaded Deck",
            results=MetricResult(
                raw=results['raw'],
                unit=results['unit'],
                mnss=results['mnss']
            ),
            cards=results['cards']
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

