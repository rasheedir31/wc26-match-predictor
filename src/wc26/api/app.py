# FastAPI prediction API.
#
# Self-contained: reads only the bundled snapshot (no external DB). Endpoints:
#
# - ``GET  /health``        liveness
# - ``GET  /teams``         the 48 WC26 teams
# - ``POST /predict``       1X2 probabilities for an arbitrary fixture
# - ``GET  /odds``          tournament odds (per-team stage + championship)
# - ``GET  /odds/{team}``   one team's odds
# - ``GET  /fixtures``      per-fixture group-stage predictions
# - ``GET  /models``        model-comparison metrics + champion name
# - ``GET  /live``          running prediction-vs-actual accuracy / log loss

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from wc26.snapshot import SnapshotStore, get_store

app = FastAPI(
    title="WC26 Match Predictor API",
    version="0.1.0",
    summary="1X2 predictions and 2026 World Cup tournament odds.",
)


def store_dep() -> SnapshotStore:
    # Dependency: the snapshot store, or 503 if no snapshot has been built yet.
    try:
        return get_store()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"Snapshot not available ({exc}). Run `make pipeline` then `make snapshot`.",
        ) from exc


class PredictRequest(BaseModel):
    home_team: str = Field(..., examples=["Argentina"])
    away_team: str = Field(..., examples=["Brazil"])
    neutral: bool = Field(True, description="True for a neutral venue (most WC matches)")


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    neutral: bool
    p_home: float
    p_draw: float
    p_away: float
    most_likely: str
    model: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/teams")
def teams(store: SnapshotStore = Depends(store_dep)) -> dict[str, Any]:
    return {"teams": store.teams()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, store: SnapshotStore = Depends(store_dep)) -> PredictResponse:
    if req.home_team == req.away_team:
        raise HTTPException(status_code=422, detail="home_team and away_team must differ")
    result = store.predict(req.home_team, req.away_team, neutral=req.neutral)
    return PredictResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        neutral=req.neutral,
        model=store.champion_name,
        **result,  # type: ignore[arg-type]
    )


@app.get("/odds")
def odds(
    top: int = Query(48, ge=1, le=48, description="Return the top-N teams by title odds"),
    store: SnapshotStore = Depends(store_dep),
) -> dict[str, Any]:
    df = store.tournament_odds().head(top)
    return {"odds": df.to_dict(orient="records")}


@app.get("/odds/{team}")
def team_odds(team: str, store: SnapshotStore = Depends(store_dep)) -> dict[str, Any]:
    df = store.tournament_odds()
    row = df[df["team"].str.lower() == team.lower()]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Unknown team: {team}")
    return row.iloc[0].to_dict()


@app.get("/fixtures")
def fixtures(store: SnapshotStore = Depends(store_dep)) -> dict[str, Any]:
    return {"fixtures": store.group_predictions().to_dict(orient="records")}


@app.get("/models")
def models(store: SnapshotStore = Depends(store_dep)) -> dict[str, Any]:
    return {
        "champion": store.champion_name,
        "comparison": store.model_comparison().to_dict(orient="records"),
    }


@app.get("/live")
def live(store: SnapshotStore = Depends(store_dep)) -> dict[str, Any]:
    return store.live_summary()
