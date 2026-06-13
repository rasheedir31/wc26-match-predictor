# WC26 Match Predictor

End-to-end system that predicts **2026 FIFA World Cup** match outcomes and simulates
the tournament - built to demonstrate **ML engineering**, **data engineering**, and
**DevOps** in one coherent, deliberately bounded portfolio project.

The point isn't raw accuracy - it's **correctness and breadth done honestly**: a real
ETL pipeline, four models compared on the same time-based holdout (including a
hand-implemented Dixon-Coles statistical model, not just framework calls), proper
probabilistic evaluation, experiment tracking, drift monitoring, a Monte Carlo
tournament simulation, a live prediction-vs-reality loop, and a one-command local
stack that also auto-deploys.

> **Live demo:** _set your Render URL here_ - it's the Streamlit dashboard. Render's
> free tier sleeps after ~15 min idle, so the **first load cold-starts (~30-60 s)**;
> after that it's snappy.

## What it does

- **Ingests** 49k+ international match results (martj42), FIFA rankings, and the WC26
  draw - with raw caching and an offline seed fallback.
- **Engineers** point-in-time, leak-free features: Elo, recent form, goal differential,
  rest days, home advantage, head-to-head.
- **Compares four models** on the same time-based holdout (see metrics below).
- **Simulates** the tournament by Monte Carlo (≥10k runs, knockout framed as an
  absorbing Markov chain) → per-team stage and championship probabilities.
- **Serves** a FastAPI prediction API + Streamlit dashboard from a self-contained
  snapshot, and **tracks** the pre-tournament forecast against live results.
- **Monitors** feature drift (Evidently + PSI) so you know when to retrain.

## Model comparison (time-based CV, 49,318 matches, 1872-2026)

Out-of-fold metrics on expanding-window time splits - **never shuffled** (shuffling
would leak the future and invalidate everything). Lower is better except accuracy;
RPS is the football-standard *ordered* score.

| Model | Log loss | Brier | RPS | Accuracy |
|---|---|---|---|---|
| **Logistic regression** 🏆 | **0.9033** | **0.5323** | **0.1778** | **58.2%** |
| XGBoost | 0.9061 | 0.5342 | 0.1784 | 57.9% |
| Elo (from scratch) | 0.9224 | 0.5422 | 0.1817 | 57.6% |
| Dixon-Coles Poisson (from scratch) | 0.9776 | 0.5742 | 0.1968 | 54.1% |

The champion is selected automatically by pooled out-of-fold log loss and ships in the
snapshot. (Honest result: the engineered features are linear-friendly, so logistic
edges XGBoost here; Dixon-Coles is a scoreline model judged on a 1X2 task, so it
trails - but it's the "deep understanding" piece, with the log-likelihood, low-score
correction, time-decay weighting, and analytic gradient all hand-written.)

## Architecture

```
data sources (martj42 CSVs, FIFA rankings, WC26 draw)
        │  ingest (cache + offline fallback)
        ▼
   Airflow DAG ───►  Postgres (local warehouse)
   ingest → validate → feature-engineer → load → train → simulate → snapshot → monitor
        │                                   │         │                          │
        │                                   ▼         ▼                          ▼
        │                               MLflow    (4 models, time-based CV)   Evidently
        │                          (experiments)   + Monte Carlo sim          (drift)
        ▼
   export portable snapshot  (SQLite + serialized champion model + featurizer)
        │
        ▼
   App image ──►  FastAPI (prediction API)  +  Streamlit (dashboard + live tracker)
   (self-contained; reads the bundled snapshot; this is the only thing that deploys)
```

The pipeline stages are plain importable functions in `wc26.pipeline`, so the **same
code** runs under `make pipeline`, the Airflow DAG (`dags/wc26_pipeline.py`), and the
GitHub Actions schedule - no duplicated logic.

## Quickstart

Requires **Python 3.12** (auto-fetched by uv), [`uv`](https://docs.astral.sh/uv/), and
GNU Make.

```bash
make setup      # uv sync + pre-commit hooks
make pipeline   # ETL → train → simulate → snapshot → monitor (downloads real data if online)
make app        # FastAPI (http://localhost:8000/docs) + Streamlit (http://localhost:8501)
```

If `raw.githubusercontent.com` is blocked on your network, point ingest at a CDN
mirror (no code change):

```bash
export WC26_SOURCE_RESULTS_URL="https://cdn.jsdelivr.net/gh/martj42/international_results@master/results.csv"
export WC26_SOURCE_SHOOTOUTS_URL="https://cdn.jsdelivr.net/gh/martj42/international_results@master/shootouts.csv"
```

A small **bootstrap snapshot is committed**, so `make app` works on a fresh clone
without running the pipeline first.

### All `make` targets

| | |
|---|---|
| `make setup` | install deps + pre-commit |
| `make lint` / `make format` | ruff check / autofix |
| `make test` | pytest with coverage |
| `make pipeline` | full run: ETL + train + simulate + snapshot + monitor |
| `make train` / `make monitor` | just the modelling / monitoring halves |
| `make snapshot` | rebuild the portable app bundle |
| `make app` / `make api` / `make dashboard` | run the services |
| `make up` / `make down` | full local Docker stack (Postgres, MLflow, Airflow, app) |

### Full local stack

```bash
make up    # Postgres :5432 · MLflow http://localhost:5000 · Airflow http://localhost:8080 · app :8501
make down
```

Only the **app** image (`docker/app.Dockerfile`) ever deploys - Postgres, MLflow, and
Airflow are local-only showcases (see scope guardrails).

## CI/CD

- **`ci.yml`** - on push/PR: lint, test, then build & push the app image to GHCR (main only).
- **`deploy.yml`** - after CI on main: trigger a Render deploy (set the `RENDER_DEPLOY_HOOK` secret).
- **`pipeline.yml`** - scheduled: re-ingest, retrain, re-simulate, regenerate the snapshot,
  refresh the live loop, and commit the updated snapshot so the next deploy serves fresh
  predictions.

## Project layout

`src/wc26/` - `ingest` · `etl` · `features` · `models` (elo, poisson, logistic, gbm) ·
`evaluate` · `simulate` · `tracking` · `monitor` · `api` · `dashboard`; plus
`pipeline.py`, `train.py`, `snapshot.py`. `dags/` (Airflow), `docker/`, `tests/`,
`.github/workflows/`. See `the README` for the authoritative conventions.

## Scope guardrails

Team-level data only (no player/event data). Only the app deploys; Postgres/Airflow
run locally. Docker Compose is the orchestration ceiling - no Kubernetes, no Terraform,
no managed cloud DB. The deployed app reads a bundled snapshot, never a runtime DB.

## License

MIT
