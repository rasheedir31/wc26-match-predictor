# Feature-drift monitoring.
#
# Compares the training feature distribution (reference) against recent matches
# (current) two ways:
#
# 1. A self-computed **Population Stability Index (PSI)** per feature - always
#    available, dependency-light, and the number we surface programmatically
#    (a feature is "drifted" at PSI > 0.2, the usual rule of thumb).
# 2. A rich **Evidently** ``DataDriftPreset`` HTML report saved to ``reports/`` for
#    eyeballing. Best-effort: if the Evidently API shifts, PSI still works.
#
# This is the "is the world still like training?" check that flags when the model
# should be retrained.

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wc26 import schema
from wc26.config import settings

logger = logging.getLogger(__name__)

PSI_DRIFT_THRESHOLD = 0.2
_EPS = 1e-6


def _reports_dir() -> Path:
    d = settings.paths.project_root / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    # PSI between two samples of one feature using quantile bins of the reference.
    #
    # PSI = sum_b (cur_b - ref_b) * ln(cur_b / ref_b) over bin proportions. 0 = identical;
    # >0.1 minor shift; >0.2 significant drift.
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Quantile edges from the reference; widen the ends to catch out-of-range values.
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_prop = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_prop = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_prop = np.clip(ref_prop, _EPS, None)
    cur_prop = np.clip(cur_prop, _EPS, None)
    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def psi_summary(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, object]:
    # PSI for every feature column + an overall drifted-share summary.
    per_feature: dict[str, float] = {}
    for col in schema.FEATURE_COLUMNS:
        if col in reference.columns and col in current.columns:
            per_feature[col] = population_stability_index(
                reference[col].to_numpy(dtype=float), current[col].to_numpy(dtype=float)
            )
    drifted = {c: v for c, v in per_feature.items() if v > PSI_DRIFT_THRESHOLD}
    return {
        "n_features": len(per_feature),
        "n_drifted": len(drifted),
        "share_drifted": (len(drifted) / len(per_feature)) if per_feature else 0.0,
        "psi": per_feature,
        "drifted_features": sorted(drifted),
    }


def _evidently_html(reference: pd.DataFrame, current: pd.DataFrame, path: Path) -> bool:
    # Best-effort Evidently DataDrift HTML report. Returns True on success.
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset

        cols = [c for c in schema.FEATURE_COLUMNS if c in reference.columns]
        data_def = DataDefinition(numerical_columns=cols)
        ref_ds = Dataset.from_pandas(reference[cols], data_definition=data_def)
        cur_ds = Dataset.from_pandas(current[cols], data_definition=data_def)
        result = Report([DataDriftPreset()]).run(reference_data=ref_ds, current_data=cur_ds)
        result.save_html(str(path))
        return True
    except Exception as exc:  # noqa: BLE001 - Evidently API drift shouldn't break monitoring
        logger.warning("Evidently report unavailable (%s); PSI summary still produced", exc)
        return False


def generate_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, object]:
    # Produce the PSI summary (returned + saved JSON) and an Evidently HTML report.
    summary = psi_summary(reference, current)
    reports = _reports_dir()
    (reports / "drift_summary.json").write_text(
        pd.Series(summary).to_json(indent=2)  # nested dict-safe JSON
    )
    summary["evidently_html"] = _evidently_html(reference, current, reports / "drift_report.html")
    logger.info(
        "Drift: %d/%d features drifted (share=%.2f)",
        summary["n_drifted"],
        summary["n_features"],
        summary["share_drifted"],
    )
    return summary
