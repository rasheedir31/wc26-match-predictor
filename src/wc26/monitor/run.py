# Monitoring entrypoint: drift report + live prediction-vs-actual refresh.
#
# Run after training/snapshot (and on the tournament schedule). Splits the feature
# matrix into an earlier reference block and a recent current block for drift, then
# refreshes the live loop from any newly-ingested results.

from __future__ import annotations

import logging

import pandas as pd

from wc26 import schema
from wc26.config import settings
from wc26.monitor.drift import generate_drift_report
from wc26.monitor.live import update_live_loop

logger = logging.getLogger(__name__)

RECENT_FRACTION = 0.1  # last 10% of matches (by date) form the "current" window


def run_monitor() -> dict[str, object]:
    # Generate the drift report and refresh the live loop; return a small summary.
    feat_path = settings.paths.processed_dir / "features.parquet"
    if not feat_path.exists():
        logger.warning("Monitor: features.parquet missing; run the pipeline first")
        return {}

    feat = pd.read_parquet(feat_path).sort_values(schema.COL_DATE).reset_index(drop=True)
    cut = max(1, int(len(feat) * (1 - RECENT_FRACTION)))
    reference, current = feat.iloc[:cut], feat.iloc[cut:]

    drift = generate_drift_report(reference, current)
    live = update_live_loop()
    logger.info("Monitor complete")
    return {"drift": drift, "live": live}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run_monitor()


if __name__ == "__main__":
    main()
