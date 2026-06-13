# Cached HTTP download with a graceful offline fallback.
#
# The pipeline's *production* behaviour is to download authoritative sources and
# cache them under ``data/raw/`` (gitignored, reproducible). When there is no
# network (offline dev, sandboxed CI without egress), callers fall back to the small
# committed seed data so the whole pipeline still runs end to end.

from __future__ import annotations

import logging
from pathlib import Path

import requests

from wc26.config import settings

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30


def download_to_cache(
    url: str,
    filename: str,
    *,
    force: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> Path | None:
    # Download ``url`` into ``data/raw/<filename>`` and return the path.
    #
    # Caching: if the file already exists and ``force`` is False, the cached copy is
    # returned without a network call. On any network/HTTP error, logs a warning and
    # returns ``None`` so the caller can fall back to seed data.
    if not url:
        return None

    settings.paths.ensure()
    dest = settings.paths.raw_dir / filename

    if dest.exists() and not force:
        logger.info("Using cached %s", dest)
        return dest

    try:
        logger.info("Downloading %s -> %s", url, dest)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:  # network down, DNS, HTTP error
        logger.warning("Download failed for %s (%s); will use fallback if available", url, exc)
        return None

    dest.write_bytes(resp.content)
    return dest
