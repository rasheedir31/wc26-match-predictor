# Launch the API and dashboard together (the ``make app`` entrypoint).
#
# Spawns uvicorn (FastAPI) and Streamlit as child processes and waits on both, so a
# single command brings up the whole app locally and in the Docker image. Either
# process exiting tears down the other.

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from types import FrameType

from wc26.config import settings

_DASHBOARD = Path(__file__).resolve().parent / "dashboard" / "app.py"


def _spawn() -> list[subprocess.Popen[bytes]]:
    # On hosts like Render the public port is $PORT - bind the dashboard to it.
    dashboard_port = int(os.environ.get("PORT", settings.dashboard_port))
    api = [
        sys.executable,
        "-m",
        "uvicorn",
        "wc26.api.app:app",
        "--host",
        settings.api_host,
        "--port",
        str(settings.api_port),
    ]
    dashboard = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_DASHBOARD),
        "--server.port",
        str(dashboard_port),
        "--server.address",
        settings.api_host,
        "--server.headless",
        "true",
    ]
    return [subprocess.Popen(api), subprocess.Popen(dashboard)]


def main() -> None:
    procs = _spawn()
    print(
        f"API:       http://localhost:{settings.api_port}  (docs at /docs)\n"
        f"Dashboard: http://localhost:{settings.dashboard_port}",
        flush=True,
    )

    def _shutdown(_sig: int, _frame: FrameType | None) -> None:
        for p in procs:
            p.terminate()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # If either process exits, stop the other and propagate the exit code.
    while True:
        for p in procs:
            code = p.poll()
            if code is not None:
                for other in procs:
                    if other is not p:
                        other.terminate()
                sys.exit(code)
        try:
            procs[0].wait(timeout=1)
        except subprocess.TimeoutExpired:
            continue


if __name__ == "__main__":
    main()
