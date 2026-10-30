"""Replay test transactions through a running API and measure latency.

python scripts/replay.py --n 500            # latency run, as fast as possible
python scripts/replay.py --n 200 --speed 600  # paced replay, 600x real time
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import httpx

from fraud.config import load_config, resolve_path
from fraud.data import load_splits
from fraud.replay import latency_summary, replay
from fraud.report import update_latency

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # per-request logging would skew timings
logger = logging.getLogger("replay")
WARMUP_CALLS = 20  # first calls pay one-off costs (connection, lazy imports); not measured


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=cfg["api"]["latency_samples"])
    parser.add_argument("--speed", type=float, default=0.0, help="0 = as fast as possible")
    parser.add_argument(
        "--write-report", action="store_true", help="save latency.json + results.md"
    )
    args = parser.parse_args()

    test = load_splits()["test"]
    rows = test.iloc[: args.n + WARMUP_CALLS]
    try:
        with httpx.Client(base_url=args.url, timeout=10.0) as client:
            results = list(replay(client, rows, speed=args.speed))
    except httpx.TransportError as exc:
        logger.error("Cannot reach %s (%s). Start it with `make serve`.", args.url, exc)
        return 1

    results = results[WARMUP_CALLS:]
    summary = latency_summary(results)
    flagged = sum(r["is_fraud"] for r in results)
    saved = sum(r["cost_saved"] for r in results)
    logger.info("Latency: %s", json.dumps(summary, indent=2))
    logger.info("Flagged %d of %d, cost saved vs flag-nothing: %.2f", flagged, len(results), saved)
    if args.write_report:
        path = resolve_path("results_md").parent / "latency.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        update_latency(summary)
        logger.info("Wrote %s and updated results.md", path)
    return 0 if summary["meets_target"] else 2


if __name__ == "__main__":
    sys.exit(main())
