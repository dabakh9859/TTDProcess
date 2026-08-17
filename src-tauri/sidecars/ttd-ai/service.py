"""TTDProcess AI sidecar — JSON-RPC service over stdin/stdout.

Started as a child process by the Rust app. Speaks newline-delimited JSON:
  - Requests come in on stdin, one per line.
  - Responses and progress events go out on stdout, one per line.
  - Errors and tracebacks go to stderr (visible to the parent for debugging).

Protocol (one JSON object per line):

  Request   {"id": <int>, "method": <str>, "params": {...}}
  Response  {"id": <int>, "result": {...}}
  Error     {"id": <int>, "error": {"code": <str>, "message": <str>, "trace": <str>}}
  Event     {"event": <str>, "id": <int>, "data": {...}}      # mid-call progress

Methods exposed:
  - health()                         — sanity check + GPU/torch info
  - train(spec)                      — train a SAITS model, stream progress
  - predict(spec)                    — impute a gap, return reconstructed series
  - model_save(model_id, path)       — export to .ttdmodel zip
  - model_load(path) -> model_id     — import from .ttdmodel zip

This file is the *only* entry point. All heavy logic lives in the `ai/`
package next to it.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# ─── stdout hygiene ────────────────────────────────────────────────────────
# Third-party libs (pypots banner, openpyxl warnings, etc.) print to stdout
# at import time. That would corrupt our JSON-RPC channel. We dup the real
# stdout to a private fd before any heavy import, then redirect fd 1 to fd 2
# so anything else that hits "stdout" actually goes to stderr (where the
# Rust parent can still read it for debugging).
_RAW_STDOUT = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1, newline="")
os.dup2(2, 1)
sys.stdout = sys.stderr

sys.path.insert(0, str(Path(__file__).parent))

# Windows DLL ordering: pandas/openpyxl/pyarrow MUST be imported before
# torch (which `health()` will load). Otherwise torch's MKL/OpenMP DLLs
# preempt the loader and pandas' later operations (concat, etc.) silently
# hang. Doing it here at startup, before any handler can be called,
# ensures the order is right regardless of which RPC method comes first.
import pandas as _pd_warmup  # noqa: F401, E402
import openpyxl as _op_warmup  # noqa: F401, E402

from ai import handlers  # noqa: E402


def write_msg(obj: dict[str, Any]) -> None:
    """Emit one JSON line on the *real* stdout and flush. The Rust parent
    reads line-by-line and would block forever without the flush."""
    _RAW_STDOUT.write(json.dumps(obj, separators=(",", ":")) + "\n")
    _RAW_STDOUT.flush()


def emit_event(req_id: int, event: str, data: dict[str, Any]) -> None:
    """Used by long-running handlers to stream progress to the parent."""
    write_msg({"event": event, "id": req_id, "data": data})


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            write_msg({
                "id": None,
                "error": {"code": "BAD_JSON", "message": str(exc)},
            })
            continue

        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        handler = getattr(handlers, method, None)
        if handler is None:
            write_msg({
                "id": req_id,
                "error": {"code": "UNKNOWN_METHOD", "message": f"method '{method}' not found"},
            })
            continue

        # Each handler may stream progress via the `emit` callable we inject.
        def emit(event: str, data: dict[str, Any]) -> None:
            emit_event(req_id, event, data)

        try:
            result = handler(params, emit)
            write_msg({"id": req_id, "result": result})
        except Exception as exc:  # noqa: BLE001
            write_msg({
                "id": req_id,
                "error": {
                    "code": exc.__class__.__name__,
                    "message": str(exc),
                    "trace": traceback.format_exc(),
                },
            })

    return 0


if __name__ == "__main__":
    sys.exit(main())
