"""Quick test: pilot the sidecar from Python via a subprocess and print
its responses. No Rust involved yet — purely validates the JSON-RPC loop.
"""
import json
import subprocess
import sys
from pathlib import Path

PY = r"C:\Users\Aziz\miniconda3\envs\ttd-ai\python.exe"
SERVICE = Path(__file__).parent / "service.py"


def call(proc: subprocess.Popen, req_id: int, method: str, params: dict) -> dict:
    """Send a request, then drain stdout until we see the matching response."""
    proc.stdin.write(json.dumps({"id": req_id, "method": method, "params": params}) + "\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("sidecar closed stdout unexpectedly")
        msg = json.loads(line)
        if "event" in msg:
            print(f"  [event] {msg['event']}: {msg['data']}")
            continue
        if msg.get("id") == req_id:
            return msg


def main() -> int:
    proc = subprocess.Popen(
        [PY, str(SERVICE)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
        text=True, encoding="utf-8", bufsize=1,
    )
    try:
        print("-> health()")
        resp = call(proc, 1, "health", {})
        print(f"<- {json.dumps(resp, indent=2)}")
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
