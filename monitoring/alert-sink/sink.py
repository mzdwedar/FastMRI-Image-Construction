"""Alertmanager webhook receiver for local runs and e2e: logs every alert as one JSON line and serves the most
recent ones back, so a test can assert that an alert arrived. Standard library only.

  POST /alerts   Alertmanager webhook payload (v4)
  GET  /alerts   JSON list of received alerts, newest last (at most KEEP)
  GET  /healthz
"""

import json
import sys
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEEP = 200
received: deque[dict] = deque(maxlen=KEEP)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send(200, {"status": "ok"})
        elif self.path == "/alerts":
            self._send(200, list(received))
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/alerts":
            self._send(404, {"error": "not found"})
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            alerts = payload["alerts"]
        except (ValueError, KeyError, TypeError):
            self._send(400, {"error": "expected an Alertmanager webhook payload"})
            return
        for a in alerts:
            labels, notes = a.get("labels", {}), a.get("annotations", {})
            entry = {
                "status": a.get("status"),
                "alertname": labels.get("alertname"),
                "severity": labels.get("severity"),
                "summary": notes.get("summary"),
                "starts_at": a.get("startsAt"),
                "labels": labels,
            }
            received.append(entry)
            print(json.dumps({"event": "alert", **entry}), flush=True)
        self._send(200, {"received": len(alerts)})

    def log_message(self, format: str, *args: object) -> None:  # request lines add nothing to the alert log
        pass


def main(port: int = 8080) -> None:
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8080)
