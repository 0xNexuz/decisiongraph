"""Minimal HydraDB HTTP query client (OSS engine, github.com/hydra-db/hydradb)."""
import json
import os
import time
import urllib.error
import urllib.request


class HydraError(RuntimeError):
    pass


class Hydra:
    def __init__(self, url=None, token=None, graph="default", cell="cell-0", namespace="default"):
        self.url = (url or os.environ.get("HYDRA_URL", "http://127.0.0.1:8443")).rstrip("/")
        self.token = token or os.environ.get("HYDRA_TOKEN", "local-development-token-32-bytes")
        self.graph, self.cell, self.ns = graph, cell, namespace
        self.calls = 0

    def q(self, query, **params):
        body = {"cell_id": self.cell, "query": query}
        if params:
            body["parameters"] = params
        req = urllib.request.Request(
            f"{self.url}/v1/graphs/{self.graph}/query",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
                     "X-Graph-Namespace": self.ns},
            method="POST",
        )
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                payload = json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")
            raise HydraError(f"{e.code}: {text[:400]} | query: {query[:200]}") from None
        if "error" in payload:
            raise HydraError(f"{payload['error']} | query: {query[:200]}")
        if "rows" not in payload:
            raise HydraError(f"unexpected response {str(payload)[:300]}")
        cols = payload.get("columns", [])
        return [{c: _val(cell) for c, cell in zip(cols, row)} for row in payload["rows"]]

    def healthy(self):
        try:
            with urllib.request.urlopen(f"{self.url}/healthz", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def wait(self, seconds=60):
        t = time.time()
        while time.time() - t < seconds:
            if self.healthy():
                return True
            time.sleep(0.5)
        return False


def _val(cell):
    if cell.get("type") == "list":
        return [_val(c) for c in cell.get("value", [])]
    return cell.get("value")
