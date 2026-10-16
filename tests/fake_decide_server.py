"""Stand-in decision server for the pod_validate dry run: python tests/fake_decide_server.py PORT systemone|s1."""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._send(200 if self.path == "/health" else 404, {"ok": True})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                p = {k: (0.6 if i == 0 else 0.4 / max(1, len(keys) - 1)) for i, k in enumerate(keys)}
                answers[qid] = {"type": "choice", "probabilities": p, "choice": keys[0]}
            elif q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.7}
            else:
                n = len(q["criteria"])
                answers[qid] = {"type": "score", "probabilities": {str(i): 1 / n for i in range(n)}, "score": (n - 1) / 2}
        self._send(200, {"answers": answers})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
