#!/usr/bin/env python3
"""Reference server for Quyet-1.0-Large: the author's `quyet` package (HF transformers, bf16) behind /v1/systemone.
Run it ONLY inside a container (docker/qwen35 image, weights mounted read-only). One request at a time.
Answers carry the package's UNROUNDED probabilities (package.raw_probabilities) so agreement is not limited by its 4-decimal
rounding. Usage: python bench/ref/quyet_ref_server.py --model /model --port 8001"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    import quyet
    from quyet.errors import QuestionError
    from quyet.questions import parse_questions
    m = quyet.load(a.model)
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *x):
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
            if self.path != "/v1/systemone":
                return self._send(404, {})
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                dec = parse_questions(body["questions"])
                with lock:
                    probs = m.raw_probabilities([{"state": body.get("state"), "questions": body["questions"]}])[0]
                out = {}
                for qid, d in dec.items():
                    p = dict(zip(d.labels, (float(x) for x in probs[qid])))
                    a_ = {"type": d.type, "probabilities": p}
                    if d.type == "noul":
                        a_["noul"] = p["true"]
                    out[qid] = a_
                self._send(200, {"model": "quyet-1.0-large", "answers": out})
            except QuestionError as e:
                self._send(422 if "no room" in str(e) or "exceed" in str(e) else 400, {"error": str(e)})
            except (KeyError, TypeError, json.JSONDecodeError) as e:
                self._send(400, {"error": f"bad request: {e}"})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": f"{type(e).__name__}: {e}"[:300]})

    srv = ThreadingHTTPServer((a.host, a.port), H)
    print("ready", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
