# SPDX-License-Identifier: MIT
"""HTTP adapter for the existing Reax13Engine path.

The model is loaded once at process start. Each POST body is passed directly to
Reax13Engine(state, questions); this module only validates and serializes the
JevBench/System One request and response contract.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
DEFAULT_MAX_BODY_BYTES = 16 * 1024 * 1024
QUESTION_TYPES = {"choice", "noul", "score"}


class RequestError(ValueError):
    """A malformed or unsupported System One request."""


class ResponseError(RuntimeError):
    """The local engine did not satisfy the HTTP response contract."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def validate_request(payload: Any) -> tuple[Any, dict[str, dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise RequestError("request body must be a JSON object")
    extra = set(payload) - {"state", "questions", "model"}
    if extra:
        raise RequestError("unsupported request field(s): " + ", ".join(sorted(extra)))
    model = payload.get("model")
    if model is not None and model != "reax13":
        raise RequestError("model must be 'reax13' when supplied")
    state = payload.get("state", "")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestError("questions must be a non-empty object keyed by question id")
    if len(questions) > 64:
        raise RequestError("one request supports at most 64 questions")

    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid:
            raise RequestError("question ids must be non-empty strings")
        if not isinstance(question, dict):
            raise RequestError(f"question {qid!r} must be an object")
        kind = question.get("type")
        if not isinstance(kind, str) or kind not in QUESTION_TYPES:
            raise RequestError(f"question {qid!r} has unsupported type {kind!r}")
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise RequestError(f"choice question {qid!r} needs 2..255 criteria")
        elif kind == "noul":
            if criteria is not None and not isinstance(criteria, dict):
                raise RequestError(f"noul question {qid!r} criteria must be an object when supplied")
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise RequestError(f"score question {qid!r} needs 2..10 ordered criteria")
    return state, questions


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RequestError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise RequestError(f"{field} must be finite and between 0 and 1")
    return result


def _distribution(answer: dict[str, Any], expected_keys: list[str], field: str) -> dict[str, float]:
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(expected_keys):
        raise RequestError(f"{field} must contain exactly the supplied option keys")
    result = {key: _number(probabilities[key], f"{field}.{key}") for key in expected_keys}
    if abs(math.fsum(result.values()) - 1.0) > 1e-9:
        raise RequestError(f"{field} must sum to 1")
    return result


def validate_response(questions: dict[str, dict[str, Any]], response: Any) -> dict[str, Any]:
    if not isinstance(response, dict) or response.get("model") != "reax13" or not isinstance(response.get("answers"), dict):
        raise RequestError("engine returned an invalid response object")
    answers = response["answers"]
    if set(answers) != set(questions):
        raise RequestError("engine response question ids do not match the request")

    for qid, question in questions.items():
        answer = answers[qid]
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise RequestError(f"engine returned an invalid answer for question {qid!r}")
        kind = question["type"]
        if kind == "choice":
            keys = list(question["criteria"])
            probabilities = _distribution(answer, keys, f"answers.{qid}.probabilities")
            if answer.get("choice") not in question["criteria"]:
                raise RequestError(f"engine returned an unknown choice for question {qid!r}")
            if answer["choice"] != max(sorted(probabilities), key=probabilities.get):
                raise RequestError(f"engine choice for question {qid!r} does not match its probability argmax")
        elif kind == "noul":
            # Preserve the native scalar field and expose its normalized binary
            # distribution as well, so callers can consume every option uniformly.
            p_true = _number(answer.get("noul"), f"answers.{qid}.noul")
            answer["probabilities"] = {"false": 1.0 - p_true, "true": p_true}
        else:
            keys = [str(i) for i in range(len(question["criteria"]))]
            probabilities = _distribution(answer, keys, f"answers.{qid}.probabilities")
            expected_score = math.fsum(int(key) * probabilities[key] for key in keys)
            if isinstance(answer.get("score"), bool) or not isinstance(answer.get("score"), (int, float)):
                raise RequestError(f"answers.{qid}.score must be numeric")
            try:
                score = float(answer.get("score"))
            except (TypeError, ValueError, OverflowError) as exc:
                raise RequestError(f"answers.{qid}.score must be numeric") from exc
            if not math.isfinite(score) or abs(score - expected_score) > 1e-8:
                raise RequestError(f"answers.{qid}.score must equal the distribution expectation")
    return response


class EngineService:
    def __init__(self, engine: Any):
        self.engine = engine
        self.lock = threading.Lock()

    def infer(self, payload: Any) -> dict[str, Any]:
        state, questions = validate_request(payload)
        with self.lock:
            response, _diagnostics = self.engine(state, questions)
        try:
            return validate_response(questions, response)
        except RequestError as exc:
            raise ResponseError(str(exc)) from exc


class ReaxHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, address: tuple[str, int], service: EngineService, max_body_bytes: int):
        super().__init__(address, ReaxHandler)
        self.service = service
        self.max_body_bytes = max_body_bytes


class ReaxHandler(BaseHTTPRequestHandler):
    server_version = "reax13-http/1"
    sys_version = ""

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep request bodies and prompt/state values out of logs.
        super().log_message(fmt, *args)

    def _send_json(self, status: HTTPStatus, value: Any) -> None:
        try:
            data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            data = b'{"error":{"code":"internal_error","message":"response was not valid JSON"}}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/healthz":
            alive = getattr(self.server.service.engine, "alive", None)
            if callable(alive) and not alive():
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"status": "unavailable", "model": "reax13", "mode": "reax13"})
                return
            self._send_json(HTTPStatus.OK, {"status": "ready", "model": "reax13", "mode": "reax13"})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"code": "not_found", "message": "unknown endpoint"}})

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/v1/systemone":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"code": "not_found", "message": "unknown endpoint"}})
            return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": {"code": "content_type", "message": "Content-Type must be application/json"}})
            return
        length_text = self.headers.get("Content-Length")
        if length_text is None or not length_text.isascii() or not length_text.isdecimal():
            self._send_json(HTTPStatus.LENGTH_REQUIRED, {"error": {"code": "content_length", "message": "a valid Content-Length is required"}})
            return
        if len(length_text) > 9:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": {"code": "body_too_large", "message": "request body exceeds the configured size limit"}})
            return
        length = int(length_text)
        if length > self.server.max_body_bytes:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": {"code": "body_too_large", "message": "request body exceeds the configured size limit"}})
            return
        raw = self.rfile.read(length)
        if len(raw) != length:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"code": "incomplete_body", "message": "request body ended before Content-Length"}})
            return
        try:
            payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"code": "invalid_json", "message": str(exc)}})
            return
        try:
            response = self.server.service.infer(payload)
        except RequestError as exc:
            self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": {"code": "invalid_request", "message": str(exc)}})
            return
        except RuntimeError as exc:
            if type(exc).__name__ == "EngineUnavailable":
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": {"code": "engine_unavailable", "message": "native backend is not running"}})
                return
            self.log_error("inference failed (%s)", type(exc).__name__)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"code": "inference_error", "message": "inference failed"}})
            return
        except ValueError as exc:
            # Reax13Engine uses Decision Index's Unsupported (a ValueError)
            # for native capacity/context refusals.
            if type(exc).__name__ == "Unsupported":
                self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": {"code": "unsupported", "message": str(exc)}})
                return
            self.log_error("inference failed (%s)", type(exc).__name__)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"code": "inference_error", "message": "inference failed"}})
            return
        except Exception as exc:  # noqa: BLE001 - keep a failing request from stopping the server
            self.log_error("inference failed (%s)", type(exc).__name__)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"code": "inference_error", "message": "inference failed"}})
            return
        self._send_json(HTTPStatus.OK, response)

    def do_PUT(self) -> None:
        self._method_not_allowed()

    def do_DELETE(self) -> None:
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", "GET, POST")
        self.send_header("Content-Length", "0")
        self.end_headers()


def build_server(engine: Any, host: str = "0.0.0.0", port: int = 8080,
                 max_body_bytes: int = DEFAULT_MAX_BODY_BYTES) -> ReaxHTTPServer:
    if not 0 < max_body_bytes <= 128 * 1024 * 1024:
        raise ValueError("max_body_bytes must be in 1..134217728")
    return ReaxHTTPServer((host, port), EngineService(engine), max_body_bytes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the reax13 Engine API over HTTP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", default=None, help="repository root containing part-g/ and part-q/")
    parser.add_argument("--config", default=str(ROOT / "reax13.json"))
    parser.add_argument("--max-body-bytes", type=int, default=DEFAULT_MAX_BODY_BYTES)
    args = parser.parse_args(argv)

    # Import only at startup so schema tests run on CPU without the GPU kit.
    from reax13_engine import Reax13Engine

    engine = Reax13Engine(config=args.config, model=args.model)
    try:
        engine.warmup()
        server = build_server(engine, args.host, args.port, args.max_body_bytes)
    except BaseException:
        engine.close()
        raise
    host, port = server.server_address[:2]
    print(f"reax13 ready at http://{host}:{port}/v1/systemone; health http://{host}:{port}/healthz", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
