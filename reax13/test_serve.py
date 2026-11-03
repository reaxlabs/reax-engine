# SPDX-License-Identifier: MIT
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from probabilities import mix
from serve import EngineService, RequestError, ResponseError, build_server


class MockBackend:
    """CPU-only stand-in for Reax13Engine's native G/Q backend responses."""

    def __init__(self):
        self.calls = []

    def __call__(self, state, questions):
        self.calls.append((state, questions))
        g = {"answers": {}}
        q = {"answers": {}}
        for qid, spec in questions.items():
            if spec["type"] == "choice":
                keys = list(spec["criteria"])
                g["answers"][qid] = {"type": "choice", "probabilities": dict(zip(keys, [0.7, 0.3]))}
                q["answers"][qid] = {"type": "choice", "probabilities": dict(zip(keys, [0.3, 0.7]))}
            elif spec["type"] == "noul":
                g["answers"][qid] = {"type": "noul", "noul": 0.2}
                q["answers"][qid] = {"type": "noul", "noul": 0.8}
            else:
                keys = [str(i) for i in range(len(spec["criteria"]))]
                g["answers"][qid] = {"type": "score", "probabilities": dict(zip(keys, [0.2, 0.3, 0.5]))}
                q["answers"][qid] = {"type": "score", "probabilities": dict(zip(keys, [0.5, 0.25, 0.25]))}
        return mix({"questions": questions}, g, q), {"mock": True}


class BadResponseEngine:
    def __call__(self, state, questions):
        return {"model": "reax13", "answers": {"q": {
            "type": "choice", "choice": "a", "probabilities": {"a": 0.2, "b": 0.2},
        }}}, None


class ServeContractTests(unittest.TestCase):
    def setUp(self):
        self.questions = {
            "pick": {"type": "choice", "instructions": "Pick one", "criteria": {"a": "A", "b": "B"}},
            "certain": {"type": "noul", "instructions": "Is this true?"},
            "rating": {"type": "score", "instructions": "Rate it", "criteria": ["low", "mid", "high"]},
        }
        self.backend = MockBackend()
        self.service = EngineService(self.backend)

    def test_joint_request_all_question_types_and_normalized_answers(self):
        response = self.service.infer({"state": {"case": "example"}, "questions": self.questions})
        answers = response["answers"]
        self.assertEqual(set(answers), set(self.questions))
        self.assertEqual(answers["pick"]["choice"], "a")
        self.assertAlmostEqual(sum(answers["pick"]["probabilities"].values()), 1.0)
        self.assertEqual(answers["certain"]["noul"], 0.5)
        self.assertEqual(answers["certain"]["probabilities"], {"false": 0.5, "true": 0.5})
        score = answers["rating"]
        self.assertAlmostEqual(sum(score["probabilities"].values()), 1.0)
        expected = sum(int(k) * p for k, p in score["probabilities"].items())
        self.assertAlmostEqual(score["score"], expected)
        self.assertEqual(len(self.backend.calls), 1)

    def test_one_question_per_request(self):
        request = {"state": "short state", "questions": {"only": self.questions["pick"]}}
        answer = self.service.infer(request)["answers"]["only"]
        self.assertEqual(answer["type"], "choice")
        self.assertEqual(set(answer["probabilities"]), {"a", "b"})

    def test_invalid_request_rejected_before_backend(self):
        for question_type in ("other", ["choice"]):
            with self.subTest(question_type=question_type):
                with self.assertRaises(RequestError):
                    self.service.infer({"state": "x", "questions": {"q": {"type": question_type}}})
        self.assertEqual(self.backend.calls, [])

    def test_incomplete_or_unnormalized_engine_response_rejected(self):
        service = EngineService(BadResponseEngine())
        with self.assertRaisesRegex(ResponseError, "sum to 1"):
            service.infer({"state": "x", "questions": {"q": self.questions["pick"]}})

    def test_http_health_and_post_contract(self):
        server = build_server(self.backend, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(base + "/healthz", timeout=2) as reply:
                self.assertEqual(json.load(reply), {"status": "ready", "model": "reax13", "mode": "reax13"})
            body = json.dumps({"state": "test", "questions": self.questions}).encode()
            request = Request(base + "/v1/systemone", data=body, headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=2) as reply:
                self.assertEqual(reply.status, 200)
                response = json.load(reply)
            self.assertEqual(set(response["answers"]), set(self.questions))
            self.assertEqual(response["answers"]["certain"]["probabilities"], {"false": 0.5, "true": 0.5})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_http_rejects_invalid_schema(self):
        server = build_server(self.backend, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            request = Request(base + "/v1/systemone", data=b'{"questions":{}}', headers={"Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as caught:
                urlopen(request, timeout=2)
            self.assertEqual(caught.exception.code, 422)
            error = json.load(caught.exception)
            self.assertEqual(error["error"]["code"], "invalid_request")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
