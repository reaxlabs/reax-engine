import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bench import agreement, loadgen, sets, stats


def write(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture()
def data(tmp_path):
    q1 = {"type": "choice", "instructions": "i", "criteria": {"a": "x", "b": "y"}}
    q2 = {"type": "noul", "instructions": "i"}
    q3 = {"type": "score", "instructions": "i", "criteria": ["lo", "mid", "hi"]}
    write(tmp_path / "p300-input.jsonl", [{"task_id": "t1", "state": "s1", "question": q1, "labels": ["a", "b"]},
                                           {"task_id": "t2", "state": "s2", "question": q2, "labels": ["no", "yes"]},
                                           {"task_id": "t3", "state": "s3", "question": q3, "labels": ["0", "1", "2"]}])
    write(tmp_path / "p300-gold.jsonl", [{"opaque_id": "t1", "expected": "a"}, {"opaque_id": "t2", "expected": "yes"},
                                          {"opaque_id": "t3", "expected": 2}])
    rows = [{"id": f"r{i}", "state": f"st{i}", "questions": {"u": q1, "v": q2}, "expected": {"u": "b", "v": False},
             "_evaluation": {"run_id": f"run{i}"}} for i in range(10)]
    write(tmp_path / "di5k-rows.jsonl", rows)
    (tmp_path / "di5k-ids.json").write_text(json.dumps({"ids": [f"run{i}" for i in range(10)]}))
    return tmp_path


def test_sets_p300_and_gold_normalisation(data):
    items = sets.load("p300", data)
    assert [i.key for i in items] == ["t1", "t2", "t3"]
    assert [i.gold for i in items] == ["a", "true", "2"]
    assert sets.option_keys(items[1].question) == ["false", "true"]


def test_sets_di_subset_and_gold(data):
    items = sets.load("di", data, limit=5)
    assert {i.req for i in items} == {"run0", "run2", "run4", "run6", "run8"}
    assert len(items) == 10 and items[1].gold == "false" and items[0].gold == "b" and items[0].key == "run0::u"
    assert len(sets.load("di", data)) == 20


def test_missing_data_dir(monkeypatch):
    monkeypatch.delenv("REAX_BENCH_DATA", raising=False)
    with pytest.raises(SystemExit):
        sets.load("p300")


def test_stats_probs_and_pairs():
    assert stats.probs_dict({"type": "noul", "noul": 0.25}, ["false", "true"]) == {"false": 0.75, "true": 0.25}
    assert stats.probs_dict({"probabilities": {"b": 0.4, "a": 0.6}}, ["a", "b"]) == {"a": 0.6, "b": 0.4}
    assert stats.probs_dict({"choice": "a"}, ["a", "b"]) is None
    m = stats.compare_pair({"a": 0.6, "b": 0.4}, {"a": 0.5, "b": 0.5})
    assert m["agree"] and m["max_dp"] == pytest.approx(0.1)
    assert not stats.compare_pair({"a": 0.6, "b": 0.4}, {"a": 0.4, "b": 0.6})["agree"]
    s = stats.summarize([m, stats.compare_pair({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0})])
    assert s["n"] == 2 and s["top_choice_agreement"] == 1.0 and s["max_abs_dp"] == pytest.approx(0.1)


def test_paired_bootstrap_detects_shift_and_zero():
    x = [1.0, 0.0] * 50
    same = stats.paired_bootstrap(x, x, reps=200)
    assert same["delta_pp"] == 0 and same["ci95"] == [0.0, 0.0]
    up = stats.paired_bootstrap([0.0] * 100, [1.0] * 100, reps=200)
    assert up["delta_pp"] == 100.0 and up["ci95"][0] > 99
    grouped = stats.paired_bootstrap([0, 0, 0, 0], [1, 1, 0, 0], groups=["a", "a", "b", "b"], reps=200)
    assert grouped["delta_pp"] == 50.0 and grouped["ci95"][0] <= 50.0 <= grouped["ci95"][1]


class Server(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    flavour = "systemone"
    shift = 0.0

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        want = "/v1/systemone" if self.flavour == "systemone" else "/internal/v1/decide"
        if self.path != want:
            return self._send(404, {})
        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "score":
                return self._send(422, {"error": "maximum context length exceeded"})
            if self.flavour == "s1" and self.headers.get("X-S1M-Internal-Token") != "tok":
                return self._send(401, {})
            if q["type"] == "choice":
                p = {"a": 0.5 + self.shift, "b": 0.5 - self.shift}
                answers[qid] = {"type": "choice", "probabilities": p, "choice": max(p, key=p.get)}
            else:
                answers[qid] = {"type": "noul", "noul": 0.7 + self.shift}
        self._send(200, {"answers": answers})

    def _send(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def serve(flavour, shift):
    h = type("H", (Server,), {"flavour": flavour, "shift": shift})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def test_agreement_end_to_end(data, tmp_path, monkeypatch):
    monkeypatch.setenv("S1M_INTERNAL_TOKEN", "tok")
    ref_srv, ref = serve("systemone", 0.0)
    cand_srv, cand = serve("s1", 0.3)
    try:
        out = tmp_path / "out" / "r.json"
        rc = agreement.main(["--ref", ref, "--ref-contract", "systemone", "--cand", cand, "--contract", "s1", "--set", "p300",
                             "--data-dir", str(data), "--workers", "2", "--out", str(out), "--reps", "50",
                             "--ref-cache", str(tmp_path / "ref.jsonl")])
        assert rc == 0
        rep = json.loads(out.read_text())
        assert rep["ref"]["answered"] == 2 and rep["ref"]["unsupported"] == 1 and rep["cand"]["errors"] == 0
        assert rep["paired"]["n"] == 2 and rep["paired"]["top_choice_agreement"] == 1.0
        assert rep["paired"]["max_abs_dp"] == pytest.approx(0.3)
        assert rep["gold"]["n"] == 2 and rep["gold"]["ref_accuracy"] == 1.0
        # a saved reference file replaces the reference server
        ref_srv.shutdown()
        rc = agreement.main(["--ref", str(tmp_path / "ref.jsonl"), "--cand", cand, "--contract", "s1", "--set", "p300",
                             "--data-dir", str(data), "--out", str(out), "--reps", "20"])
        assert json.loads(out.read_text())["paired"]["n"] == 2
    finally:
        cand_srv.shutdown()


def test_loadgen_levels(data, tmp_path):
    srv, url = serve("systemone", 0.0)
    try:
        out = tmp_path / "lg.json"
        loadgen.main(["--url", url, "--set", "di", "--data-dir", str(data), "--limit", "3", "--concurrency", "1,2", "--seconds", "0.3",
                      "--grouped", "--out", str(out)])
        lv = json.loads(out.read_text())["levels"]
        assert [x["concurrency"] for x in lv] == [1, 2] and all(x["decisions"] > 0 for x in lv)
        assert lv[0]["req_p50_ms"] >= 0
    finally:
        srv.shutdown()


def test_fetch_only_then_file_reference(data, tmp_path):
    srv, url = serve("systemone", 0.0)
    try:
        cache = tmp_path / "ref.jsonl"
        assert agreement.main(["--ref", url, "--set", "p300", "--data-dir", str(data), "--fetch-only", "--ref-cache", str(cache)]) == 0
        assert len(cache.read_text().splitlines()) == 3
    finally:
        srv.shutdown()
