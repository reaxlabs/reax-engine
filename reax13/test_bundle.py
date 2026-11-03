# SPDX-License-Identifier: MIT
"""CPU tests for the declared input cap, the size-routing predicate and dead-backend handling.

A stub backend speaking backend.py's JSON line protocol stands in for the GPU process. Needs the Decision Index kit on
PYTHONPATH (as in the runtime image); falls back to a minimal copy of its Engine/Unsupported when it is absent.
"""
import ast
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
try:
    import decision_index.engines  # noqa: F401
except ImportError:
    pkg = types.ModuleType('decision_index')
    eng = types.ModuleType('decision_index.engines')

    class Unsupported(ValueError):
        pass

    class Engine:
        def __init__(self, **options):
            self.options = options

        def warmup(self):
            pass

    eng.Engine, eng.Unsupported = Engine, Unsupported
    pkg.engines = eng
    sys.modules.update({'decision_index': pkg, 'decision_index.engines': eng})

import reax13_engine as module  # noqa: E402
from serve import build_server  # noqa: E402

CONFIG = json.loads((HERE / 'reax13.json').read_text())
HAS_ROUTE = 'route_u_px' in CONFIG

STUB = r'''
import json, sys, time
print(json.dumps({"ready": True}), flush=True)
log = open(sys.argv[1], "a")
for line in sys.stdin:
    r = json.loads(line)
    state = r["state"]
    log.write(json.dumps({**{k: v for k, v in r.items() if k.startswith("_")}, "_n": len(state) if isinstance(state, str) else None, "_head": state[:8] if isinstance(state, str) else None, "_tail": state[-8:] if isinstance(state, str) else None}) + "\n"); log.flush()
    if state == "SLEEP":
        time.sleep(30)
    if state == "STRAY":
        print("stray native library output", flush=True)
    if state == "STRAYDICT":
        print(json.dumps({"event": "autotune"}), flush=True)
    if state == "STRAYOK":
        print(json.dumps({"status": "ok"}), flush=True)
    if state == "STRAYINFO":
        print(json.dumps({"status": "info", "note": "tuning"}), flush=True)
    if state == "STRAYERR":
        print(json.dumps({"status": "error", "error": "x"}), flush=True)
    if state == "AFTERDICT":
        print(json.dumps({"status": "ok", "Duo": {"model": "reax13", "answers": {}}, "timings_ms": {}, "peak_bytes": 0, "peak_reserved_bytes": 0, "optimization": {"name": "stub"}}), flush=True)
        print(json.dumps({"event": "autotune"}), flush=True)
        continue
    if state == "ARRAY":
        print("[1, 2]", flush=True)
        continue
    if state == "PARTIAL":
        sys.stdout.write('{"status": "ok", "Duo'); sys.stdout.flush()
        time.sleep(3)
    if state == "CRASH":
        sys.exit(3)
    if state == "OOM":
        print(json.dumps({"status": "error", "error": "CUDA out of memory", "exception": "OutOfMemoryError"}), flush=True)
        continue
    answers = {}
    for qid, q in r["questions"].items():
        keys = list(q["criteria"])
        answers[qid] = {"type": "choice", "choice": sorted(keys)[0], "probabilities": {k: (0.6 if k == sorted(keys)[0] else 0.4 / (len(keys) - 1)) for k in keys}}
    print(json.dumps({"status": "ok", "Duo": {"model": "reax13", "answers": answers}, "timings_ms": {}, "peak_bytes": 0,
                      "peak_reserved_bytes": 0, "optimization": {"name": "stub"}}), flush=True)
'''

Q = {'q': {'type': 'choice', 'instructions': 'Pick', 'criteria': {'a': 'A', 'b': 'B'}}}


def state_for(target):
    """A state string whose request has exactly `target` proxy tokens (floor(len/4))."""
    base = len(json.dumps({'model': 'jev-1.13.0', 'state': '', 'questions': Q}, ensure_ascii=False, separators=(',', ':')))
    state = 'x' * (4 * target - base)
    assert module.proxy_tokens(state, Q) == target
    return state


class StubEngineCase(unittest.TestCase):
    def make(self, **options):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for part in ('part-g', 'part-q'):
            (root / part).mkdir()
        (root / 'stub.py').write_text(STUB)
        self.log = root / 'requests.log'
        real_popen = module.subprocess.Popen
        real_exists = Path.exists

        def popen(cmd, **kw):
            return real_popen([sys.executable, '-u', str(root / 'stub.py'), str(self.log)], **kw)

        env = {'DUO_ISOLATED': '1'}
        env.update(options.pop('env', {}))
        with mock.patch.dict(os.environ, env), mock.patch.object(module.subprocess, 'Popen', popen), \
                mock.patch.object(Path, 'exists', lambda p: True if str(p) == '/.dockerenv' else real_exists(p)):
            engine = module.Reax13Engine(config=str(HERE / 'reax13.json'), model=str(root), **options)
        self.addCleanup(engine.close)
        return engine

    def seen(self):
        return [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.exists() else []


class CapTests(StubEngineCase):
    def test_config_value(self):
        self.assertEqual(CONFIG['max_input_proxy_tokens'], 32000)
        self.assertEqual(self.make().max_input_proxy_tokens, 32000)

    def test_boundary_32000_ok_32001_unsupported_before_model_work(self):
        engine = self.make()
        engine(state_for(32000), Q)
        self.assertEqual(len(self.seen()), 1)
        with self.assertRaises(module.Unsupported):
            engine(state_for(32001), Q)
        self.assertEqual(len(self.seen()), 1, 'over-cap request must not reach the backend')
        self.assertTrue(engine.alive())
        engine('short', Q)
        self.assertEqual(len(self.seen()), 2)

    def test_never_truncates(self):
        engine = self.make()
        state = 'HEAD1234' + state_for(32000)[16:] + 'TAIL5678'
        self.assertEqual(module.proxy_tokens(state, Q), 32000)
        engine(state, Q)
        seen = self.seen()[-1]
        self.assertEqual((seen['_n'], seen['_head'], seen['_tail']), (len(state), 'HEAD1234', 'TAIL5678'))

    def test_cap_is_configurable(self):
        engine = self.make(max_input_proxy_tokens=50)
        with self.assertRaises(module.Unsupported):
            engine(state_for(51), Q)
        engine(state_for(50), Q)

    def test_proxy_matches_suite_formula(self):
        for state in ('', 'abc', {'k': 'äöü€'}, ['x'] * 50):
            expected = max(1, len(json.dumps({'model': 'jev-1.13.0', 'state': state, 'questions': Q}, ensure_ascii=False, separators=(',', ':'))) // 4)
            self.assertEqual(module.proxy_tokens(state, Q), expected)

    def test_http_422_unsupported(self):
        engine = self.make()
        server = build_server(engine, '127.0.0.1', 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port = server.server_address[1]

        def post(state):
            body = json.dumps({'state': state, 'questions': Q}).encode()
            try:
                with urlopen(Request(f'http://127.0.0.1:{port}/v1/systemone', body, {'Content-Type': 'application/json'})) as r:
                    return r.status, json.loads(r.read())
            except HTTPError as e:
                return e.code, json.loads(e.read())

        self.assertEqual(post(state_for(32000))[0], 200)
        status, body = post(state_for(32001))
        self.assertEqual((status, body['error']['code']), (422, 'unsupported'))

    def test_http_422_text_carries_kit_capacity_marker(self):
        markers = ("options per choice", "a choice needs at least two options", "a score takes 2 to 10 levels", "the canvas holds",
                   "maximum context length", "maximum model length", "longer than the maximum model length", "context window", "too many tokens")
        engine = self.make()
        with self.assertRaises(module.Unsupported) as ctx:
            engine(state_for(32001), Q)
        self.assertTrue(any(m in str(ctx.exception) for m in markers), str(ctx.exception))
        server = build_server(engine, '127.0.0.1', 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        body = json.dumps({'state': state_for(32001), 'questions': Q}).encode()
        with self.assertRaises(HTTPError) as http:
            urlopen(Request(f'http://127.0.0.1:{server.server_address[1]}/v1/systemone', body, {'Content-Type': 'application/json'}))
        self.assertEqual(http.exception.code, 422)
        body = http.exception.read().decode()
        self.assertTrue(any(m in body for m in markers), body)

    def test_warmup_filters_over_cap_before_picking_longest(self):
        rows = Path(tempfile.mkdtemp()) / 'rows.jsonl'
        lines = [json.dumps({'state': state_for(32001), 'questions': Q})] + [json.dumps({'state': 's' * (i + 1), 'questions': Q}) for i in range(14)]
        rows.write_text('\n'.join(lines) + '\n')
        engine = self.make(warmup_rows=str(rows))
        # The kit's own Engine.warmup (present in the runtime image) sends rows of its own; count only this engine's.
        with mock.patch.object(module.Engine, 'warmup', lambda self: None):
            engine.warmup()
        sent = self.seen()
        self.assertEqual(len(sent), 13, 'over-cap row skipped, then 3 longest + first 10 fitting rows')
        self.assertEqual(sorted(s['_n'] for s in sent[:3]), [12, 13, 14])
        self.assertNotIn(len('x' * 130000), [s['_n'] for s in sent])


class DeadEngineTests(StubEngineCase):
    def http(self, engine):
        server = build_server(engine, '127.0.0.1', 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port = server.server_address[1]

        def get():
            try:
                with urlopen(f'http://127.0.0.1:{port}/healthz') as r:
                    return r.status, json.loads(r.read())
            except HTTPError as e:
                return e.code, json.loads(e.read())

        def post(state):
            body = json.dumps({'state': state, 'questions': Q}).encode()
            try:
                with urlopen(Request(f'http://127.0.0.1:{port}/v1/systemone', body, {'Content-Type': 'application/json'})) as r:
                    return r.status, json.loads(r.read())
            except HTTPError as e:
                return e.code, json.loads(e.read())
        return get, post

    def test_normal_rows_unchanged_and_healthy(self):
        engine = self.make()
        first = engine('s', Q)[0]
        get, post = self.http(engine)
        self.assertEqual(get()[0], 200)
        self.assertEqual(post('s')[1]['answers'], first['answers'])

    def test_timeout_then_loud_failure_and_unhealthy(self):
        engine = self.make(request_timeout=0.5)
        engine('before', Q)
        with self.assertRaises(TimeoutError):
            engine('SLEEP', Q)
        self.assertFalse(engine.alive())
        with self.assertRaises(module.EngineUnavailable) as ctx:
            engine('after', Q)
        self.assertNotIsInstance(ctx.exception, AttributeError)
        get, post = self.http(engine)
        self.assertEqual(get(), (503, {'status': 'unavailable', 'model': 'reax13', 'mode': 'reax13'}))
        status, body = post('after')
        self.assertEqual((status, body['error']['code']), (503, 'engine_unavailable'))

    def test_oom_closes_engine_then_loud_failure(self):
        engine = self.make()
        with self.assertRaises(RuntimeError):
            engine('OOM', Q)
        self.assertFalse(engine.alive())
        with self.assertRaises(module.EngineUnavailable):
            engine('after', Q)

    def test_backend_crash_is_unhealthy(self):
        engine = self.make()
        with self.assertRaises(RuntimeError):
            engine('CRASH', Q)
        self.assertFalse(engine.alive())
        with self.assertRaises(module.EngineUnavailable):
            engine('after', Q)
        self.assertEqual(self.http(engine)[0]()[0], 503)

    def test_stray_or_non_object_reply_line_closes_backend_no_desync(self):
        for bad in ('STRAY', 'ARRAY'):
            engine = self.make()
            engine('before', Q)
            with self.assertRaises(RuntimeError):
                engine(bad, Q)
            self.assertFalse(engine.alive(), bad)
            with self.assertRaises(module.EngineUnavailable):
                engine('row-3', Q)

    def test_stray_json_object_line_never_shifts_later_replies(self):
        # A line that parses as a JSON object but is not a reply: the real reply stays queued, so the backend must close.
        engine = self.make()
        with self.assertRaises(RuntimeError):
            engine('STRAYDICT', Q)
        self.assertFalse(engine.alive())
        for row in ('row-3', 'row-4'):
            with self.assertRaises(module.EngineUnavailable):
                engine(row, Q)

    def test_stray_status_lines_not_reply_shaped_close_backend(self):
        # Lines with a `status` key but not the exact shape of backend.py's replies must not shift later answers.
        for bad in ('STRAYOK', 'STRAYINFO', 'STRAYERR'):
            engine = self.make()
            engine('before', Q)
            with self.assertRaises(RuntimeError):
                engine(bad, Q)
            self.assertFalse(engine.alive(), bad)
            for row in ('row-3', 'row-4'):
                with self.assertRaises(module.EngineUnavailable):
                    engine(row, Q)

    def test_stray_json_object_line_right_after_a_good_reply(self):
        engine = self.make()
        self.assertEqual(engine('AFTERDICT', Q)[0]['model'], 'reax13')
        with self.assertRaises(RuntimeError):
            engine('row-2', Q)
        self.assertFalse(engine.alive())
        with self.assertRaises(module.EngineUnavailable):
            engine('row-3', Q)
        self.assertEqual(self.http(engine)[0]()[0], 503)

    def test_deadline_enforced_across_partial_line(self):
        engine = self.make(request_timeout=0.7)
        with self.assertRaises(TimeoutError):
            engine('PARTIAL', Q)
        self.assertFalse(engine.alive())

    def test_close_is_idempotent_and_requests_after_close_fail_loudly(self):
        engine = self.make()
        engine.close()
        engine.close()
        with self.assertRaises(module.EngineUnavailable):
            engine('x', Q)


class BenchmarkDriverTests(unittest.TestCase):
    def test_over_cap_row_is_recorded_unsupported_and_run_continues(self):
        import benchmark

        class Fake:
            provenance, latency = {}, 'x'

            def __init__(self, **kw):
                pass

            def runtime(self):
                return {}

            def warmup(self):
                pass

            def synchronize(self):
                pass

            def close(self):
                pass

            def __call__(self, state, questions):
                if state == 'BIG':
                    raise module.Unsupported('exceeds this engine\'s declared maximum context length')
                return {'model': 'reax13', 'answers': {'q': {'type': 'choice', 'choice': 'a', 'probabilities': {'a': 0.6, 'b': 0.4}}}}, {}

        tmp = Path(tempfile.mkdtemp())
        rows = tmp / 'rows.jsonl'
        rows.write_text('\n'.join(json.dumps({'state': s, 'questions': Q, '_evaluation': {'id': s}}) for s in ('a', 'BIG', 'c')) + '\n')
        with mock.patch.object(benchmark, 'Reax13Engine', Fake), mock.patch.object(sys, 'argv', ['benchmark', '--rows', str(rows), '--out', str(tmp / 'out')]):
            benchmark.main()
        out = [json.loads(x) for x in (tmp / 'out' / 'results.jsonl').read_text().splitlines()]
        self.assertEqual([(o['id'], o['status']) for o in out], [('a', 'ok'), ('BIG', 'unsupported'), ('c', 'ok')])
        self.assertEqual(json.loads((tmp / 'out' / 'status.json').read_text())['counts'], {'ok': 2, 'unsupported': 1})


def backend_proxy_tokens():
    """Compile the backend's own proxy_tokens source (backend.py needs a GPU to import)."""
    tree = ast.parse((HERE / 'backend.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'proxy_tokens':
            scope = {'json': json}
            exec(compile(ast.Module([node], []), 'backend.py', 'exec'), scope)
            return scope['proxy_tokens']
    return None


@unittest.skipUnless(HAS_ROUTE, 'routing rule only in engine-EC-R')
class RouteTests(StubEngineCase):
    def test_default_1000_in_json(self):
        self.assertEqual(CONFIG['route_u_px'], 1000)
        self.assertEqual(self.make().route_u_px, 1000)

    def test_env_overrides_and_zero_is_off(self):
        self.assertEqual(self.make(env={'REAX_ROUTE_U_PX': '5000'}).route_u_px, 5000)
        self.assertEqual(self.make(env={'REAX_ROUTE_U_PX': '0'}).route_u_px, 0)

    def test_engine_sends_threshold_to_backend(self):
        engine = self.make()
        engine('s', Q)
        self.assertEqual(self.seen()[-1]['_route_u_px'], 1000)

    def test_predicate_boundaries(self):
        fn = backend_proxy_tokens()
        self.assertIsNotNone(fn)
        for target, routed_at_1000, over_cap in ((1000, False, False), (1001, True, False), (32000, True, False), (32001, True, True)):
            state = state_for(target)
            r = {'state': state, 'questions': Q}
            self.assertEqual(fn(r), target)
            self.assertEqual(fn(r) > 1000, routed_at_1000, target)
            self.assertEqual(fn(r) > CONFIG['max_input_proxy_tokens'], over_cap, target)

    def test_backend_and_engine_predicates_agree(self):
        fn = backend_proxy_tokens()
        for state in ('', 'abc', {'k': 'äöü€' * 100}, ['x'] * 50):
            self.assertEqual(fn({'state': state, 'questions': Q}), module.proxy_tokens(state, Q))

    def test_backend_routing_source(self):
        text = (HERE / 'backend.py').read_text()
        self.assertIn("routed=kind=='concurrent' and route_px>0 and proxy_tokens(r)>route_px", text)


@unittest.skipIf(HAS_ROUTE, 'engine-EC has no routing rule')
class NoRouteTests(StubEngineCase):
    def test_ec_sends_no_route_and_backend_has_none(self):
        engine = self.make()
        engine('s', Q)
        self.assertNotIn('_route_u_px', self.seen()[-1])
        self.assertNotIn('_route_u_px', (HERE / 'backend.py').read_text())
        self.assertNotIn('REAX_ROUTE_U_PX', (HERE / 'reax13_engine.py').read_text())


class KernelGuardTests(unittest.TestCase):
    """fast_fp8_gemm.py wrapper and asserts, CPU only (the asserts fire before any kernel launch)."""

    def setUp(self):
        try:
            import torch
            import fast_fp8_gemm
        except ImportError as exc:
            self.skipTest('torch/triton not importable: %s' % exc)
        self.torch, self.f = torch, fast_fp8_gemm

    def weights(self, n, k):
        t = self.torch
        return t.zeros(n, k, dtype=t.float8_e4m3fn), t.ones((n + 127) // 128, (k + 127) // 128, dtype=t.float32)

    def test_upstream_k_and_bs_asserts_kept(self):
        t = self.torch
        W, S = self.weights(256, 256)
        with self.assertRaisesRegex(AssertionError, 'K mismatch'):
            self.f.matmul(t.zeros(4, 128, dtype=t.bfloat16), W, S, [128, 128], t.bfloat16)
        with self.assertRaisesRegex(AssertionError, 'Bs shape'):
            self.f.matmul(t.zeros(4, 256, dtype=t.bfloat16), W, S[:1], [128, 128], t.bfloat16)
        with self.assertRaisesRegex(AssertionError, 'block_size'):
            self.f.matmul(t.zeros(4, 256, dtype=t.bfloat16), W, S, [128], t.bfloat16)

    def test_dispatch_guards(self):
        t = self.torch
        calls = []
        fake = types.ModuleType('transformers.integrations.finegrained_fp8')
        fake.fp8_linear = lambda *a: calls.append(('orig', a[-1])) or 'orig'
        mods = {'transformers': types.ModuleType('transformers'), 'transformers.integrations': types.ModuleType('transformers.integrations'),
                'transformers.integrations.finegrained_fp8': fake}
        mods['transformers'].integrations = mods['transformers.integrations']
        mods['transformers.integrations'].finegrained_fp8 = fake
        with mock.patch.dict(sys.modules, mods), mock.patch.object(self.f, 'matmul', lambda *a: calls.append('fast') or 'fast'):
            self.f.install()
            x = t.zeros(3, 256, dtype=t.bfloat16)
            W, S = self.weights(256, 256)
            self.assertEqual(fake.fp8_linear(x, W, S, [128, 128]), 'fast')
            # whole-weight block (N=K=128): not this kernel's shape -> upstream, DeepGEMM off
            Wt, St = self.weights(128, 128)
            self.assertEqual(fake.fp8_linear(t.zeros(3, 128, dtype=t.bfloat16), Wt, St, [128, 128]), 'orig')
            # a call that falls through must not allow DeepGEMM even when the caller asks for it
            self.assertEqual(fake.fp8_linear(x, W, S, [128, 128], None, None, True), 'fast')
            self.assertEqual(fake.fp8_linear(x, W, S, [128, 128], t.ones(256, dtype=t.bfloat16), None, True), 'orig')
        self.assertEqual(calls, ['fast', ('orig', False), 'fast', ('orig', False)])


if __name__ == '__main__':
    unittest.main()
