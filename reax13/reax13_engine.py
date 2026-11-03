# SPDX-License-Identifier: MIT
"""Decision Index adapter. Start one persistent isolated GPU backend per engine."""
import atexit
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

from decision_index.engines import Engine, Unsupported

ROOT = Path(__file__).resolve().parent
PROXY_MODEL = 'jev-1.13.0'


class EngineUnavailable(RuntimeError):
    """The native backend is gone (timeout, GPU failure or crash); this engine cannot serve until restarted."""


def proxy_tokens(state, questions):
    """A character-count size proxy: length of the compact request JSON // 4 (the literal model name is fixed, not looked up). No tokenizer, no clock, no answer-dependent step."""
    return max(1, len(json.dumps({'model': PROXY_MODEL, 'state': state, 'questions': questions},
                                 ensure_ascii=False, separators=(',', ':'))) // 4)


class Reax13Engine(Engine):
    name = 'reax13'
    latency = ('Synchronous request wall time: JSON IPC, both native prompt renderers, tokenization, '
               'two FP8 forwards on GPU0, CUDA synchronization and equal probability averaging; '
               'excludes loading and warmup. One external request at a time.')

    def __init__(self, config=None, model=None, **options):
        super().__init__(**options)
        self.config = json.loads(Path(config or ROOT / 'reax13.json').read_text())
        self.config.update(options)
        self.max_input_proxy_tokens = int(self.config['max_input_proxy_tokens'])
        if self.max_input_proxy_tokens <= 0:
            raise ValueError('max_input_proxy_tokens must be positive')
        if self.config['mode'] != 'reax13' or self.config['quantization'] != 'fp8':
            raise ValueError('this engine supports reax13 FP8 only')
        if self.config['optimization'] not in ('fastconv_longcache', 'plain'):
            raise ValueError('unmeasured optimization requested')
        if os.environ.get('DUO_ISOLATED') != '1' or not Path('/.dockerenv').exists():
            raise RuntimeError('run inside the supplied isolated GPU container')
        model_root = Path(model or self.config['model_root']).resolve()
        paths = {part: (model_root / self.config[part]).resolve() for part in ('part_g', 'part_q')}
        if any(not p.is_dir() for p in paths.values()):
            raise FileNotFoundError('model root must contain both part-g/ and part-q/')
        env = dict(os.environ, REAX_PART_G=str(paths['part_g']), REAX_PART_Q=str(paths['part_q']))
        self.lock = threading.Lock()
        self._buf = b''
        self.process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'backend.py')],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1, env=env)
        atexit.register(self.close)
        try:
            ready = self._read(float(self.config['startup_timeout']))
            if not ready.get('ready'):
                raise RuntimeError('backend did not become ready: ' + str(ready))
            self.ready = ready
        except BaseException:
            self.close()
            raise
        self.provenance = {'mode': 'reax13', 'config': self.config,
                           'parts': {k: str(v) for k, v in paths.items()},
                           'mix': '0.5*G probability + 0.5*Q probability; normalize roundoff only',
                           'kit_revision': '25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d',
                           'policy': 'native prompts and temperatures; no item-specific rules or truncation'}

    def _read(self, timeout):
        """One reply line within `timeout` seconds in total (also across a partial line). A bad line closes the backend: a stale reply must never be matched to a later request."""
        deadline = time.monotonic() + timeout
        fd = self.process.stdout.fileno()
        while b'\n' not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                self.close()
                raise TimeoutError('REAX backend response deadline exceeded; backend closed')
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                status = self.process.poll()
                self.close()
                raise RuntimeError('REAX backend exited: ' + str(status))
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b'\n')
        try:
            reply = json.loads(line.decode('utf-8'))
        except ValueError:
            self.close()
            raise RuntimeError('REAX backend sent an unreadable reply line; backend closed')
        if not isinstance(reply, dict):
            self.close()
            raise RuntimeError('REAX backend sent a non-object reply line; backend closed')
        return reply

    # Exact reply shapes of backend.py (the 'ok' and 'error' prints at the end of its request loop).
    OK_KEYS = ('Duo', 'timings_ms', 'peak_bytes', 'peak_reserved_bytes', 'optimization')
    ERROR_KEYS = ('error', 'exception')

    @classmethod
    def _is_reply(cls, reply):
        """True only for a line shaped like one of backend.py's two replies. Known limit: a stray line with exactly the shape of a real
        reply cannot be detected without request ids; only a line that is not reply-shaped closes the backend."""
        status = reply.get('status')
        if status == 'ok':
            return all(key in reply for key in cls.OK_KEYS)
        if status == 'error':
            return all(key in reply for key in cls.ERROR_KEYS)
        return False

    def alive(self):
        process = getattr(self, 'process', None)
        return process is not None and process.poll() is None

    def __call__(self, state, questions):
        if not questions or len(questions) > 64:
            raise Unsupported('supports 1..64 questions per request')
        for question in questions.values():
            if question['type'] not in ('choice', 'noul', 'score'):
                raise Unsupported('unsupported question type ' + question['type'])
            if question['type'] == 'choice' and not 2 <= len(question['criteria']) <= 255:
                raise Unsupported('options per choice must be 2..255')
        size = proxy_tokens(state, questions)
        if size > self.max_input_proxy_tokens:
            # Declared input cap: refused before any model work, never truncated.
            raise Unsupported('request of %d proxy tokens exceeds this engine\'s declared maximum context length of %d proxy tokens; input is never truncated'
                              % (size, self.max_input_proxy_tokens))
        with self.lock:
            if not self.alive():
                raise EngineUnavailable('REAX backend is not running (closed after a timeout, GPU failure or crash); restart the engine')
            request = {'state': state, 'questions': questions,
                       '_kind': 'concurrent' if self.config['optimization'] != 'plain' else 'sequential',
                       '_ubatch': int(self.config['u_batch_size']), '_opt': self.config['optimization']}
            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()
            reply = self._read(float(self.config['request_timeout']))
            if not self._is_reply(reply):
                # Not a reply (e.g. a stray JSON object printed by a native library): the real reply is still queued.
                self.close()
                raise RuntimeError('REAX backend sent a line that is not a reply; backend closed')
            if reply.get('status') != 'ok':
                message = reply.get('error', 'native backend error')
                if reply.get('exception') in ('OutOfMemoryError', 'AcceleratorError') or 'device-side assert' in message:
                    self.close()
                    raise RuntimeError('device-side assert or GPU memory failure; engine closed: ' + message)
                if reply.get('exception') == 'Unsupported' or 'maximum context length' in message:
                    raise Unsupported(message)
                raise RuntimeError(message)
            return reply['Duo'], {k: reply[k] for k in ('timings_ms', 'peak_bytes', 'peak_reserved_bytes', 'optimization')}

    def warmup(self):
        super().warmup()
        path = self.config.get('warmup_rows')
        if path:
            rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
            fits = [row for row in rows if proxy_tokens(row['state'], row['questions']) <= self.max_input_proxy_tokens]
            warm = sorted(fits, key=lambda row: len(json.dumps(row)), reverse=True)[:3] + fits[:10]
            for row in warm:
                try:
                    self(row['state'], row['questions'])
                except Unsupported:
                    pass

    def runtime(self):
        return self.ready

    def close(self):
        process = getattr(self, 'process', None)
        if process is not None:
            self.process = None
            self._buf = b''
            if process.poll() is None:
                process.stdin.close()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
