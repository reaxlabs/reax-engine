# SPDX-License-Identifier: MIT
"""Index-style sequential Engine benchmark, including native score question validation.

The pinned official kit only validates choice/noul. This driver retains its Engine
API and timing boundaries for the frozen mixed choice/noul/score latency sample.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import time

from probabilities import distribution
from decision_index.engines import Unsupported
from reax13_engine import Reax13Engine


def validate(questions, response):
    if set(response.get('answers', {})) != set(questions):
        raise ValueError('question keys mismatch')
    for key, q in questions.items():
        a = response['answers'][key]
        if a.get('type') != q['type']:
            raise ValueError('question type mismatch')
        if q['type'] == 'noul':
            if not math.isfinite(a['noul']) or not 0 <= a['noul'] <= 1:
                raise ValueError('invalid noul probability')
        else:
            p = distribution(q, a)
            if abs(math.fsum(p.values()) - 1) > 1e-12:
                raise ValueError('not normalized')
            if q['type'] == 'choice' and a['choice'] != max(sorted(p), key=p.get):
                raise ValueError('choice does not match argmax')
            if q['type'] == 'score' and abs(a['score'] - math.fsum(int(k)*v for k,v in p.items())) > 1e-9:
                raise ValueError('score does not match expectation')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default='/engine/reax13.json')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines()]
    def status(**fields):
        record = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields}
        (out/'status.json').write_text(json.dumps(record))
        print(json.dumps(record), flush=True)
    engine = None
    try:
        status(event='loading')
        started = time.perf_counter()
        engine = Reax13Engine(config=args.config, warmup_rows=args.rows)
        (out/'environment.json').write_text(json.dumps({'provenance':engine.provenance, **engine.runtime(),
             'loaded_seconds':time.perf_counter()-started, 'latency':engine.latency,
             'driver':'own index-style Engine API clock; adds native score validation to official choice/noul contract'},indent=2))
        engine.warmup()
        engine.synchronize()
        status(event='ready')
        start = time.perf_counter()
        counts = {'ok':0, 'unsupported':0}
        with (out/'results.jsonl').open('w') as f:
            for i, row in enumerate(rows):
                # Same clock boundaries as official runner: synchronize, request,
                # synchronize, validate. Includes JSON IPC and all native work.
                tick = time.perf_counter()
                engine.synchronize()
                try:
                    response, raw = engine(row['state'], row['questions'])
                except Unsupported as exc:
                    elapsed = (time.perf_counter()-tick)*1000
                    f.write(json.dumps({**row['_evaluation'], 'status':'unsupported', 'error':str(exc), 'total_wall_ms':elapsed,
                                        'payload':{'state':row['state'],'questions':row['questions']}})+'\n')
                    f.flush()
                    counts['unsupported'] += 1
                    continue
                engine.synchronize()
                validate(row['questions'], response)
                elapsed = (time.perf_counter()-tick)*1000
                result = {**row['_evaluation'], 'status':'ok', 'response':response,
                          'raw_output':raw, 'total_wall_ms':elapsed, 'model_request_wall_ms':elapsed,
                          'payload':{'state':row['state'],'questions':row['questions']}}
                f.write(json.dumps(result)+'\n')
                f.flush()
                counts['ok'] += 1
                if (i+1)%10 == 0:
                    status(event='progress',completed=i+1,counts=dict(counts),elapsed_seconds=time.perf_counter()-start)
        status(event='complete',completed=len(rows),counts=dict(counts),elapsed_seconds=time.perf_counter()-start)
    finally:
        if engine is not None:
            engine.close()


if __name__ == '__main__':
    main()
