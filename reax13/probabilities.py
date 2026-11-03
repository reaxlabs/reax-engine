# SPDX-License-Identifier: MIT
"""Validate each native distribution, then average probabilities at equal weight."""
import math


def keys_for(q):
    if q['type'] == 'choice':
        return list(q['criteria'])
    if q['type'] == 'score':
        return [str(i) for i in range(len(q['criteria']))]
    raise ValueError('unsupported question type ' + str(q['type']))


def distribution(q, answer):
    if answer.get('type') != q['type']:
        raise ValueError('native question type mismatch')
    keys = keys_for(q)
    probs = answer.get('probabilities', {})
    if set(keys) != set(probs):
        raise ValueError('native response dropped or added options')
    values = {k: float(probs[k]) for k in keys}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError('invalid native option probability')
    total = math.fsum(values.values())
    if total <= 0 or abs(total - 1) > .01:
        raise ValueError('native probabilities are not normalized')
    return values


def mix(request, g, q):
    questions = request['questions']
    if set(g.get('answers', {})) != set(questions) or set(q.get('answers', {})) != set(questions):
        raise ValueError('native question keys mismatch')
    answers = {}
    for qid, spec in questions.items():
        x, y = g['answers'][qid], q['answers'][qid]
        typ = spec['type']
        answer = {'type': typ}
        if typ == 'noul':
            values = [float(a['noul']) for a in (x, y)]
            if any(a.get('type') != typ for a in (x, y)) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError('invalid native noul probability')
            answer['noul'] = .5 * (values[0] + values[1])
        else:
            px, py = distribution(spec, x), distribution(spec, y)
            probs = {k: .5 * (px[k] + py[k]) for k in px}
            total = math.fsum(probs.values())
            probs = {k: v / total for k, v in probs.items()}
            answer['probabilities'] = probs
            if typ == 'choice':
                answer['choice'] = max(sorted(probs), key=probs.get)
            else:
                answer['score'] = math.fsum(int(k) * v for k, v in probs.items())
        answers[qid] = answer
    return {'model': 'reax13', 'answers': answers}
