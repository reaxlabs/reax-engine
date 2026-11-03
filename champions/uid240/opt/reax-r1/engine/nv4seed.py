"""nv4seed.py (lane QX): the PUBLIC seed set the validator replays before every session (reax-subnet engrunner.
reference_self_check: items_for_pool(pool, timed_items, seed=0x52454158)), rendered as request bodies {state, questions}
exactly like the validator's client (parse_decision(item.decision(), pool)). Generator = vendored reax_seed/ (MIT).
This is the only calibration source used by nv4.calibrate (FAQ: activations computed at start from the public seed set)."""
SEED = 0x52454158


def bodies(n, pool="s1-pro"):
    from reax_seed.s1contract import parse_decision
    from reax_seed.s1items import items_for_pool
    out = []
    for it in items_for_pool(pool, n, seed=SEED):
        d = parse_decision(it.decision(), pool)
        out.append({"state": d.state, "questions": {d.question.ident: d.question.author_dict()}})
    return out
