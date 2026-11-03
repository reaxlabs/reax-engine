# SPDX-License-Identifier: MIT
"""Protect Triton's shared host autotuner bookkeeping; GPU streams remain independent."""
import functools
import threading

_LAUNCH_LOCK = threading.RLock()


def install():
    from triton.runtime.autotuner import Autotuner
    if getattr(Autotuner.run, '_reax_thread_safe', False):
        return
    original = Autotuner.run

    @functools.wraps(original)
    def guarded(self, *args, **kwargs):
        # run() mutates self.nargs/cache/best_config and clears nargs on return.
        # Lock only host dispatch/tuning, not the asynchronously launched GPU work.
        with _LAUNCH_LOCK:
            return original(self, *args, **kwargs)

    guarded._reax_thread_safe = True
    Autotuner.run = guarded


# Part Q's answers depend on these configs (measured), and stock autotune
# chooses them per process by timing. One config per kernel makes Part Q's answers independent of the process.
_ANSWER_CONFIGS = (
    ('fla.modules.l2norm', 'l2norm_fwd_kernel', {'BT': 16, 'num_warps': 8}),
    ('fla.ops.common.chunk_o', 'chunk_fwd_kernel_o', {'BK': 32, 'BV': 32}),
    ('fla.ops.common.chunk_delta_h', 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64', {'BV': 32, 'num_stages': 2}),
)


def pin_answer_configs():
    import importlib
    from triton.runtime.autotuner import Autotuner
    for module, name, wanted in _ANSWER_CONFIGS:
        tuner = getattr(importlib.import_module(module), name)
        while not isinstance(tuner, Autotuner):
            tuner = tuner.fn
        hits = [c for c in tuner.configs if all(c.all_kwargs().get(k) == v for k, v in wanted.items())]
        if not hits:
            raise RuntimeError(f'answer config {name} {wanted} not in the kernel grid')
        tuner.configs = hits[:1]
        tuner.cache.clear()
