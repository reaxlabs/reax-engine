# SPDX-License-Identifier: Apache-2.0
"""Dense MLP / routed MoE overlap in Blink's decoder layers (BLINK_PAR=1).

A Gemma-4 decoder layer computes a dense MLP and a routed MoE from the same residual and adds the two normed results;
neither branch reads the other. vLLM runs them back to back on one stream: the MLP (compute-bound bf16 GEMMs), then
the router and the MoE (FP4 expert weights streamed from memory). Here the decoder layer computes the router and
launches the MoE branch (vLLM's own MoE runner: same kernels, same inputs) on a second CUDA stream before the MLP,
runs the MLP on the main stream meanwhile, and makes the main stream wait for the MoE result right before the kernel
that reads it. Every kernel sees the same inputs, so the outputs are bit-identical; inside a CUDA graph the two
branches are parallel graph branches.

The join takes the MLP's last GEMM output as an ordering input (so it is scheduled after the MLP) and the MoE inputs as
keep-alive inputs (so the compiled graph does not reuse their memory while the second stream still reads them). The
MoE result is handed from the fork to the join outside the graph (a FIFO), so the join returns a fresh tensor.
Installed by sitecustomize.py; patches Gemma4DecoderLayer.forward when vllm.model_executor.models.gemma4 is imported.
"""
import importlib.abc
import os
import sys

ON = os.environ.get("BLINK_PAR", "0") == "1"
# E64: BLINK_PAR_PRIO=1: the MoE branch's stream has high priority, so the GPU dispatches its CTAs (head glue, the
# persistent grouped-GEMM grids) before the dense MLP's queued CTAs; scheduling only, every kernel is unchanged
PRIO = os.environ.get("BLINK_PAR_PRIO", "0") == "1"
_TARGET = "vllm.model_executor.models.gemma4"
_S = {}


def _install_ops():
    import torch
    from torch.library import Library
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr
    from vllm.utils.torch_utils import direct_register_custom_op

    lib = Library("blink", "FRAGMENT")
    _S["lib"] = lib

    def streams():
        if "aux" not in _S:
            _S["aux"] = torch.cuda.Stream(priority=-1) if PRIO else torch.cuda.Stream()
            _S["ev_in"] = torch.cuda.Event()
            _S["ev_out"] = torch.cuda.Event()
            _S["fifo"] = []
        return _S

    def moe_fork(hidden_states: torch.Tensor, router_logits: torch.Tensor, layer_name: mr._layer_name_type,
                 hidden_dim_unpadded: int) -> torch.Tensor:
        s = streams()
        main = torch.cuda.current_stream()
        last = sys.modules.get("blink_last")      # E64 B7: the last layer's MoE on the rows that reach the logits
        idx = last.moe_rows(layer_name, hidden_states.shape[0]) if last is not None else None
        s["ev_in"].record(main)
        with torch.cuda.stream(s["aux"]):
            s["ev_in"].wait(s["aux"])
            if idx is None:
                out = mr._moe_forward(hidden_states, router_logits, None, None, layer_name, hidden_dim_unpadded)
                if out.untyped_storage().data_ptr() == hidden_states.untyped_storage().data_ptr():
                    out = out.clone()          # an in-place MoE result must not alias the join's inputs
            else:
                sel = mr._moe_forward(hidden_states.index_select(0, idx), router_logits.index_select(0, idx), None,
                                      None, layer_name, hidden_dim_unpadded)
                out = torch.zeros((hidden_states.shape[0],) + tuple(sel.shape[1:]), device=sel.device, dtype=sel.dtype)
                out.index_copy_(0, idx, sel)
            s["ev_out"].record(s["aux"])
        s["fifo"].append((out, idx))     # E62 C2: idx lives until the join (read on the aux stream)
        return torch.empty(1, device=hidden_states.device, dtype=torch.int8)

    def moe_fork_fake(hidden_states, router_logits, layer_name, hidden_dim_unpadded):
        return torch.empty(1, device=hidden_states.device, dtype=torch.int8)

    def moe_forward(token: torch.Tensor, after: torch.Tensor, hidden_states: torch.Tensor,
                    router_logits: torch.Tensor, hidden_dim_unpadded: int) -> torch.Tensor:
        """the join (named like vLLM's op so the fused kernels that read its result keep their names)"""
        s = streams()
        s["ev_out"].wait(torch.cuda.current_stream())
        out, _idx = s["fifo"].pop(0)     # E62 C2: idx is released only after the main stream waited for ev_out
        return out

    def moe_forward_fake(token, after, hidden_states, router_logits, hidden_dim_unpadded):
        if hidden_dim_unpadded > 0:
            return hidden_states.new_empty((*hidden_states.shape[:-1], hidden_dim_unpadded))
        return torch.empty_like(hidden_states)

    direct_register_custom_op(op_name="moe_fork", op_func=moe_fork, mutates_args=["hidden_states"],
                              fake_impl=moe_fork_fake, target_lib=lib)
    direct_register_custom_op(op_name="moe_forward", op_func=moe_forward, mutates_args=[],
                              fake_impl=moe_forward_fake, target_lib=lib,
                              tags=(torch.Tag.needs_fixed_stride_order,))


def _runner_ok(runner, hidden_size):
    """the MoE runner's own pre/post-processing around its custom op (MoERunner.forward) must be no-ops for the
    split to be exact: no shared experts, no routed input/output transform, scaling 1, no padding, no TP/EP reduce,
    no zero expert"""
    try:
        cfg = runner.moe_config
        return (runner._shared_experts is None and runner.routed_input_transform is None
                and runner.routed_output_transform is None and float(runner.routed_scaling_factor) == 1.0
                and cfg.hidden_dim == hidden_size and cfg.tp_size == 1 and cfg.ep_size == 1
                and type(runner.router).__name__ != "ZeroExpertRouter")
    except Exception as exc:  # noqa: BLE001
        print(f"[blink-par] runner check failed ({exc!r}): stock layer forward", file=sys.stderr, flush=True)
        return False


def _patch(mod):
    import torch
    if os.environ.get("BLINK_K1_GLU", "0") == "1":     # K1: fused gate_up + GELU(tanh)*up in Gemma4MLP (blink_k1.py)
        try:
            import blink_k1
            blink_k1.patch_mlp(mod)
        except Exception as exc:  # noqa: BLE001
            print(f"[blink-par] K1 GLU patch failed ({exc!r}): stock MLP", file=sys.stderr, flush=True)
    cls = mod.Gemma4DecoderLayer
    orig_forward = cls.forward
    _install_ops()

    def forward(self, positions, hidden_states, residual, per_layer_input=None, **kwargs):
        if not self.enable_moe_block or not _runner_ok(self.moe.experts, self.hidden_size):
            return orig_forward(self, positions, hidden_states, residual, per_layer_input, **kwargs)
        residual = hidden_states
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, **kwargs)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        # MoE branch first: its inputs, then the fork onto the second stream
        runner = self.moe.experts
        hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
        router_logits = self.router(residual)
        unpadded = runner.moe_config.hidden_dim_unpadded if runner._quant_method.has_unpadded_output else 0
        token = torch.ops.blink.moe_fork(hidden_states_2, router_logits, runner._encode_layer_name(), unpadded)

        # dense MLP on the main stream meanwhile
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)

        # join: after the MLP's last GEMM, keeping the MoE inputs alive until here
        moe_out = torch.ops.blink.moe_forward(token, mlp_out, hidden_states_2, router_logits, unpadded)
        hidden_states_1 = self.post_feedforward_layernorm_1(mlp_out)
        hidden_states_2 = self.post_feedforward_layernorm_2(moe_out)
        hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        if per_layer_input is not None and self.per_layer_input_gate is not None:
            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            gated_per_layer = gate * per_layer_input
            per_layer_contribution = self.per_layer_projection(gated_per_layer)
            per_layer_contribution = self.post_per_layer_input_norm(per_layer_contribution)
            hidden_states = hidden_states + per_layer_contribution
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None

    cls.forward = forward
    print("[blink-par] MoE branch on a second stream, overlapped with the dense MLP", file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = None
        for f in sys.meta_path:
            if f is self:
                continue
            try:
                spec = f.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if ON:
    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
