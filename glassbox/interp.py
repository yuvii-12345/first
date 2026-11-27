"""Mechanistic Interpretability Engine for Glassbox Transformers.

Implements genuine Transformer Circuits techniques:
1. Attention Pattern Inspection + Automatic Head Classification (Induction, Prev-Token, BOS Sink)
2. Interactive Head Ablation (Zero & Mean) with KL Divergence & Probability Shift
3. Logit Lens (Intermediate Residual Projection across Depth & Token Position)
4. Direct Logit Attribution (Exact Additive Decomposition across Embed, Heads, and MLPs)
5. Causal Activation Patching (Clean -> Corrupted Denoising Intervention Heatmap)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from glassbox.bpe import BPETokenizer
from glassbox.tensor import Tensor, no_grad
from glassbox.transformer import GPTTransformer, InterpretabilityTrace


def _softmax_list(logits: Sequence[float]) -> List[float]:
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log(max(p, 1e-12)) for p in probs if p > 1e-12)


def _kl_div(p: Sequence[float], q: Sequence[float]) -> float:
    return sum(
        pi * math.log(max(pi, 1e-12) / max(qi, 1e-12))
        for pi, qi in zip(p, q)
        if pi > 1e-12
    )


def _classify_head(
    matrix: List[List[float]], token_ids: Sequence[int]
) -> Dict[str, Any]:
    """Compute mechanistic head metrics (Induction, Prev-Token, BOS Sink, Self, Entropy)."""
    T = len(matrix)
    if T <= 1:
        return {
            "prev_token_score": 0.0,
            "self_score": 1.0,
            "bos_sink_score": 1.0,
            "induction_score": 0.0,
            "norm_entropy": 0.0,
            "role": "Single Token",
        }

    prev_sum = 0.0
    self_sum = 0.0
    bos_sum = 0.0
    ent_sum = 0.0

    for i in range(T):
        self_sum += matrix[i][i]
        if i >= 1:
            prev_sum += matrix[i][i - 1]
            bos_sum += matrix[i][0]
            row = matrix[i][: i + 1]
            max_ent = math.log(i + 1)
            ent_sum += (_entropy(row) / max_ent) if max_ent > 1e-9 else 0.0

    prev_score = prev_sum / (T - 1)
    self_score = self_sum / T
    bos_score = bos_sum / (T - 1)
    norm_ent = ent_sum / (T - 1)

    # Olsson et al. prefix-matching induction score:
    # When token_ids[i] occurred earlier at j < i - 1, how much does i attend to j + 1?
    ind_sum = 0.0
    ind_count = 0
    for i in range(2, T):
        cur_tok = token_ids[i]
        matches = [j for j in range(0, i - 1) if token_ids[j] == cur_tok]
        if matches:
            ind_sum += sum(matrix[i][j + 1] for j in matches)
            ind_count += 1
    induction_score = (ind_sum / ind_count) if ind_count > 0 else 0.0

    # Assign primary emergent role
    if ind_count > 0 and induction_score >= 0.22:
        role = "Induction"
    elif prev_score >= 0.32:
        role = "Prev-Token"
    elif bos_score >= 0.45:
        role = "BOS Sink"
    elif self_score >= 0.45:
        role = "Self-Attend"
    elif norm_ent < 0.42:
        role = "Sharp Focus"
    else:
        role = "Broad Context"

    return {
        "prev_token_score": round(prev_score, 4),
        "self_score": round(self_score, 4),
        "bos_sink_score": round(bos_score, 4),
        "induction_score": round(induction_score, 4),
        "induction_opportunities": ind_count,
        "norm_entropy": round(norm_ent, 4),
        "role": role,
    }


def _extract_last_logits(logits: Tensor, V: int, T_len: int) -> List[float]:
    off = (T_len - 1) * V
    return [float(logits._buf[off + v]) for v in range(V)]


def _top_predictions(
    logits: Sequence[float], probs: Sequence[float], tokenizer: BPETokenizer, k: int = 10
) -> List[Dict[str, Any]]:
    ranked = sorted(enumerate(probs), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        {
            "id": tid,
            "display": tokenizer.token_display(tid),
            "text": tokenizer.token_text(tid),
            "prob": round(float(p), 5),
            "logit": round(float(logits[tid]), 4),
        }
        for tid, p in ranked
    ]


def analyze_attention_and_ablation(
    model: GPTTransformer,
    tokenizer: BPETokenizer,
    prompt: str,
    ablate_heads: Optional[Sequence[Sequence[int]]] = None,
    ablation_mode: str = "zero",
) -> Dict[str, Any]:
    """Capture all attention maps, classify heads, and optionally run live head ablation."""
    ids = tokenizer.encode(prompt, add_bos=True)
    if len(ids) > model.cfg.context_len:
        ids = ids[-model.cfg.context_len :]
    T_len = len(ids)
    V = model.cfg.vocab_size

    tokens = [
        {"pos": i, "id": tid, "display": tokenizer.token_display(tid), "text": tokenizer.token_text(tid)}
        for i, tid in enumerate(ids)
    ]

    with no_grad():
        base_trace = InterpretabilityTrace(capture_attn=True)
        base_logits_t, _ = model.forward([ids], trace=base_trace)
        base_logits = _extract_last_logits(base_logits_t, V, T_len)
        base_probs = _softmax_list(base_logits)
        base_entropy = _entropy(base_probs)
        base_top = _top_predictions(base_logits, base_probs, tokenizer, k=10)

        layers_out = []
        for l in range(model.cfg.n_layers):
            heads_out = []
            layer_mats = base_trace.attn_probs[l]
            for h in range(model.cfg.n_heads):
                mat = layer_mats[h]
                stats = _classify_head(mat, ids)
                # Round matrix cleanly for JSON transport
                rounded_mat = [[round(val, 4) for val in row] for row in mat]
                heads_out.append(
                    {
                        "layer": l,
                        "head": h,
                        "matrix": rounded_mat,
                        **stats,
                    }
                )
            layers_out.append({"layer": l, "heads": heads_out})

        ablation_result = None
        ablate_set: Set[Tuple[int, int]] = set()
        if ablate_heads:
            for item in ablate_heads:
                if len(item) == 2:
                    ablate_set.add((int(item[0]), int(item[1])))

        if ablate_set:
            abl_trace = InterpretabilityTrace(
                capture_attn=False,
                ablate_heads=ablate_set,
                ablation_mode=ablation_mode,
            )
            abl_logits_t, _ = model.forward([ids], trace=abl_trace)
            abl_logits = _extract_last_logits(abl_logits_t, V, T_len)
            abl_probs = _softmax_list(abl_logits)
            abl_entropy = _entropy(abl_probs)
            abl_top = _top_predictions(abl_logits, abl_probs, tokenizer, k=10)
            kl = _kl_div(base_probs, abl_probs)

            # Compare shifts on union of top baseline predictions
            comparisons = []
            for pred in base_top[:8]:
                tid = pred["id"]
                comparisons.append(
                    {
                        "id": tid,
                        "display": pred["display"],
                        "base_prob": round(base_probs[tid], 5),
                        "ablated_prob": round(abl_probs[tid], 5),
                        "delta_prob": round(abl_probs[tid] - base_probs[tid], 5),
                        "delta_logit": round(abl_logits[tid] - base_logits[tid], 4),
                    }
                )

            ablation_result = {
                "ablated_heads": [[l, h] for (l, h) in sorted(ablate_set)],
                "mode": ablation_mode,
                "kl_divergence": round(kl, 5),
                "base_entropy": round(base_entropy, 4),
                "ablated_entropy": round(abl_entropy, 4),
                "top_predictions": abl_top,
                "comparisons": comparisons,
            }

    return {
        "prompt": prompt,
        "tokens": tokens,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "baseline": {
            "entropy": round(base_entropy, 4),
            "top_predictions": base_top,
        },
        "layers": layers_out,
        "ablation": ablation_result,
    }


def analyze_logit_lens_and_dla(
    model: GPTTransformer,
    tokenizer: BPETokenizer,
    prompt: str,
    target_token: Optional[str] = None,
    foil_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute full Layer x Position Logit Lens + Direct Logit Attribution at final position."""
    ids = tokenizer.encode(prompt, add_bos=True)
    if len(ids) > model.cfg.context_len:
        ids = ids[-model.cfg.context_len :]
    T_len = len(ids)
    V = model.cfg.vocab_size
    D = model.cfg.d_model

    tokens = [
        {"pos": i, "id": tid, "display": tokenizer.token_display(tid), "text": tokenizer.token_text(tid)}
        for i, tid in enumerate(ids)
    ]

    with no_grad():
        trace = InterpretabilityTrace(capture_residuals=True, capture_dla=True)
        final_logits_t, _ = model.forward([ids], trace=trace)

        # Compute final distribution at every position for rank/KL comparisons
        final_probs_by_pos: List[List[float]] = []
        final_top1_by_pos: List[int] = []
        for pos in range(T_len):
            row_lg = [float(final_logits_t._buf[pos * V + v]) for v in range(V)]
            row_pr = _softmax_list(row_lg)
            final_probs_by_pos.append(row_pr)
            final_top1_by_pos.append(max(range(V), key=lambda v: row_pr[v]))

        # Project every captured residual stage through final_norm + W_U
        stages_out = []
        for stage_label, resid_tensor in trace.residuals:
            stage_logits = model.project_to_logits(resid_tensor)
            cells = []
            for pos in range(T_len):
                off = pos * V
                row_lg = [float(stage_logits._buf[off + v]) for v in range(V)]
                row_pr = _softmax_list(row_lg)
                ent = _entropy(row_pr)

                ranked = sorted(enumerate(row_pr), key=lambda kv: kv[1], reverse=True)
                top1_id, top1_p = ranked[0]
                final_tid = final_top1_by_pos[pos]
                final_rank = 1
                for r_idx, (tid, _) in enumerate(ranked):
                    if tid == final_tid:
                        final_rank = r_idx + 1
                        break

                kl_to_final = _kl_div(final_probs_by_pos[pos], row_pr)
                cells.append(
                    {
                        "pos": pos,
                        "top1_id": top1_id,
                        "top1_display": tokenizer.token_display(top1_id),
                        "top1_prob": round(top1_p, 5),
                        "entropy": round(ent, 4),
                        "final_token_prob": round(row_pr[final_tid], 5),
                        "final_token_rank": final_rank,
                        "kl_to_final": round(kl_to_final, 4),
                        "top6": [
                            {
                                "id": tid,
                                "display": tokenizer.token_display(tid),
                                "prob": round(p, 5),
                            }
                            for tid, p in ranked[:6]
                        ],
                    }
                )
            stages_out.append({"stage": stage_label, "cells": cells})

        # ---- Direct Logit Attribution (DLA) at final position T_len - 1 ----
        last_pos = T_len - 1
        last_logits = _extract_last_logits(final_logits_t, V, T_len)
        last_probs = final_probs_by_pos[last_pos]
        ranked_last = sorted(enumerate(last_logits), key=lambda kv: kv[1], reverse=True)

        # Resolve target & optional foil token IDs
        if target_token:
            enc_t = tokenizer.encode(target_token, add_bos=False)
            target_id = enc_t[0] if enc_t else ranked_last[0][0]
        else:
            target_id = ranked_last[0][0]

        foil_id: Optional[int] = None
        if foil_token:
            enc_f = tokenizer.encode(foil_token, add_bos=False)
            if enc_f and enc_f[0] != target_id:
                foil_id = enc_f[0]

        # Compute exact final RMSNorm scaling vector at last_pos
        pre_norm = trace.final_pre_norm
        base_off = last_pos * D
        ss = sum(float(pre_norm._buf[base_off + d]) ** 2 for d in range(D))
        inv_rms = 1.0 / math.sqrt(ss / D + model.cfg.rmsnorm_eps)
        gamma = [float(model.final_norm.weight._buf[d]) for d in range(D)]
        w_u = model.unembed_weight()._buf

        # Effective unembedding direction u_dir = (W_U[target] - W_U[foil]) * gamma * inv_rms
        u_dir = [0.0] * D
        t_off = target_id * D
        f_off = (foil_id * D) if foil_id is not None else -1
        for d in range(D):
            w_val = float(w_u[t_off + d]) - (float(w_u[f_off + d]) if f_off >= 0 else 0.0)
            u_dir[d] = w_val * gamma[d] * inv_rms

        def _dot_last(comp_tensor: Tensor) -> float:
            off = last_pos * D
            return sum(float(comp_tensor._buf[off + d]) * u_dir[d] for d in range(D))

        dla_components: List[Dict[str, Any]] = []
        if trace.embed_contrib is not None:
            dla_components.append(
                {"label": "Embed", "kind": "embed", "layer": -1, "contrib": round(_dot_last(trace.embed_contrib), 5)}
            )
        for l in range(model.cfg.n_layers):
            for h in range(model.cfg.n_heads):
                hc = trace.head_contribs.get((l, h))
                if hc is not None:
                    dla_components.append(
                        {
                            "label": f"L{l}.H{h}",
                            "kind": "head",
                            "layer": l,
                            "head": h,
                            "contrib": round(_dot_last(hc), 5),
                        }
                    )
            mc = trace.mlp_contribs.get(l)
            if mc is not None:
                dla_components.append(
                    {
                        "label": f"L{l}.MLP",
                        "kind": "mlp",
                        "layer": l,
                        "contrib": round(_dot_last(mc), 5),
                    }
                )

        actual_val = last_logits[target_id] - (last_logits[foil_id] if foil_id is not None else 0.0)
        reconstructed_val = sum(c["contrib"] for c in dla_components)

    return {
        "prompt": prompt,
        "tokens": tokens,
        "stages": stages_out,
        "final_top_predictions": _top_predictions(last_logits, last_probs, tokenizer, k=8),
        "dla": {
            "target_id": target_id,
            "target_display": tokenizer.token_display(target_id),
            "target_prob": round(last_probs[target_id], 5),
            "foil_id": foil_id,
            "foil_display": tokenizer.token_display(foil_id) if foil_id is not None else None,
            "actual_logit": round(actual_val, 5),
            "sum_of_contribs": round(reconstructed_val, 5),
            "components": dla_components,
        },
    }


def run_causal_activation_patching(
    model: GPTTransformer,
    tokenizer: BPETokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    clean_target: Optional[str] = None,
    corrupted_target: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Denoising Causal Activation Patching across every (layer, token_pos).

    Patches clean residual stream activations into the corrupted forward pass and measures
    fractional recovery of (logit(clean_target) - logit(corrupted_target)).
    """
    clean_ids = tokenizer.encode(clean_prompt, add_bos=True)
    corrupt_ids = tokenizer.encode(corrupted_prompt, add_bos=True)

    max_len = model.cfg.context_len
    if len(clean_ids) > max_len:
        clean_ids = clean_ids[-max_len:]
    if len(corrupt_ids) > max_len:
        corrupt_ids = corrupt_ids[-max_len:]

    # Right-align if tokenization lengths differ slightly
    length_aligned = len(clean_ids) == len(corrupt_ids)
    if not length_aligned:
        min_l = min(len(clean_ids), len(corrupt_ids))
        clean_ids = clean_ids[-min_l:]
        corrupt_ids = corrupt_ids[-min_l:]

    T_len = len(clean_ids)
    V = model.cfg.vocab_size
    D = model.cfg.d_model

    with no_grad():
        # 1. Clean run with residual capture
        clean_trace = InterpretabilityTrace(capture_residuals=True)
        clean_logits_t, _ = model.forward([clean_ids], trace=clean_trace)
        clean_logits = _extract_last_logits(clean_logits_t, V, T_len)
        clean_probs = _softmax_list(clean_logits)

        # 2. Corrupted run
        corrupt_logits_t, _ = model.forward([corrupt_ids])
        corrupt_logits = _extract_last_logits(corrupt_logits_t, V, T_len)
        corrupt_probs = _softmax_list(corrupt_logits)

        # Resolve target and foil IDs
        clean_ranked = sorted(enumerate(clean_logits), key=lambda kv: kv[1], reverse=True)
        corrupt_ranked = sorted(enumerate(corrupt_logits), key=lambda kv: kv[1], reverse=True)

        if clean_target:
            enc = tokenizer.encode(clean_target, add_bos=False)
            t_clean = enc[0] if enc else clean_ranked[0][0]
        else:
            t_clean = clean_ranked[0][0]

        if corrupted_target:
            enc = tokenizer.encode(corrupted_target, add_bos=False)
            t_corrupt = enc[0] if enc else corrupt_ranked[0][0]
        else:
            t_corrupt = corrupt_ranked[0][0] if corrupt_ranked[0][0] != t_clean else corrupt_ranked[1][0]

        ld_clean = clean_logits[t_clean] - clean_logits[t_corrupt]
        ld_corrupt = corrupt_logits[t_clean] - corrupt_logits[t_corrupt]
        denom = ld_clean - ld_corrupt

        # Extract residual tensors at Embed (-1) and each block output (0..n_layers-1)
        stage_tensors: List[Tuple[int, str, Tensor]] = []
        for label, tensor in clean_trace.residuals:
            if label == "Embed":
                stage_tensors.append((-1, "Embed", tensor))
            elif label.endswith(".out"):
                l_idx = int(label[1 : label.index(".")])
                stage_tensors.append((l_idx, f"Block {l_idx}", tensor))

        # 3. Sweep every (stage, pos) patching clean vector into corrupted run
        rows = []
        top_hotspots = []
        for stage_idx, stage_label, resid_t in stage_tensors:
            row_recovery = []
            row_ld = []
            for pos in range(T_len):
                off = pos * D
                vec = [float(resid_t._buf[off + d]) for d in range(D)]
                patch_trace = InterpretabilityTrace(patch_residual={(stage_idx, pos): vec})
                p_logits_t, _ = model.forward([corrupt_ids], trace=patch_trace)
                p_logits = _extract_last_logits(p_logits_t, V, T_len)
                ld_patched = p_logits[t_clean] - p_logits[t_corrupt]
                if abs(denom) > 1e-5:
                    rec = (ld_patched - ld_corrupt) / denom
                else:
                    rec = ld_patched - ld_corrupt
                rec_clamped = max(-1.5, min(2.0, rec))
                row_recovery.append(round(rec_clamped, 4))
                row_ld.append(round(ld_patched, 4))
                top_hotspots.append(
                    {
                        "stage": stage_label,
                        "pos": pos,
                        "clean_tok": tokenizer.token_display(clean_ids[pos]),
                        "corrupt_tok": tokenizer.token_display(corrupt_ids[pos]),
                        "recovery": round(rec_clamped, 4),
                        "logit_diff": round(ld_patched, 4),
                    }
                )
            rows.append(
                {
                    "stage_idx": stage_idx,
                    "stage_label": stage_label,
                    "recovery": row_recovery,
                    "logit_diff": row_ld,
                }
            )

        top_hotspots.sort(key=lambda x: x["recovery"], reverse=True)

    return {
        "length_aligned": length_aligned,
        "clean_tokens": [
            {"pos": i, "id": tid, "display": tokenizer.token_display(tid)}
            for i, tid in enumerate(clean_ids)
        ],
        "corrupt_tokens": [
            {"pos": i, "id": tid, "display": tokenizer.token_display(tid)}
            for i, tid in enumerate(corrupt_ids)
        ],
        "clean_target": {
            "id": t_clean,
            "display": tokenizer.token_display(t_clean),
            "clean_prob": round(clean_probs[t_clean], 5),
            "corrupt_prob": round(corrupt_probs[t_clean], 5),
        },
        "corrupted_target": {
            "id": t_corrupt,
            "display": tokenizer.token_display(t_corrupt),
            "clean_prob": round(clean_probs[t_corrupt], 5),
            "corrupt_prob": round(corrupt_probs[t_corrupt], 5),
        },
        "logit_diff_clean": round(ld_clean, 4),
        "logit_diff_corrupt": round(ld_corrupt, 4),
        "rows": rows,
        "top_hotspots": top_hotspots[:8],
    }
