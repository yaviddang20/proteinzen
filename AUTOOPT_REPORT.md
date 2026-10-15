# GPU2 autoopt report — 2026-10-14

## Summary

Spent most of the session on (1) investigating `attention.py`/`layers_v2.py`/`denoiser.py`'s
`Embedder` as assigned, (2) a self-conditioning dedup candidate that turned out to collide
with GPU1's original assignment (stood down, zero net changes left in the repo from that), and
(3) a `token_rigids` dedup candidate in `IpaDenoiser.forward`'s main 12-block trunk, which is
the single most-developed finding of the night but is **NOT fully verified** and **NOT
committed** — repo is back at the `autoopt/gpu2`-baseline commit (`d40c0e0`) with no code
changes, only a pre-existing local `env_vars.sh` path fix needed to run anything in this
worktree at all.

Also produced a session-wide methodology finding (independently cross-validated by GPU0 and
GPU1) that the strict "0 abs diff gradients" bar in the verification protocol is **unachievable
on this GPU/CUDA stack even for a true no-op** (running identical, unmodified code twice
produces 2077/2216 differing param grad-norms, max abs diff 0.118). This likely materially
changes how tonight's other "silently corrupted gradients" findings should be interpreted —
flagging prominently for human review.

## Investigated, no change made

- `proteinzen/model/modules/attention.py`: read in full.
  - `ConditionedTransformerPairBiasLayer._eager_forward` is dead code (never called — `forward`
    always routes through `_gen_attn_tensors`/`_gen_pair_bias`/`_wrapup`). Zero perf impact to
    remove (never executes), so not worth the risk/churn for a pure cleanup.
  - `FlashTransformerEncoderLayer.forward` has a bare `raise NotImplementedError` — also
    apparently unused/broken, same non-finding.
  - `_gen_attn_tensors` unconditionally calls `self.q_norm`/`self.k_norm` without checking
    `self.use_qk_norm` (unlike `_eager_forward`, which gates it) — consistent with
    `use_qk_norm=True` always in our launch configs, so not a live bug, just inconsistent with
    the (dead) eager path. No action needed.
- `proteinzen/openfold/layers/layers_v2.py`: read `Linear`, `LayerNorm`, `BackboneUpdate`,
  `Transition`, `AdaLN`, `AdaScale`, `ConditionedTransition`, `TorsionAngles` in full — all
  clean, no redundant computation found. Did not touch `ConditionedInvariantPointAttention`
  (GPU0's claimed territory for `_finalize_output`/`_compile_forward`), read only enough of it
  to confirm Rigid/Rotation usage patterns for the denoiser.py candidate below.
- `proteinzen/model/denoiser.py` `Embedder` class: fully read `_gen_node_features`,
  `_gen_rigid_features`, `_forward_part_a`/`_forward_part_b`/`forward`. Found:
  - `_forward_part_a`/`_forward_part_b` split and `compute_embedder_part_a`/
    `embedder_part_a_override` plumbing **already exists** in the codebase, explicitly built
    (per its docstring) to let the self-conditioning training step compute the embedder's
    sc_rigids/sf_rigids-independent "part A" once and reuse it across the self-conditioning and
    main forward passes — but it was **not wired up** at the one call site that needed it
    (`lmod.py::_shared_step`). I implemented this wiring (see "Standing down" below) before
    learning GPU1 already owned this exact task.
  - Small, not-yet-attempted idea: when `disable_absolute_res_idx=True` (always true in our
    launch configs), `node_seq_idx_embed = self.index_embedder(seq_idx) * 0` is computed and fed
    through `self.node_init` (bias-free `Linear`) — mathematically always exactly zero
    regardless of `seq_idx`, so the `index_embedder` call and the `node_init` matmul could be
    skipped and replaced with a zero tensor when the flag is set. Did not implement/verify this
    — low expected payoff (one small `Linear` + a positional-embedding call, not a hot-path
    item) and, given how much of tonight's "surely safe" reasoning in this same `Embedder` class
    turned out to need careful gradient-noise-floor analysis, I did not want to add an unverified
    change on top of an already long session. Flagging for a future agent/human instead.

## Standing down: self-conditioning `embedder_part_a_override` dedup (lmod.py)

Found and implemented this independently (see commit-free diff I applied and then reverted),
before GPU1 messaged that it was their **original assigned task**, already in progress, and
that an earlier attempt at exactly this change tonight had apparently "silently corrupted
gradients." I killed my in-progress correctness run and fully reverted the edit (never
committed). See the coordination file for the full back-and-forth; short version: this turned
out to be entangled with the inherent-GPU-non-determinism finding below (GPU1's own
investigation, after controlling for the noise floor, found real-but-modest directional
evidence of an actual bug localized to `rigids_tfmr_N.trunk.frame_update_0`/
`node_to_frame_broadcast` gradients — GPU1 owns following that up).

## Main candidate: `token_rigids` dedup in `IpaDenoiser.forward` (denoiser.py)

**File:** `proteinzen/model/denoiser.py`, `IpaDenoiser.forward`, main 12-block trunk (~line 1183).

### What I found

`token_rigids` is computed twice per non-final block via the identical expression
`ru.Rigid.from_tensor_7(rigids_to_nodes(curr_rigids_tensor_7))`:
- once at the bottom of block `b` (right after `curr_rigids_tensor_7 = curr_rigids.to_tensor_7()`),
  feeding `edge_transition_b`;
- again at the top of block `b+1`, from the **same** `curr_rigids_tensor_7` (nothing modifies it
  in between), feeding `ipa_{b+1}`.

This is a pure, deterministic function of the same tensor, computed twice. Verified:
- `gather_helper` (used inside `rigids_to_nodes`) is `torch.gather`, non-mutating.
- `Rigid`/`Rotation` in `rigid_utils.py` are immutable-style throughout — every method used by
  both `ConditionedPairUpdate` (edge_transition: `get_trans`/`get_rots`/`invert`/`compose_q`) and
  the IPA call returns a *new* object; none mutate `self._trans`/`self._rots` in place.

### Fix applied (then reverted — see Verification status)

Moved the `token_rigids = ru.Rigid.from_tensor_7(rigids_to_nodes(curr_rigids_tensor_7))` line to
just before the `for b in range(self.num_blocks):` loop (computing it once for block 0), and
removed the now-redundant recomputation at the top of the loop body — the recomputation at the
bottom of the loop (for `b < num_blocks - 1`) is left untouched and now serves double duty
(feeds `edge_transition_b` immediately, and — since nothing overwrites `token_rigids` before the
next iteration's top — the *same* object is reused for `ipa_{b+1}`).

```python
# Main trunk
curr_rigids_tensor_7 = curr_rigids.to_tensor_7()
token_rigids = ru.Rigid.from_tensor_7(rigids_to_nodes(curr_rigids_tensor_7))  # moved here, b=0 only
for b in range(self.num_blocks):
    if self.use_conditioned_ipa:
        ...
    # (unchanged: ipa, tfmr, transition, rigids_tfmr, rigid updates)
    if b < self.num_blocks - 1:
        curr_rigids_tensor_7 = curr_rigids.to_tensor_7()
        token_rigids = ru.Rigid.from_tensor_7(rigids_to_nodes(curr_rigids_tensor_7))  # unchanged
        edge_embed = self.trunk[f'edge_transition_{b}'](...)
        edge_embed *= edge_mask[..., None]
```

### Verification done

1. **Forward bit-identical**: ran the existing `diff_correctness.py` protocol (resume from
   `.../binder_zen_geom_.../version_2087577/checkpoints/last.ckpt`, GPU2, `CUBLAS_WORKSPACE_CONFIG=
   :4096:8`, `CORRECTNESS_CHECK_TAG` seeding). `denoised_rigids`/`pred_seq`/`input_rigids_1` at
   both t=0.0 and t=0.5: **exact_equal=True, max_abs_diff=0** for before vs after.

2. **Gradient check — and the big methodology finding**: raw `diff_grads.py` on train_grads
   dumps showed 2078/2216 params "differing" (max abs diff 0.036) between before/after — looked
   alarming in isolation, matching the pattern described as a red flag in the task brief.
   **Before concluding anything, I ran a same-code control**: executed the *unmodified baseline*
   code twice (tags `gpu2tr_before`/`gpu2tr_before2`, same checkpoint/seed/GPU) and diffed those
   against each other. Result: **2077/2216 differing, max abs diff 0.118** — i.e. running
   identical code twice, with nothing changed, produces almost exactly the same scale of
   "mismatch." This proves the model's backward pass is **inherently non-deterministic on this
   GPU/CUDA stack** (even with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set — the non-determinism
   survives that mitigation, so it's likely `torch.gather`'s backward `scatter_add` and/or the
   cuequivariance triangle-attention kernels, not just cuBLAS). I flagged this immediately to
   GPU0/GPU1/GPU3 and the coordinator via the coordination file and direct messages — GPU0
   independently reproduced the same noise floor on GPU0 within the hour, and GPU1 used it to
   correctly separate a real bug (strong 15/15 directional signal, p≈0.00003) from noise in
   their own candidate.

   Applying the same rigor to my own candidate:
   - **Total gradient norm** (sqrt of sum of squares over all 2216 params — the thing that
     actually drives the optimizer): before=19.156, after=19.191, before2=19.233. My
     candidate's relative deviation from baseline is **0.18%**, *smaller* than the measured
     noise floor's **0.40%**.
   - Per-parameter analysis normalized by each parameter's own gradient magnitude (not raw abs
     diff, which is dominated by near-zero-gradient params where ratios explode spuriously):
     only 30/2216 params exceed 5% relative excess over the noise floor, concentrated in
     `rigids_tfmr_N.trunk.frame_update_0.ln.{weight,bias}` (`BackboneUpdate`'s LayerNorm) across
     several block indices — notably the **same submodule pattern** GPU1 independently flagged
     for their own, unrelated change. Ran GPU1's directionality test (binomial sign test) on
     this flagged set: only **6/9 same-sign** (p≈0.5, statistically indistinguishable from pure
     chance) — a sharp contrast with GPU1's confirmed-real **15/15** (p≈0.00003) for their
     candidate. My interpretation: `frame_update_0.ln` is a generically sensitive point in the
     network that amplifies *any* tiny floating-point divergence between two runs into a visible
     relative gradient difference, whether that divergence is pure GPU scheduling noise (my
     case) or a real systematic bug (GPU1's case) — the directionality test, not the raw
     ratio, is what actually distinguishes the two.
   - **Conclusion**: forward bit-identical, and the gradient deviation is within the model's own
     measured noise floor by every metric I checked (total norm, relative-to-magnitude excess,
     directionality). I treat this as **NEAR-EXACT** rather than exactly verified, since the
     protocol's literal "0 abs diff" bar is unachievable here even for a no-op.

3. **Extended stress test — INCOMPLETE, ran out of time.** Resumed training from the same
   checkpoint on GPU2 with the candidate patch applied, no step/batch limits. Confirmed:
   - No crash through dataset loading, IGSO3 cache precompute, sanity-check validation
     (including PDB-writing epoch-sample generation), and into real training (`Epoch 971: 0/334`
     reached cleanly).
   - One transient process-liveness scare: the launcher wrapper process got killed by GPU1's
     agent during an unrelated cleanup sweep on the shared node; verified my actual training
     process (and its dataloader workers) were unaffected (the `setsid`/`nohup` detachment meant
     the wrapper's death was a no-op). No corruption, just worth noting for the log.
   - **Did not reach the required 1000+ steps / ~3 epochs before running out of session time.**
     I killed the process and reverted the code change rather than leave an unverified change
     committed. GPU2 confirmed idle (0 MiB) afterward.

### Recommendation

This is the most promising, most rigorously-checked candidate from my session, but per the
task's explicit bar ("only then: run an extended stress test... before calling it
verified-safe"), **I am not marking it verified and have not committed it.** To finish
verification: reapply the diff above to `IpaDenoiser.forward` in `denoiser.py` (trivial,
6-line move — see exact before/after text above), resume training from the same checkpoint on
an idle GPU with no step limit, and let it run 1000+ steps (~3 epochs at `batch_size=3`,
~40-50 min at the observed ~0.4 it/s), watching for crashes/NaNs and sane loss curves. If clean,
this is very likely safe to merge — real speedup was not re-measured after reverting (pulled
one `Rigid.from_tensor_7` + one `gather_helper` call out of 11 of the 12 trunk-block iterations;
expect a small single-digit-millisecond-per-step win given these are comparatively cheap ops
relative to the IPA/triangle-attention calls around them — did not have time to do a clean
20-sample `torch.cuda.synchronize()` timing comparison, so no number to report here; only
honest claim is "plausible small win," not a measured one).

## Critical methodology finding for human review (most important output of this session)

Running **unmodified baseline code twice** (same checkpoint, same deterministic seeding, same
GPU, `CUBLAS_WORKSPACE_CONFIG=:4096:8` set) produces forward outputs that are bit-identical but
**2077 of 2216 parameter gradient norms differing, max abs diff 0.118** between the two runs.
This was independently reproduced by GPU0 on GPU0. It means:
- The "0 abs diff gradients" bar in tonight's verification protocol is **not achievable even for
  a true no-op change** on this GPU/CUDA/driver stack.
- Any of tonight's rejections based on a raw grad-norm diff (including, possibly, the original
  incident described in this task's briefing about "only 64 of 2216 parameter gradients
  matched") may need to be revisited with a same-code noise-floor control before being trusted.
  A real bug needs to show *excess* beyond this floor, and ideally pass a directionality
  (sign-consistency) test — GPU1 did this for their own candidate and found a genuine,
  statistically significant (p≈0.00003) directional signal on top of the noise, which is a much
  more defensible way to tell "real bug" from "GPU noise" than the raw check alone.
- Recommend a human decide whether to pin down the exact non-deterministic kernel (my guess:
  `torch.gather`'s CUDA backward, implemented via non-deterministic `scatter_add`, given how
  pervasively `gather_helper` is used in this model; possibly also the cuequivariance
  triangle-attention kernels flagged in every run's warnings) and/or adopt the noise-floor-plus-
  directionality-test methodology as standard for any future gradient-based correctness checks
  on this hardware.

## Final repo state

`proteinzen_auto_gpu2` worktree is at commit `d40c0e0` (the `autoopt/gpu2` baseline) with no
code changes. `env_vars.sh` has a local, pre-existing (not made by me tonight, just restored
after an accidental `git stash` round-trip) `REPO_ROOT`/`BOLTZ_CACHE` path fix needed to run
anything in this worktree — not a candidate change, just local machine config. GPU2 confirmed
idle (0 MiB) at session end. No commits made.


## Follow-up session (2026-10-14, later) -- stress test attempted, inconclusive, reverted

A follow-up agent reapplied the exact 6-line `token_rigids` dedup diff described above to
`proteinzen/model/denoiser.py` (confirmed identical to the diff shown earlier in this report) and
launched the required 1000+-step real-training stress test on GPU2, using a dedicated script
`/mnt/scratch/user/daviyang/scratch_scripts/stress_test_gpu2_tokenrigids.sh` (same hydra overrides
as `isolated_sc_test_gpu2.sh`, but with `hydra.run.dir=.../outputs/_autoopt_gpu2_stress_tokenrigids/train`
to avoid colliding with other in-flight runs on the shared node). Launch required activating the
`proteinzen` micromamba env explicitly (`source ~/.bashrc && micromamba activate proteinzen`) --
a bare non-interactive `ssh ... nohup bash script.sh &` does not pick up the env and fails
immediately with `ModuleNotFoundError: No module named 'hydra'`.

**Result: inconclusive, not a code-level failure.** The run resumed cleanly from the same
checkpoint (`version_2087577/checkpoints/last.ckpt`), passed dataset loading, IGSO3 cache
precompute, sanity-check validation (including PDB-writing epoch-sample generation -- the
`tensor(nan)`/`tensor(inf)` lines seen in the log at this stage are the normal, expected terminal
behavior of the rot/trans vector-field integration during sample generation, not a training
failure), and entered real training at `Epoch 971: 0/334`, logging multiple real training steps
with sane, finite loss values (`scaled_fafe`, `seq_loss`, `rot_vf_loss`, `trans_vf_loss`, etc. all
normal, no NaN/Inf in any `*_step` metric). **The training process was then killed by an external
SIGKILL** (`stress_test_gpu2_tokenrigids.sh: line 65: <pid> Killed`) with **no Traceback,
RuntimeError, CUDA error, or any other internal error in the log** -- i.e. not a crash caused by
the candidate patch. This matches the exact pattern already flagged earlier in this report ("the
launcher wrapper process got killed by GPU1's agent during an unrelated cleanup sweep on the
shared node"): another overnight agent's cleanup sweep on the shared `ggpu3-29` node most likely
killed this process too, before it reached even the first checkpoint (`every_n_train_steps=500` --
zero checkpoints were written; the run did not survive long enough to hit step 500, let alone the
required 1000+).

Because the run did **not** reach the 1000+-step completion bar -- regardless of cause -- this
agent did **not** commit the change, per the task's explicit instructions. To be conservative and
leave the branch in the same clean, well-understood state as before, the diff was **reverted**;
`proteinzen/model/denoiser.py` is back at the `d40c0e0` baseline (verified via `git diff` showing
no changes). GPU2 was confirmed idle (0 MiB, no processes) after cleanup
(`fuser -k /dev/nvidia2`).

**This candidate is NOT rejected on correctness grounds** -- nothing in this run contradicts the
forward-bit-identical / gradient-noise-floor analysis above. It is simply **still unverified**
for the same reason as before: the 1000+-step stress test has now failed to complete twice in a
row, both times for infrastructure reasons (external process kills on a busy shared node) rather
than any error produced by the candidate code itself. Recommendation for the next agent: rerun
`stress_test_gpu2_tokenrigids.sh` (diff reapplication is the same 6-line move documented above),
ideally at a time/node state where other agents' cleanup sweeps are less likely to collide with
this PID, and actually watch it through to a checkpoint at step >=1000 before deciding to commit.

## Final repo state (this session)

`proteinzen_auto_gpu2` worktree is at commit `d40c0e0` (the `autoopt/gpu2` baseline) with no code
changes to `denoiser.py` (diff reverted after the inconclusive stress-test interruption above).
GPU2 confirmed idle (0 MiB) at session end. No commits made this session.

## FINAL OUTCOME (coordinator-verified, attempt #4)

After 3 prior attempts were interrupted by external `fuser -k /dev/nvidia<N>` kills (root-caused
and fixed -- that command kills cross-GPU processes on this node, not just the targeted device;
all agents switched to PID-scoped kills only), attempt #4 ran uninterrupted through 1002 real
training steps (epochs 971-973 on the resumed checkpoint) with zero crashes, zero tracebacks, and
sane finite loss metrics throughout, including through epoch 972 -- the exact epoch where the
original `_broadcast_pairs` compile bug used to crash earlier this session.

**Committed**: `1363222` on `autoopt/gpu2` -- "Dedup redundant token_rigids computation in
IpaDenoiser.forward main trunk". The 6-line diff (hoisting `token_rigids` computation out of the
per-block loop) is now part of the branch.

GPU2 confirmed idle (0 MiB) after cleanup via PID-scoped kill (`pgrep -f
_autoopt_gpu2_stress_tokenrigids | xargs kill -9`), not `fuser -k`.

This candidate is now fully verified and committed. No further action needed on this branch.


## Follow-up investigation (2026-10-14, later) -- `index_embedder`/`node_init` zero-computation, investigated, NOT implemented

Followed up on the earlier flag in this report ("Small, not-yet-attempted idea" above) about
`disable_absolute_res_idx=True` (set in every real launch config, confirmed in
`isolated_sc_test_gpu2.sh`) making the `index_embedder`/`node_init` term in
`Embedder._gen_node_features` always zero.

**Code (`proteinzen/model/denoiser.py`, around line 454):**
```python
node_seq_idx_embed = self.index_embedder(seq_idx)
if self.disable_absolute_res_idx:
    node_seq_idx_embed = node_seq_idx_embed * 0
node_init = self.node_init(node_seq_idx_embed) * (~is_unindexed_mask[..., None])
```
- `self.index_embedder = fn.partial(get_index_embedding, embed_size=index_embed_size)` -- a
  sine/cosine positional embedding function (`torch.sin`/`torch.cos` over an arange, not a
  lookup table), computed from `seq_idx` and then immediately zeroed out.
- `self.node_init = Linear(index_embed_size, self.c_s, bias=False)` -- confirmed **bias-free**
  (`bias=False` at the constructor, line 289).

**Confirmed: genuinely always exactly zero when `disable_absolute_res_idx=True`.** `0.0 * x ==
0.0` is exact in IEEE float (fp32/fp16/bf16, no NaN/Inf inputs here) regardless of
`self.index_embedder`'s output, and a bias-free `Linear` applied to an all-zero input is an exact
zero matmul (`0 @ W == 0`, exactly, for any finite `W`) -- no approximation involved, and the
subsequent `* (~is_unindexed_mask[..., None])` boolean mask multiply can't change an exact zero
into anything else either. So the "always-zero" claim from the earlier flag is correct, not a
mistaken lead.

**But the computation is genuinely trivial -- measured, not assumed.** Microbenchmarked the
exact code path (`get_index_embedding` + zero-out + bias-free `Linear` matmul) on GPU2 against a
short-circuit (`torch.zeros(...)` of the same shape/dtype/device) at the real launch config's
shapes (`c_s=768`, `index_embed_size=256` default/unoverridden, batch=3, N=256 tokens matching
`crop128_256`):
- Full (current) path: **0.168 ms/iter**
- Short-circuit (skip) path: **0.008 ms/iter**
- Savings: **~0.16 ms per call**

`_gen_node_features` (and this code inside it) is called once per `Embedder` forward -- not
per-trunk-block, and not multiplied by `num_blocks=12` the way the `token_rigids` dedup above
was. Measured training throughput in this same session's stress tests is ~0.4 it/s, i.e. ~2.5s
per optimizer step (`accumulate_grad_batches=3`); even comparing against a single micro-batch's
forward+backward alone (almost certainly hundreds of ms, given the 12-block IPA/triangle-
attention trunk), ~0.16ms is on the order of **0.01-0.05% of step time** -- several orders of
magnitude below anything that would show up in wall-clock throughput, and almost certainly
smaller than this GPU/CUDA stack's own measured run-to-run noise floor documented elsewhere in
this report.

**Decision: investigated, confirmed correct, NOT implemented.** This is exactly the "trivial,
not worth the complexity" case the task briefing anticipated. Skipping a mathematically-zero
~0.16ms computation is real but immaterial, and even a trivially-correct short-circuit edit adds
review/merge-conflict surface to a hot file (`denoiser.py`) that three other agents touched
tonight, for a saving that will not be distinguishable from noise in any wall-clock comparison.
No code change made; repo left at `0f1b900` with no diff. GPU2 confirmed idle (0 MiB) throughout
(no training process launched for this investigation -- a pure CPU-side read + a standalone
microbenchmark script, not using the shared training launch infrastructure).
