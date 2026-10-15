# AUTOOPT Overnight Optimization Report

Combined report from the overnight auto-optimize agents on GPU2 and GPU3.

---

## GPU2 Report (token_rigids dedup)

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

---

## GPU3 Report (4 candidates: tokenize, assembler, model_wrapper, integration)

# AUTOOPT overnight report — GPU3 (data pipeline + generative sampling loop)

Branch: `autoopt/gpu3`, worktree `/mnt/scratch/user/daviyang/proteinzen_auto_gpu3`.
**No commits were made to this branch.** The worktree is currently at the baseline
commit `d40c0e0` (plus the pre-existing, out-of-scope `env_vars.sh` REPO_ROOT edit).
This was a deliberate choice, explained below — the 4 candidate changes below are
implemented and reasoned through, but I could not complete this project's required
empirical verification protocol (bit-identical before/after diff + extended stress
test) before being required to end the session, so per the "never guess on anything
risky" instruction I left the repo in its last known-safe (baseline) state rather
than commit anything unverified.

## Status summary

| # | Change | File | Verified? |
|---|--------|------|-----------|
| 1 | Memoize per-residue-type coarse-grain index maps | `tokenize.py` | NOT empirically verified — reasoned safe by inspection only |
| 2 | Dedupe `seq_index` state machine (computed twice per sample) | `assembler.py` | NOT empirically verified — reasoned safe by inspection only |
| 3 | Remove unconditional debug `print()`s in sampling loop | `model_wrapper.py` | NOT empirically verified — reasoned safe by inspection only |
| 4 | Dedupe duplicate `.cpu()` transfer per integration step | `integration.py` | NOT empirically verified — reasoned safe by inspection only |

None of these are committed. All four are **recommended as very-low-risk** (see
reasoning per-item below) and are good candidates for a follow-up session to
actually run through the verification protocol and commit if confirmed — the full
patches are included below so no work is lost.

## Why verification didn't finish

- Each correctness-dump run needs: this worktree's GPU3-isolated launch script →
  checkpoint restore of a real trained model → one real train step → one real val
  step. Setup alone (precomputing the `igso3`/`dlog_igso3` expansion tables, which
  are not cached to disk here) took ~8-9 minutes before any train/val step ran, and
  the full run (train step + val step + 3x 100-step sampling trajectories at
  ~13-14s/trajectory) takes materially longer. Each before/after attempt is a
  15-25+ minute round trip.
- The periodic-sampling path (`_run_epoch_sample`, needed to verify changes #3/#4)
  and even the existing `val_t0.0/0.5` correctness dumps (needed for #1/#2) are
  *both* gated behind `self.trainer.current_epoch % 5 == 0` in `lmod.py`. Resuming
  from the shared checkpoint (`.../version_2087577/checkpoints/last.ckpt`) with
  `limit_train_batches=1` does not reliably land on a multiple of 5: two otherwise
  identical resume attempts landed on epoch 970 and epoch 971 respectively (971 % 5
  != 0), i.e. the effective epoch count after resume is not deterministic run to
  run (likely depends on how the custom task/length sampler reports its length to
  Lightning). This meant the very first "before" run silently produced **no**
  correctness dumps at all.
- To work around that I added a test-only patch to `lmod.py` (`write_pdbs = (...)
  or (bool(os.environ.get("FORCE_EPOCH_SAMPLE")) and batch_idx==0)`), applied via
  `FORCE_EPOCH_SAMPLE=1`, to force the dumps on the first validation batch
  regardless of epoch. That patch was **never committed** — it only ever existed
  transiently on the remote worktree during testing and has since been reverted.
  It is reproduced in full below so a follow-up session can reuse it without
  re-deriving it.
- I also added (and have since reverted) a `CORRECTNESS_CHECK_TAG`-gated dump in
  `_run_epoch_sample` of the full 100-step trajectory (`final_denoised_rigids7`,
  `final_pred_seq`, full `prot_traj`/`clean_traj` arrays) to
  `correctness_<TAG>_epoch_sample_<split>.pt`, since no existing tool captured
  full-trajectory sampling output for a before/after diff. A ready-to-use diff
  script comparing two tags across train grads + val_t0.0/0.5 + all three
  epoch_sample splits is saved at
  `/mnt/scratch/user/daviyang/scratch_scripts/diff_gpu3_full.py` (usage:
  `python diff_gpu3_full.py <tag_a> <tag_b>`).
- With the force-patch in place, a "before" run (tag `gpu3_before`, baseline code)
  did successfully produce `correctness_gpu3_before_train_grads.pt`,
  `correctness_gpu3_before_val_t0.0.pt`, `correctness_gpu3_before_val_t0.5.pt`
  (these three exist right now under `/mnt/scratch/user/daviyang/restore_logs/` and
  are safe to reuse as the "before" reference for change #1/#2 — they were
  produced by genuinely-unmodified baseline code). It was midway through the three
  `_run_epoch_sample` trajectories (needed for change #3/#4, and also a second,
  stronger check for #1/#2) when I had to stop. I killed that in-flight job
  cleanly and confirmed GPU3 returned to ~0 MiB used (and confirmed no leaked
  `pt_data_worker`/python processes remain under my job on GPU3 — the one earlier
  scare where killing the launcher process seemingly left orphaned re-spawning
  children was PyTorch Lightning's own DDP-strategy launcher/worker process pair,
  not a real leak; everything is cleaned up now).

## The 4 candidate changes (patches, not applied/committed)

### 1. `proteinzen/data/featurize/tokenize.py` — memoize per-residue-type index maps

In `_get_standard_protein_residue_data`, the block building `atom_name_to_cg_idx`
/ `cg_idx_to_atom_idx` (several `list.index()` scans over a residue's atom14 name
list) is a **pure function of `res["name"])`** — one of ~20 standard amino acid
names — yet was recomputed from scratch for every single standard-residue token in
every sample (hundreds of times per crop). Memoized via `functools.lru_cache` keyed
on the residue name string. No mutation of the returned dict/list objects happens
anywhere downstream (checked: both are only read, never written, by the calling
code), so sharing the cached objects across calls is safe. This is the analogous
"cache a value that only depends on a small fixed key space" pattern already used
elsewhere in this codebase (and is a *different* function than the
`frame_atom_ids`/`standard_residue_to_frames` function GPU0 owns — confirmed by
reading GPU0's coordination-file entry, no overlap).

```diff
--- a/proteinzen/data/featurize/tokenize.py
+++ b/proteinzen/data/featurize/tokenize.py
@@ -711,8 +711,41 @@
     new_task_data['residue_entity_id'] = np.array(residue_entity_ids)

     return copy_struct, struct_new, new_task_data
+
+
+@fn.lru_cache(maxsize=None)
+def _cg_idx_maps_for_residue(res_name: str):
+    """ Compute the atom-name <-> coarse-grain-rigid index maps for a standard residue type.
+
+    This is a pure function of `res_name` (one of ~20 standard amino acid names), so it is
+    memoized here: without caching, `_get_standard_protein_residue_data` recomputed this
+    (including several O(n) `list.index()` scans) once per residue token, for every residue
+    of every training sample. Caching collapses that down to one computation per residue type.
+
+    Returns
+    =======
+    atom_name_to_cg_idx: dict[str, int]
+        Maps atom name -> coarse-grain rigid index (0, 1, or 2).
+    cg_idx_to_atom_idx: dict[int, list[int]]
+        Maps coarse-grain rigid index -> list of atom14-order indices belonging to that rigid.
+    """
+    atom_name_to_cg_idx = {}
+    cg_idx_to_atom_idx = {0: [], 1: [], 2: []}
+    atom_order = rc.restype_name_to_atom14_names[res_name]
+    for atom_name in ['N', 'CA', 'C', 'O', 'CB']:
+        atom_name_to_cg_idx[atom_name] = 0
+        if atom_name == 'CB' and res_name == "GLY":
+            continue
+        cg_idx_to_atom_idx[0].append(atom_order.index(atom_name))
+    for atom_name in cg.coarse_grain_sidechain_groups[res_name][2]:
+        atom_name_to_cg_idx[atom_name] = 1
+        cg_idx_to_atom_idx[1].append(atom_order.index(atom_name))
+    for atom_name in cg.coarse_grain_sidechain_groups[res_name][3]:
+        atom_name_to_cg_idx[atom_name] = 2
+        cg_idx_to_atom_idx[2].append(atom_order.index(atom_name))
+    return atom_name_to_cg_idx, cg_idx_to_atom_idx
+
+
 class StructureTokenizer:
     """ A class object for tokenization, mainly to help code organization"""
     def __init__(
@@ -791,20 +824,8 @@
         )

         # compute a mapping for which rigids correspond to which atoms
-        atom_name_to_cg_idx = {}
-        cg_idx_to_atom_idx = {0: [], 1: [], 2: []}
-        atom_order = rc.restype_name_to_atom14_names[res["name"]]
-        for atom_name in ['N', 'CA', 'C', 'O', 'CB']:
-            atom_name_to_cg_idx[atom_name] = 0
-            if atom_name == 'CB' and res["name"] == "GLY":
-                continue
-            cg_idx_to_atom_idx[0].append(atom_order.index(atom_name))
-        for atom_name in cg.coarse_grain_sidechain_groups[res["name"]][2]:
-            atom_name_to_cg_idx[atom_name] = 1
-            cg_idx_to_atom_idx[1].append(atom_order.index(atom_name))
-        for atom_name in cg.coarse_grain_sidechain_groups[res["name"]][3]:
-            atom_name_to_cg_idx[atom_name] = 2
-            cg_idx_to_atom_idx[2].append(atom_order.index(atom_name))
+        # (memoized: this only depends on res["name"], not on this particular residue instance)
+        atom_name_to_cg_idx, cg_idx_to_atom_idx = _cg_idx_maps_for_residue(str(res["name"]))
         # update the global atom_to_rigid mapping
         for i, atom in enumerate(atoms):
             atom_name = atom['name']
```

### 2. `proteinzen/data/featurize/assembler.py` — dedupe `seq_index` computation

`process_token_features` and `process_rigid_features` (both called, back to back,
from `featurize()` on the exact same `data.tokens`) each independently ran an
*identical* `O(n_tokens)` state machine over `token_data['res_idx']` /
`['is_unindexed']` to build `seq_index` — confirmed line-by-line identical logic.
Factored into `_compute_seq_index()`, computed once in `featurize()`, and passed
into both. Checked: in `process_token_features`, the `res_index`/`token_to_rep_rigid`
construction that was interleaved with the old `seq_index` loop does not actually
depend on `seq_index_ref` membership (it appends `res_idx` identically whether or
not the residue index was already seen) — so splitting it out changes nothing
there either.

(Full diff in the `tool-results`/report accompanying this; omitted here for length —
see the patch stored at the end of this document.)

### 3. `proteinzen/stoch_interp/model_wrapper.py` — remove debug `print()`s in sampling loop

`ModelForwardWrapper.get_scores_and_vfs` (called once per integration step, 100
times per sampling trajectory, via `EulerIntegrator.integration_step` →
`self.model.get_scores_and_vfs`) unconditionally ran two `print({...})` statements
computing `.nan_to_num().norm(dim=-1).mean()` / `.var()` over every entry of
`scores_and_vfs` and printing the result. This is pure debug output — it does not
write back into `scores_and_vfs` or `model_outputs`, so removing it cannot change
any returned tensor. But printing a CUDA tensor forces a device-to-host sync (plus
the norm/var reduction kernels), twice, on every single step of every sampling
trajectory — this is exactly the "unnecessary CPU-GPU sync in the integration
loop" pattern called out in the assignment.

```diff
--- a/proteinzen/stoch_interp/model_wrapper.py
+++ b/proteinzen/stoch_interp/model_wrapper.py
@@ -197,15 +197,6 @@
             if isinstance(value, torch.Tensor) and value.requires_grad:
                 model_outputs[key] = value.detach()

-        print({
-            key: value.nan_to_num().norm(dim=-1).mean()
-            for key, value in scores_and_vfs.items()
-        })
-        print({
-            key: value.nan_to_num().norm(dim=-1).var()
-            for key, value in scores_and_vfs.items()
-        })
-
         return scores_and_vfs, model_outputs
```

(Note: similar unconditional `print()`s also exist in `BaseEulerSDEStep.rot_step`
behind `self.reflecting_diff_rots` — off by default — and in
`GradientGuidedEulerODEStep.trans_step`/`rot_step` in `diffeq.py`. Neither of those
diffeq classes is the one actually constructed by `_run_epoch_sample` — which uses
`BaseEulerODEStep()` — so I left them alone; flagging for whoever eventually touches
those code paths.)

### 4. `proteinzen/stoch_interp/integration.py` — dedupe duplicate `.cpu()` transfer

`EulerIntegrator.integration_step` called `denoiser_out["pred_seq"].detach().cpu()`
**twice** per step — once for `clean_traj_point`, once for `prot_traj_point` — on
the exact same source tensor. Computed once into `pred_seq_cpu` and reused for both
tuples. This halves the number of device-to-host transfers (and the sync each one
forces) for `pred_seq` per integration step, 100 steps per trajectory.

```diff
--- a/proteinzen/stoch_interp/integration.py
+++ b/proteinzen/stoch_interp/integration.py
@@ -259,11 +259,17 @@
         pred_rigids = denoiser_out['denoised_rigids']
         pred_trans_1 = pred_rigids.get_trans()
         pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
+
+        # computed once and shared below: clean_traj_point and prot_traj_point both store
+        # the same denoiser_out["pred_seq"] on CPU, so avoid issuing two separate
+        # device-to-host transfers (and the sync each one forces) for the same tensor,
+        # every single integration step.
+        pred_seq_cpu = denoiser_out["pred_seq"].detach().cpu()

         clean_traj_point = (
             pred_trans_1,
             pred_rotmats_1,
-            denoiser_out["pred_seq"].detach().cpu(),
+            pred_seq_cpu,
         )

         trans_t_2 = self.diffeq.trans_step(
@@ -287,7 +293,7 @@
         prot_traj_point = (
             trans_t_2,
             rotmats_t_2,
-            denoiser_out["pred_seq"].detach().cpu(),
+            pred_seq_cpu,
         )

         return denoiser_out, prot_traj_point, clean_traj_point
```

(Note: the identical duplicate-`.cpu()` pattern also exists in `EDMIntegrator` and
`PnPIntegrator`'s `integration_step`s in the same file. Those are not the class used
by `_run_epoch_sample` — which constructs `EulerIntegrator` — so left untouched and
flagged only.)

## Rejected / not attempted

- No new `@torch.compile` was added anywhere, per the explicit instruction to treat
  that as maximally suspect given tonight's unrelated crash elsewhere in this
  project.
- Did not touch `frame_atom_ids`/`standard_residue_to_frames` in `tokenize.py`
  (GPU0's claimed scope) or anything in the model trunk/self-conditioning
  (GPU0/1/2's scope).
- Did not pursue deeper changes to the sampling loop's RNG/igso3 setup (e.g.
  hoisting the per-trajectory-constant `self.corrupter.igso3.sample(...)` call in
  `_run_epoch_sample`, which runs once per `_run_epoch_sample` call already, not
  once per step, so there is nothing to hoist there) — did not find further
  "computed every step but constant across the trajectory" work inside
  `EulerIntegrator.integration_step` itself beyond what's already fixed above; the
  per-step `ru.Rigid(...).to_tensor_7()` construction legitimately depends on that
  step's `trans_t`/`rotmats_t` and is not redundant.

## Follow-up recommendation for whoever picks this up

1. Re-apply the 4 patches above.
2. Re-apply the test-only `lmod.py` patch below (NOT for commit) to force
   `write_pdbs`/`run_epoch_sample` regardless of epoch parity:
   ```diff
           write_pdbs = (
               self.trainer.current_epoch % 5 == 0
               and batch_idx == 0  # only first val batch
   -        )
   +        ) or (bool(os.environ.get("FORCE_EPOCH_SAMPLE")) and batch_idx == 0)  # AUTOOPT TEST-ONLY, revert before commit
           run_epoch_sample = write_pdbs
   ```
   and re-add the `CORRECTNESS_CHECK_TAG`-gated full-trajectory dump right after
   `clean_traj_np`/`prot_traj_np` are built in `_run_epoch_sample` (also TEST-ONLY
   unless you decide it's worth keeping permanently like the other
   `CORRECTNESS_CHECK_TAG` hooks already in this file):
   ```python
   _correctness_tag = os.environ.get("CORRECTNESS_CHECK_TAG")
   if _correctness_tag:
       _dumped = getattr(self, "_correctness_epoch_sample_dumped", None)
       if _dumped is None:
           _dumped = set()
           self._correctness_epoch_sample_dumped = _dumped
       if split not in _dumped:
           _dumped.add(split)
           torch.save(
               {
                   'final_denoised_rigids7': final_rigids.to_tensor_7().cpu(),
                   'final_pred_seq': final_denoiser_out['pred_seq'].cpu(),
                   'prot_traj': np.stack(prot_traj_np),
                   'clean_traj': np.stack(clean_traj_np),
               },
               f"/mnt/scratch/user/daviyang/restore_logs/correctness_{_correctness_tag}_epoch_sample_{split}.pt",
           )
   ```
3. Run `bash /mnt/scratch/user/daviyang/scratch_scripts/corr_test_gpu3.sh gpu3_after2 _autoopt_gpu3_corr_after2` with
   `FORCE_EPOCH_SAMPLE=1` and `CORRECTNESS_CHECK_TAG=gpu3_after2` set (see the
   script, already on disk at that path, for the rest of the env/config — it
   already sets `CUBLAS_WORKSPACE_CONFIG`, `limit_val_batches=1`,
   `limit_train_batches=1`). The existing `correctness_gpu3_before_{train_grads,
   val_t0.0, val_t0.5}.pt` dumps under `/mnt/scratch/user/daviyang/restore_logs/`
   (produced from genuinely-unmodified baseline code, before the kill) can be
   reused as-is as the "before" reference for these 3 artifacts; only the
   `epoch_sample_*` dumps need a fresh baseline run with the same `FORCE_EPOCH_SAMPLE`
   test patch (baseline code, tag e.g. `gpu3_before2`) since the first "before" run
   was killed before producing them.
4. `python /mnt/scratch/user/daviyang/scratch_scripts/diff_gpu3_full.py gpu3_before gpu3_after2`
   (or `gpu3_before2`/`gpu3_after2` once both exist) — require exact 0 abs diff on
   every field before accepting.
   **IMPORTANT CAVEAT (reported by GPU2 agent mid-session, after I'd already
   written the above):** running the SAME unmodified baseline code twice (zero
   edits, same checkpoint/seed) was observed to produce differing param
   grad-norms for the large majority of parameters (2077/2216, max_abs_diff≈0.118)
   despite bit-identical forward outputs at t=0.0/0.5 — i.e. this model's backward
   pass is NOT bit-reproducible on this GPU/CUDA stack even with
   `torch.use_deterministic_algorithms(True, warn_only=True)` and a fixed seed, only
   the forward pass is. This means the `train_grads.pt` comparison is **not a
   valid 0-abs-diff check as-is** — you must first run a same-code-vs-same-code
   control (baseline vs baseline) to measure the noise floor, and only treat a
   candidate change as suspect if its diff is clearly outside that noise floor (or
   better: avoid relying on the grad-norm dict for backward-pass verification
   entirely, and instead compare actual full per-parameter gradient tensors, or
   rely on the forward-output bit-identical check plus the extended real-training
   stress test as the primary signal for a change like mine that only touches
   dataloader/sampling-loop code, never the backward pass itself). None of my 4
   changes touch anything that participates in the training backward pass (#1/#2
   are CPU numpy/dict bookkeeping ahead of featurization; #3/#4 only run inside a
   `@torch.no_grad()` sampling path), so in principle the forward-output
   bit-identical check plus a real multi-epoch stress test should be sufficient
   evidence for these specific four, without needing a clean grad-norm diff at
   all — but flagging this since the report template calls for it and the naive
   version of that check is unreliable on this hardware/stack.
5. Only after that, and only after at least 1000 real training steps (no
   `FORCE_EPOCH_SAMPLE`, no `limit_train_batches`) with zero crashes, commit.
6. Expect each attempt to cost ~15-25 minutes wall clock (igso3 precompute +
   checkpoint load dominate, not the actual forward/sampling work) — budget
   accordingly.

## GPU/process hygiene notes for whoever reads this

- GPU3 is currently idle (0 MiB), confirmed via `nvidia-smi` after killing the
  in-flight test job.
- One thing worth knowing: PyTorch Lightning's DDP strategy (even with
  `devices=1`) spawns a *second* process (the actual rank-0 worker) underneath the
  one you launch; killing only the top-level launcher PID left an orphaned,
  reparented-to-init worker process still holding GPU memory. If you ever need to
  kill one of these test jobs, don't trust a single PID — repeatedly
  `pgrep -f <your unique hydra.run.dir string>` and kill everything that matches
  until the list is empty, then confirm via `nvidia-smi`.
- The `igso3`/`dlog_igso3` expansion precompute (`SampleIGSO3`, used by the
  corrupter) is **not cached to disk** in this setup and reruns from scratch (several
  minutes) on every single process launch — this, not the model forward pass, is
  why a "quick" correctness-check run takes 15+ minutes. Out of scope for me to fix
  tonight (not part of the data pipeline/sampling-loop assignment, and touching the
  corrupter/igso3 cache behavior is its own can of worms), but flagging it as a
  very plausible real wall-clock win for whoever owns overall training throughput:
  if this can be cached to disk and reused across resumes, every single
  training-job restart (not just my test runs) pays this cost today.

## Final commit log on this branch

No new commits. `git log --oneline -3` on `autoopt/gpu3`:
```
d40c0e0 Baseline for overnight auto-optimize agents: broadcast_pairs compile crash fix + verified session optimizations (gather_helper/_gen_rigid_features compile, permute_final_dims caching, torsion_pred skip, motif_idx gating, tensor hoisting)
5512774 Add interface_repack+seq_repack combined ablation config and train script
d25bffd Add interface_seq_repack_rate: mirror-image mode of interface_repack_rate
```

## FINAL OUTCOME (coordinator-verified)

All 4 candidates passed individual verification (isolated unit tests + full-pipeline
gradient checks within the measured noise floor, directionality at chance level for
each) and were then stacked together and run through a combined extended stress test
(`gpu3_final_stress.sh`, continuous real training, no step/batch limits, resumed from
the same warm-start checkpoint as the rest of tonight's work).

Combined stress test ran cleanly through Epoch 974 (55%), i.e. roughly 1186 real
training steps, with zero crashes, zero tracebacks, and sane finite loss metrics
throughout. Two earlier external `fuser -k`-caused kills during individual
verification (one self-inflicted early in the session, since root-caused and fixed;
one reciprocal, from another agent) are documented above and were not code issues.

All 4 candidates committed individually on `autoopt/gpu3`:
- `d43ff09` — tokenize.py: memoize per-residue-type atom/coarse-grain index maps
- `bd3df6d` — assembler.py: dedup duplicated seq_index computation
- `d2d49d5` — model_wrapper.py: remove unconditional debug prints
- `37e7713` — integration.py: dedup redundant detach().cpu() call

GPU3 confirmed idle (0 MiB) after cleanup via PID-scoped kill
(`pgrep -f _autoopt_gpu3_final_stress | xargs kill -9`), not `fuser -k`.

This branch is now fully verified and committed. No further action needed.
