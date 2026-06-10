"""Endpoint-steered ProteinZen integrator."""

from __future__ import annotations

import inspect
from typing import Any

import torch
import tqdm

from proteinzen.openfold.utils import rigid_utils as ru
from proteinzen.stoch_interp.diffeq import DifferentialEquation
from proteinzen.stoch_interp.integration import EulerIntegrator, Integrator
from proteinzen.stoch_interp.model_wrapper import ModelForwardWrapper
from proteinzen.stoch_interp.steering.reward_scoring import EndpointRewardScorer, RewardFn
from proteinzen.stoch_interp.steering.utils import (
    clone_tree,
    initialize_beam,
    get_batch_size,
    index_tree,
    replace_mapping_inplace,
    select_batch,
    select_traj_point,
    tree_batch_size,
)


class SteeredIntegrator(Integrator):
    """Integrator that steers sampling by endpoint rewards.

    At steering steps, each active sample is branched ``n_branch`` times, each
    branch is advanced to the next timestep, then every branch is rolled out to
    the end of the full trajectory and scored at that endpoint. Beam mode keeps
    the top ``beam_width`` branches per original sample. FK mode samples
    ``beam_width`` branches from a reward-softmax distribution.
    """

    def __init__(
        self,
        *,
        wrapped_model: ModelForwardWrapper,
        diffeq: DifferentialEquation,
        reward_model: Any | None = None,
        reward_fn: RewardFn | None = None,
        reward_scorer: EndpointRewardScorer | None = None,
        reward_kwargs: dict[str, Any] | None = None,
        mode: str = "beam",
        beam_width: int = 4,
        n_branch: int = 4,
        fk_temperature: float = 1.0,
        fk_replacement: bool = True,
        steer_every: int = 1,
        steer_start_step: int | None = None,
        steer_end_step: int | None = None,
        max_batch_size: int | None = None,
        step_integrator: Integrator | None = None,
        step_integrator_cls: type[Integrator] = EulerIntegrator,
        step_integrator_kwargs: dict[str, Any] | None = None,
        score_dir: str | None = None,
        keep_endpoint_pdbs: bool = False,
        endpoint_rollout_speedup: int = 1,
        pass_sequence: bool | None = None,
        pass_structure: bool | None = None,
        drop_copy_inputs: bool = True,
        disable_tqdm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(wrapped_model=wrapped_model, diffeq=diffeq, **kwargs)
        mode = mode.lower()
        if mode not in {"beam", "fk"}:
            raise ValueError(f"mode must be 'beam' or 'fk', got {mode!r}")
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")
        if n_branch < 1:
            raise ValueError(f"n_branch must be >= 1, got {n_branch}")
        if fk_temperature <= 0:
            raise ValueError(f"fk_temperature must be > 0, got {fk_temperature}")
        if steer_every < 1:
            raise ValueError(f"steer_every must be >= 1, got {steer_every}")
        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")

        self.mode = mode
        self.beam_width = beam_width
        self.n_branch = n_branch
        self.fk_temperature = fk_temperature
        self.fk_replacement = fk_replacement
        self.steer_every = steer_every
        if steer_start_step:
            self.steer_start_step = steer_start_step
        else:
            self.steer_start_step = steer_every - 1
        self.steer_end_step = steer_end_step
        self.max_batch_size = max_batch_size
        self.endpoint_rollout_speedup = endpoint_rollout_speedup
        self.disable_tqdm = disable_tqdm

        if step_integrator is None:
            step_integrator_kwargs = dict(step_integrator_kwargs or {})
            step_integrator = step_integrator_cls(
                wrapped_model=wrapped_model,
                diffeq=diffeq,
                **step_integrator_kwargs,
            )
        self.step_integrator = step_integrator

        self.reward_scorer = reward_scorer or EndpointRewardScorer(
            reward_model=reward_model,
            reward_fn=reward_fn,
            reward_kwargs=reward_kwargs,
            work_dir=score_dir,
            keep_pdbs=keep_endpoint_pdbs,
            pass_sequence=pass_sequence,
            pass_structure=pass_structure,
            drop_copy_inputs=drop_copy_inputs,
        )
        self.last_reward_history: list[dict[str, torch.Tensor | int]] = []

    def integration_step(
        self,
        batch,
        trans_t,
        rotmats_t,
        t_1,
        t_2,
        aux_inputs=None,
        self_conditioning=None,
        **kwargs,
    ):
        """Delegate single-step dynamics to the configured base integrator."""

        signature = inspect.signature(self.step_integrator.integration_step)
        if "aux_inputs" in signature.parameters:
            return self.step_integrator.integration_step(
                batch,
                trans_t,
                rotmats_t,
                t_1,
                t_2,
                aux_inputs=aux_inputs,
                self_conditioning=self_conditioning,
                **kwargs,
            )
        else:
            assert aux_inputs is None, (
                f"SteeredIntegrator was given `aux_inputs` but the base Integrator {self.step_integrator} "
                "does not accept this keyword input!"
            )
            return self.step_integrator.integration_step(
                batch,
                trans_t,
                rotmats_t,
                t_1,
                t_2,
                self_conditioning=self_conditioning,
                **kwargs,
            )

    def sample(self, batch, ts):
        """Sample with endpoint steering and return the standard integrator tuple."""
        assert (ts[0] == 0).all(), "SteeredIntegrator is not currently not configured for partial diffusion"
        original_batch = batch
        num_orig = get_batch_size(batch)
        # initialize beams by repeat_interleave and setting random noise for all noised rigids
        work_batch = initialize_beam(batch, self.beam_width)

        # we set all the rigids_x fields to rigids_1 which holds the initial datapoint
        rigids_data = work_batch["rigids"]
        rigids_data["rigids_t"] = rigids_data["rigids_1"]
        rigids_0 = ru.Rigid.from_tensor_7(rigids_data["rigids_t"])
        trans_t = rigids_0.get_trans()
        rotmats_t = rigids_0.get_rots().get_rot_mats()

        self_conditioning = None
        prot_traj = []
        clean_traj = []
        self.last_reward_history = []

        for step_idx, t_2 in enumerate(tqdm.tqdm(ts[1:], disable=self.disable_tqdm)):
            t_1 = ts[step_idx]
            if self._should_steer(step_idx):
                (
                    work_batch,
                    trans_t,
                    rotmats_t,
                    self_conditioning,
                    step_clean_point,
                    step_prot_point,
                    parent_indices,
                ) = self._steered_step(
                    work_batch,
                    trans_t,
                    rotmats_t,
                    t_1,
                    t_2,
                    step_idx,
                    ts,
                    self_conditioning,
                    num_orig,
                )
                if prot_traj:
                    prot_traj = [select_traj_point(point, parent_indices) for point in prot_traj]
                    clean_traj = [select_traj_point(point, parent_indices) for point in clean_traj]
                prot_traj.append(step_prot_point)
                clean_traj.append(step_clean_point)
            else:
                self_conditioning, step_prot_point, step_clean_point = self.integration_step(
                    work_batch,
                    trans_t,
                    rotmats_t,
                    t_1,
                    t_2,
                    self_conditioning=self_conditioning,
                )
                trans_t, rotmats_t, _ = step_prot_point
                prot_traj.append(step_prot_point)
                clean_traj.append(step_clean_point)

        final_denoiser_out, _, _ = self.integration_step(
            work_batch,
            trans_t,
            rotmats_t,
            ts[-1],
            ts[-1],
            self_conditioning=self_conditioning,
        )
        replace_mapping_inplace(original_batch, work_batch)
        return clean_traj, prot_traj, final_denoiser_out

    def _should_steer(self, step_idx: int):
        if step_idx < self.steer_start_step:
            return False
        if self.steer_end_step is not None and step_idx >= self.steer_end_step:
            return False
        return (step_idx - self.steer_start_step) % self.steer_every == 0

    def _steered_step(
        self,
        batch,
        trans_t,
        rotmats_t,
        t_1,
        t_2,
        step_idx: int,
        ts,
        self_conditioning,
        num_orig: int,
    ):
        current_size = trans_t.shape[0]
        device = trans_t.device
        branch_from = torch.arange(current_size, device=device).repeat_interleave(self.n_branch)
        branch_batch = select_batch(batch, branch_from)
        branch_trans = trans_t.index_select(0, branch_from)
        branch_rotmats = rotmats_t.index_select(0, branch_from)
        branch_self_conditioning = self._index_self_conditioning(
            self_conditioning,
            branch_from,
            current_size,
        )

        candidate_sc, candidate_prot_point, candidate_clean_point = self.integration_step(
            branch_batch,
            branch_trans,
            branch_rotmats,
            t_1,
            t_2,
            self_conditioning=branch_self_conditioning,
        )
        candidate_trans, candidate_rotmats, _ = candidate_prot_point

        rewards = self._score_candidates(
            branch_batch,
            candidate_trans,
            candidate_rotmats,
            candidate_sc,
            step_idx + 1,
            ts,
        ).to(device=device)
        select_indices = self._select_candidates(rewards, num_orig, step_idx, device)
        parent_indices = branch_from.index_select(0, select_indices)

        selected_batch = select_batch(branch_batch, select_indices)
        selected_sc = self._index_self_conditioning(
            candidate_sc,
            select_indices,
            candidate_trans.shape[0],
        )
        selected_trans = candidate_trans.index_select(0, select_indices)
        selected_rotmats = candidate_rotmats.index_select(0, select_indices)
        selected_prot_point = select_traj_point(candidate_prot_point, select_indices)
        selected_clean_point = select_traj_point(candidate_clean_point, select_indices)

        return (
            selected_batch,
            selected_trans,
            selected_rotmats,
            selected_sc,
            selected_clean_point,
            selected_prot_point,
            parent_indices,
        )

    def _score_candidates(
        self,
        batch,
        trans_t,
        rotmats_t,
        self_conditioning,
        start_index: int,
        ts,
    ) -> torch.Tensor:
        total = trans_t.shape[0]
        if self.max_batch_size is None or total <= self.max_batch_size:
            return self._score_candidate_chunk(batch, trans_t, rotmats_t, self_conditioning, start_index, ts)

        rewards = []
        for start in range(0, total, self.max_batch_size):
            end = min(start + self.max_batch_size, total)
            indices = torch.arange(start, end, device=trans_t.device)
            chunk_batch = select_batch(batch, indices)
            chunk_sc = self._index_self_conditioning(self_conditioning, indices, total)
            rewards.append(
                self._score_candidate_chunk(
                    chunk_batch,
                    trans_t.index_select(0, indices),
                    rotmats_t.index_select(0, indices),
                    chunk_sc,
                    start_index,
                    ts,
                    tag_offset=start,
                )
            )
        return torch.cat(rewards, dim=0)

    def _score_candidate_chunk(
        self,
        batch,
        trans_t,
        rotmats_t,
        self_conditioning,
        start_index: int,
        ts,
        tag_offset: int = 0,
    ) -> torch.Tensor:
        rollout_batch, final_denoiser_out = self._rollout_to_end(
            batch,
            trans_t,
            rotmats_t,
            start_index,
            ts,
            self_conditioning,
        )
        tags = [f"step{start_index}_cand{tag_offset + i}" for i in range(trans_t.shape[0])]
        rewards = self.reward_scorer.score_batch(rollout_batch, final_denoiser_out, tags=tags)
        return rewards.detach().float()

    def _rollout_to_end(
        self,
        batch,
        trans_t,
        rotmats_t,
        start_index: int,
        ts,
        self_conditioning,
    ):
        rollout_batch = clone_tree(batch)
        rollout_sc = clone_tree(self_conditioning)
        rollout_trans = trans_t.clone()
        rollout_rotmats = rotmats_t.clone()

        for next_index in tqdm.tqdm(range(start_index + 1, len(ts), self.endpoint_rollout_speedup), desc="Rolling out to end"):
            rollout_sc, prot_point, _ = self.integration_step(
                rollout_batch,
                rollout_trans,
                rollout_rotmats,
                ts[next_index - 1],
                ts[next_index],
                self_conditioning=rollout_sc,
            )
            rollout_trans, rollout_rotmats, _ = prot_point

        final_denoiser_out, _, _ = self.integration_step(
            rollout_batch,
            rollout_trans,
            rollout_rotmats,
            ts[-1],
            ts[-1],
            self_conditioning=rollout_sc,
        )
        return rollout_batch, final_denoiser_out

    def _select_candidates(
        self,
        rewards: torch.Tensor,
        num_orig: int,
        step_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        candidates_per_orig = self.beam_width * self.n_branch
        expected = num_orig * candidates_per_orig
        if rewards.numel() != expected:
            raise ValueError(
                f"Expected {expected} candidate rewards, got {rewards.numel()} "
                f"({num_orig=}, {self.beam_width=}, {self.n_branch=})"
            )

        rewards_for_selection = rewards.reshape(num_orig, candidates_per_orig)
        if self.mode == "beam":
            local_indices = torch.topk(rewards_for_selection, k=self.beam_width, dim=1).indices
        else:
            logits = rewards_for_selection / self.fk_temperature
            probs = torch.softmax(logits, dim=1)
            bad_rows = torch.isnan(probs).any(dim=1) | (probs.sum(dim=1) <= 1e-8)
            if bad_rows.any():
                probs[bad_rows] = torch.ones_like(probs[bad_rows]) / probs.shape[1]
            local_indices = torch.multinomial(
                probs,
                num_samples=self.beam_width,
                replacement=self.fk_replacement,
            )

        offsets = torch.arange(num_orig, device=device).unsqueeze(1) * candidates_per_orig
        select_indices = (local_indices.to(device) + offsets).reshape(-1)
        selected_rewards = rewards.index_select(0, select_indices.to(rewards.device))
        self.last_reward_history.append(
            {
                "step_idx": step_idx,
                "candidate_rewards": rewards.detach().cpu(),
                "selected_rewards": selected_rewards.detach().cpu(),
            }
        )
        return select_indices

    @staticmethod
    def _index_self_conditioning(self_conditioning, indices: torch.Tensor, batch_size: int):
        if self_conditioning is None:
            return None
        tree_size = tree_batch_size(self_conditioning)
        return index_tree(self_conditioning, indices, tree_size or batch_size)
