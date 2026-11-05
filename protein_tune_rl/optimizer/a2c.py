import copy
from typing import Optional

import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from transformers.modeling_outputs import CausalLMOutput

from protein_tune_rl.util.util import normalize_across_processes


class TokenValue(nn.Module):
    def __init__(self, model_in, name):
        super(TokenValue, self).__init__()
        self.model_in = copy.deepcopy(model_in.module.model)
        self.linear_head = nn.Linear(self.model_in.lm_head.in_features, 1)
        self.name = name

    def forward(
        self, input_ids, token_type_ids=None, labels=None, **kwargs
    ) -> CausalLMOutput:

        model_out = self.model_in(
            input_ids=input_ids,
            labels=labels,
            return_dict=True,
            output_hidden_states=True,
            **kwargs,
        )

        last_hidden_state = model_out["hidden_states"][-1]
        values = self.linear_head(last_hidden_state).squeeze(-1)
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            values = values * attention_mask.to(values.dtype)
        return values


class A2C:
    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 1e-3,
        entropy_weight: float = 5e-3,
        value_loss_weight: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = False,
        max_grad_norm: Optional[float] = None,
        constrained_ids=None,
    ):
        self.entropy_weight = entropy_weight
        self.value_loss_weight = value_loss_weight
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_advantage
        self.max_grad_norm = max_grad_norm

        self.policy = policy
        self.policy_optimizer = Adam(policy.parameters(), lr=learning_rate, eps=1e-5)

        # Follow the supplied policy placement. CUDA availability alone is not
        # sufficient because callers may deliberately construct a CPU policy.
        self.device = next(policy.parameters()).device

        # Token ids masked to -inf during sampling; the on-policy log-probs and
        # entropy must be computed over the same constrained support.
        self.constrained_ids = (
            None
            if constrained_ids is None
            else torch.as_tensor(constrained_ids, dtype=torch.long, device=self.device)
        )

        state_value = TokenValue(policy, "value_function")
        state_value.eval()
        state_value.to(self.device)
        if self.device.type == "cuda":
            self.state_value = DDP(state_value, device_ids=[self.device.index])
        else:
            self.state_value = DDP(state_value)

        self.value_optimizer = Adam(
            self.state_value.parameters(), lr=learning_rate, eps=1e-5
        )

    def _compute_gae(self, rewards, values, mask, last_values=None, truncated=None):
        batch_size, steps = rewards.size()
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(batch_size, device=rewards.device)

        # Bootstrap column appended after the last step. Episodes that were cut
        # off by max_length (truncated) continue past their last token, so we
        # bootstrap with V(s_T); episodes that emitted the stop token are true
        # terminals and get a zero (no bootstrap).
        if truncated is None or last_values is None:
            boot_val = torch.zeros(batch_size, device=values.device)
            boot_mask = torch.zeros(batch_size, device=mask.device)
        else:
            boot_mask = truncated.to(mask.dtype)
            boot_val = last_values.to(values.dtype)

        values_pad = torch.cat([values, boot_val[:, None]], dim=1)
        mask_pad = torch.cat([mask, boot_mask[:, None]], dim=1)

        for t in reversed(range(steps)):
            next_mask = mask_pad[:, t + 1]
            delta = (
                rewards[:, t]
                + self.gamma * values_pad[:, t + 1] * next_mask
                - values_pad[:, t]
            )
            gae = delta + self.gamma * self.gae_lambda * gae * next_mask
            advantages[:, t] = gae * mask[:, t]

        return advantages

    def _build_step_rewards(self, reward, action_mask, values_like):
        rewards = torch.zeros_like(values_like)
        lengths = action_mask.sum(1).long()
        last_idx = lengths - 1
        batch_idx = torch.arange(rewards.size(0), device=rewards.device)
        valid = last_idx >= 0
        if valid.any():
            rewards[batch_idx[valid], last_idx[valid]] = reward[valid]
        return rewards

    def step(self, reward, baseline, logp, entropy, batch):
        init_size = batch["init_size"]
        state = {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "position_ids": batch["position_ids"].to(self.device),
        }
        action_mask = batch.get("action_mask")
        if action_mask is None:
            action_mask = batch["attention_mask"][:, init_size:]

        action_mask = action_mask.to(self.device)

        action = batch["input_ids"][:, init_size:].to(self.device)

        model_out = self.policy(**state)
        logits = model_out.logits[:, init_size - 1 : -1, :]
        if self.constrained_ids is not None:
            logits = logits.index_fill(-1, self.constrained_ids, float("-inf"))
        # Categorical is used (rather than a manual gather) so entropy stays
        # NaN-safe when the constrained logits are -inf.
        policy_dist = Categorical(logits=logits)
        step_logp = policy_dist.log_prob(action)
        step_entropy = policy_dist.entropy()

        action_mask_float = action_mask.to(step_logp.dtype)
        step_logp = step_logp * action_mask_float
        step_entropy = step_entropy * action_mask_float

        full_values = self.state_value(**state)
        # V(s_T): value of the state after the final generated token, used only
        # to bootstrap truncated episodes.
        last_values = full_values[:, -1].detach()
        values = full_values[:, init_size - 1 : -1] * action_mask_float

        truncated = batch.get("truncated")
        if truncated is not None:
            truncated = truncated.to(self.device)

        rewards = self._build_step_rewards(reward, action_mask, values)
        values_detached = values.detach()
        advantages = self._compute_gae(
            rewards,
            values_detached,
            action_mask_float,
            last_values=last_values,
            truncated=truncated,
        )

        if self.normalize_adv:
            valid_adv = advantages[action_mask]
            if valid_adv.numel() > 0:
                normalized = normalize_across_processes(valid_adv)
                advantages = advantages.clone()
                advantages[action_mask] = normalized

        returns = advantages + values_detached

        valid_count = action_mask_float.sum().clamp(min=1.0)
        policy_loss = -(advantages.detach() * step_logp).sum() / valid_count
        entropy_loss = -self.entropy_weight * step_entropy.sum() / valid_count
        value_loss = (
            ((values - returns) ** 2) * action_mask_float
        ).sum() / valid_count

        loss = policy_loss + self.value_loss_weight * value_loss + entropy_loss

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.state_value.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        self.value_optimizer.step()
