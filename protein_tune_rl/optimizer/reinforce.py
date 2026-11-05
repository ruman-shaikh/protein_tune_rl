import torch
from torch.optim import Adam


class Reinforce:
    def __init__(
        self,
        policy: torch.nn.Module,
        learning_rate: float = 1e-3,
        entropy_weight: float = 5e-3,
        constrained_ids=None,
    ):
        self.policy_optimizer = Adam(policy.parameters(), lr=learning_rate)
        self.entropy_weight = entropy_weight
        # REINFORCE optimizes the sampler's log-probs directly, which are
        # already masked to the constrained support, so nothing to apply here.
        # Accepted only to keep the optimizer factory signature uniform.
        self.constrained_ids = constrained_ids

    def _compute_loss(self, reward, baseline, logp, entropy):
        return (
            -((reward - baseline) * logp).mean() - self.entropy_weight * entropy.mean()
        )

    def step(self, reward, baseline, logp, entropy, sequences):
        self.policy_optimizer.zero_grad()
        loss = self._compute_loss(reward, baseline, logp, entropy)
        loss.backward()
        self.policy_optimizer.step()
