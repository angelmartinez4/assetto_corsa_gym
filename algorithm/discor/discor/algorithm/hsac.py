import os

import numpy as np
import torch
from torch.optim import Adam

from .base import Algorithm
from ..network import HybridTwinnedStateActionFunction, GaussianHybridPolicy
from ..utils import disable_gradients, soft_update, update_params, \
    assert_action


import logging
logger = logging.getLogger(__name__)


class HSAC(Algorithm):

    def __init__(self, state_dim, action_cont_dim, action_disc_dims, device, gamma=0.99,
                 nstep=1, policy_lr=0.0003, q_lr=0.0003, entropy_lr=0.0003,
                 policy_hidden_units=[256, 256], q_hidden_units=[256, 256],
                 target_update_coef=0.005, log_interval=10, seed=0):
        super().__init__(
            state_dim, action_cont_dim, device, gamma, nstep, log_interval, seed)
        self._action_cont_dim = action_cont_dim
        self._action_disc_dims = action_disc_dims

        # Build networks.
        self._policy_net = GaussianHybridPolicy(
            state_dim=self._state_dim,
            action_cont_dim=self._action_cont_dim,
            action_disc_dims=self._action_disc_dims,
            hidden_units=policy_hidden_units
            ).to(self._device)
        self._online_q_net = HybridTwinnedStateActionFunction(
            state_dim=self._state_dim,
            action_cont_dim=self._action_cont_dim,
            action_disc_dims=self._action_disc_dims,
            hidden_units=q_hidden_units
            ).to(self._device)
        self._target_q_net = HybridTwinnedStateActionFunction(
            state_dim=self._state_dim,
            action_cont_dim=self._action_cont_dim,
            action_disc_dims=self._action_disc_dims,
            hidden_units=q_hidden_units
            ).to(self._device).eval()

        # Copy parameters of the learning network to the target network.
        self._target_q_net.load_state_dict(self._online_q_net.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self._target_q_net)

        # Optimizers.
        self._policy_optim = Adam(self._policy_net.parameters(), lr=policy_lr)
        self._q_optim = Adam(self._online_q_net.parameters(), lr=q_lr)

        # Target entropy is -|A_c|. (continuous)
        self._target_entropy_cont = -float(self._action_cont_dim)

        # Target discrete entropy
        self._target_entropy_disc = [0.5 * torch.log(torch.tensor(float(k)))
                                     for k in self._action_disc_dims]

        # We optimize log(alpha), instead of alpha.
        self._log_alpha_cont = torch.zeros(
            1, device=self._device, requires_grad=True)
        self._log_alpha_disc = torch.zeros(1, device=self._device, requires_grad=True)
        self._alpha_cont = self._log_alpha_cont.detach().exp()
        self._alpha_disc = self._log_alpha_disc.detach().exp()

        self._alpha_cont_optim = Adam([self._log_alpha_cont], lr=entropy_lr)
        self._alpha_disc_optim = Adam([self._log_alpha_disc], lr=entropy_lr)

        self._target_update_coef = target_update_coef
        self.update_entropy = True

    def explore(self, state):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            cont_actions, cont_entropies, _, disc_probs_list = self._policy_net(state)

        cont_actions = cont_actions.cpu().numpy()[0]

        # choose best action (argmax from probs) for each discrete action
        disc_actions = [
            probs.argmax(dim=-1).cpu().numpy()[0]
            for probs in disc_probs_list
        ]

        return cont_actions, disc_actions

    def exploit(self, state):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            cont_actions, _, _, disc_probs_list = self._policy_net(state)
        cont_actions = cont_actions.cpu().numpy()[0]
        assert_action(cont_actions)

        # choose best action (argmax from probs) for each discrete action
        disc_actions = [
            probs.argmax(dim=-1).cpu().numpy()[0]
            for probs in disc_probs_list
        ]

        return cont_actions, disc_actions

    def update_target_networks(self):
        soft_update(
            self._target_q_net, self._online_q_net, self._target_update_coef)

    def update_online_networks(self, batch, writer):
        self._learning_steps += 1
        stats = self.update_policy_and_entropy(batch, writer)
        self.update_q_functions(batch, writer)
        return stats

    def update_policy_and_entropy(self, batch, writer):
        states, actions, rewards, next_states, dones = batch

        # Update policy.
        policy_loss, cont_entropies, disc_probs_list = self.calc_policy_loss(states)
        update_params(self._policy_optim, policy_loss)

        # Update the entropy coefficient.
        entropy_loss_cont = torch.zeros(1, device=self._device)
        entropy_loss_disc = torch.zeros(1, device=self._device)
        if self.update_entropy:
            entropy_loss_cont, entropy_loss_disc = self.calc_entropy_loss(cont_entropies, disc_probs_list)
            update_params(self._alpha_cont_optim, entropy_loss_cont)
            update_params(self._alpha_disc_optim, entropy_loss_disc)
            entropy_loss_cont = entropy_loss_cont.detach().item()
            entropy_loss_disc = entropy_loss_disc.detach().item()
        self._alpha_cont = self._log_alpha_cont.detach().exp()

        if self._learning_steps % self._log_interval == 0:
            writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self._learning_steps)
            writer.add_scalar(
                'loss/entropy_cont', entropy_loss_cont,
                self._learning_steps)
            writer.add_scalar(
                'loss/entropy_disc', entropy_loss_disc,
                self._learning_steps)
            writer.add_scalar(
                'stats/alpha', self._alpha_cont.item(),
                self._learning_steps)
            writer.add_scalar(
                'stats/entropy', cont_entropies.detach().mean().item(),
                self._learning_steps)

            return {"policy_loss": policy_loss.detach().item(),
                    "entropy_loss": entropy_loss_cont,
                    "alpha": self._alpha_cont.item(), "entropy": cont_entropies.detach().mean().item()}

    def calc_policy_loss(self, states) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        # Resample actions to calculate expectations of Q.
        cont_actions, cont_entropies, _, disc_probs_list = self._policy_net(states)

        # Expectations of Q with clipped double Q technique.
        # each qi_list has a list for all discrete heads
        # each list has a list for all possible values
        # i.e. [[(q1,a0=0), (q1,a0=1)], [(q2, a1=-1), (q2, a1=0), (q2,a1=1)]
        q1_list, q2_list = self._online_q_net(states, cont_actions)

        #policy_loss = 0.0
        policy_loss = torch.zeros(1, device=self._device)

        for q1, q2, disc_probs in zip(q1_list, q2_list, disc_probs_list):
            qs = torch.min(q1, q2)  # (B, Ki)
            qs_expected = (disc_probs * qs).sum(dim=-1, keepdim=True)  # (B, 1)

            disc_log_probs = torch.log(disc_probs + 1e-8)
            disc_entropies = -(disc_probs * disc_log_probs).sum(dim=-1, keepdim=True)  # (B, 1)

            # Maximizar Q_esperado + αd * H_discreta + αc * H_continua
            assert qs_expected.shape == disc_entropies.shape == cont_entropies.shape
            policy_loss += torch.mean(
                -qs_expected
                - self._alpha_disc * disc_entropies
                - self._alpha_cont * cont_entropies
            )

        return policy_loss, cont_entropies.detach_(), [p.detach() for p in disc_probs_list]

    def calc_entropy_loss(self, cont_entropies, disc_probs_list) -> tuple[torch.Tensor, torch.Tensor]:
        assert not cont_entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss_cont = -torch.mean(
            self._log_alpha_cont * (self._target_entropy_cont - cont_entropies))

        # discrete alphas, for each discrete head (action)
        #entropy_loss_disc = 0.0
        entropy_loss_disc = torch.zeros(1, device=self._device)
        for i, disc_probs in enumerate(disc_probs_list):
            disc_log_probs = torch.log(disc_probs)
            disc_entropies = -(disc_probs * disc_log_probs).sum(dim=-1, keepdim=True)
            assert not disc_entropies.requires_grad

            entropy_loss_disc += -torch.mean(
                self._log_alpha_disc * (self._target_entropy_disc[i] - disc_entropies))

        return entropy_loss_cont, entropy_loss_disc

    def update_q_functions(self, batch, writer, imp_ws1=None, imp_ws2=None):
        states, actions, rewards, next_states, dones = batch

        # Calculate current and target Q values.
        curr_qs1, curr_qs2 = self.calc_current_qs(states, actions)
        target_qs = self.calc_target_qs(rewards, next_states, dones)

        # Update Q functions.
        q_loss, mean_q1, mean_q2 = \
            self.calc_q_loss(curr_qs1, curr_qs2, target_qs, imp_ws1, imp_ws2)
        update_params(self._q_optim, q_loss)

        if self._learning_steps % self._log_interval == 0:
            writer.add_scalar(
                'loss/Q', q_loss.detach().item(),
                self._learning_steps)
            writer.add_scalar(
                'stats/mean_Q1', mean_q1, self._learning_steps)
            writer.add_scalar(
                'stats/mean_Q2', mean_q2, self._learning_steps)

        # Return there values for DisCor algorithm.
        return curr_qs1.detach(), curr_qs2.detach(), target_qs

    def calc_current_qs(self, states, actions):
        curr_qs1, curr_qs2 = self._online_q_net(states, actions)
        return curr_qs1, curr_qs2

    def calc_target_qs(self, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self._policy_net(next_states)
            next_qs1, next_qs2 = self._target_q_net(next_states, next_actions)
            next_qs = \
                torch.min(next_qs1, next_qs2) + self._alpha_cont * next_entropies

        assert rewards.shape == next_qs.shape
        target_qs = rewards + (1.0 - dones) * self._discount * next_qs

        return target_qs

    def calc_q_loss(self, curr_qs1, curr_qs2, target_qs, imp_ws1=None,
                    imp_ws2=None):
        assert imp_ws1 is None or imp_ws1.shape == curr_qs1.shape
        assert imp_ws2 is None or imp_ws2.shape == curr_qs2.shape
        assert not target_qs.requires_grad
        assert curr_qs1.shape == target_qs.shape

        # Q loss is mean squared TD errors with importance weights.
        if imp_ws1 is None:
            q1_loss = torch.mean((curr_qs1 - target_qs).pow(2))
            q2_loss = torch.mean((curr_qs2 - target_qs).pow(2))

        else:
            q1_loss = torch.sum((curr_qs1 - target_qs).pow(2) * imp_ws1)
            q2_loss = torch.sum((curr_qs2 - target_qs).pow(2) * imp_ws2)

        # Mean Q values for logging.
        mean_q1 = curr_qs1.detach().mean().item()
        mean_q2 = curr_qs2.detach().mean().item()

        return q1_loss + q2_loss, mean_q1, mean_q2

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self._policy_net.save(os.path.join(save_dir, 'policy_net.pth'))
        self._online_q_net.save(os.path.join(save_dir, 'online_q_net.pth'))
        self._target_q_net.save(os.path.join(save_dir, 'target_q_net.pth'))

    def load_models(self, load_dir):
        self._policy_net.load(os.path.join(load_dir, 'policy_net.pth'))
        self._online_q_net.load(os.path.join(load_dir, 'online_q_net.pth'))
        self._target_q_net.load(os.path.join(load_dir, 'target_q_net.pth'))