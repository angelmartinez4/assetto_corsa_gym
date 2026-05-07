import os

import numpy as np
import torch
from torch.optim import Adam

from algorithm.discor.discor.algorithm.base import Algorithm
from algorithm.discor.discor.network import HybridTwinnedStateActionFunction, GaussianHybridPolicy
from algorithm.discor.discor.utils import disable_gradients, soft_update, update_params, \
    assert_action

import logging

logger = logging.getLogger(__name__)


class HSAC(Algorithm):

    def __init__(self, state_dim, action_cont_dim, device, action_disc_dims=None, gamma=0.99,
                 nstep=1, policy_lr=0.0003, q_lr=0.0003, entropy_lr=0.0003,
                 policy_hidden_units=[256, 256], q_hidden_units=[256, 256],
                 target_update_coef=0.005, log_interval=10, seed=0, use_heuristic_gear=False,
                 heuristic_gear_steps=0, heuristic_rpm_range=[1000, 5000], heuristic_speed_gear_range=None,
                 load_from_sac=False, load_sac_dir=None, biased_exploration=False, heuristic_discrete_logits=False):
        super().__init__(
            state_dim, action_cont_dim, device, gamma, nstep, log_interval, seed, has_disc_actions=True)
        assert action_disc_dims is not None
        self._action_cont_dim = action_cont_dim
        self._action_disc_dims = action_disc_dims
        self.has_heuristic_gear = use_heuristic_gear
        self.heuristic_gear_steps = heuristic_gear_steps
        self.heuristic_rpm_range = heuristic_rpm_range
        self.heuristic_speed_gear_range = heuristic_speed_gear_range
        self.load_from_sac = load_from_sac
        self.load_sac_dir = load_sac_dir
        self.biased_exploration = biased_exploration
        self.heuristic_discrete_logits = heuristic_discrete_logits

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
        #self._policy_optim = Adam(self._policy_net.parameters(), lr=policy_lr)
        if not self.load_from_sac:
            self._policy_optim = Adam(self._policy_net.parameters(), lr=policy_lr)
        else:
            self._policy_optim = Adam([
                {'params': self._policy_net.net.parameters(), 'lr': policy_lr * 0.00000000000001},  # slow trunk
                {'params': self._policy_net.continuous_head.parameters(), 'lr': policy_lr * 0.00000000000001},  # cont slow
                {'params': self._policy_net.discrete_heads.parameters(), 'lr': policy_lr},  # disc normal
            ])
        self._q_optim = Adam(self._online_q_net.parameters(), lr=q_lr)

        # Target entropy is -|A_c|. (continuous)
        self._target_entropy_cont = -float(self._action_cont_dim)

        # Target discrete entropy. Each head has a different one, initialized based on size
        self._target_entropy_disc = torch.tensor([0.3 * torch.log(torch.tensor(float(k)))
                                                 for k in self._action_disc_dims],
                                                 device=self._device)

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

        if self.load_from_sac:
            self.load_cont_weights_from_sac()

        if self.heuristic_discrete_logits:
            self._policy_net.apply_heuristic_logits()

    def explore(self, state):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            cont_actions, cont_entropies, _, disc_probs_list, _ = self._policy_net(state)

        cont_actions = cont_actions.cpu().numpy()[0]

        # sample action randomly (to explore!) following multinomial distribution
        disc_actions = [
            torch.multinomial(prob, num_samples=1).cpu().numpy()[0]  # binomial multiclass sampling for each action
            for prob in disc_probs_list
        ]

        return np.concatenate([cont_actions, np.concatenate(disc_actions)]), cont_entropies

    def exploit(self, state):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            _, cont_entropies, cont_actions, disc_probs_list, _ = self._policy_net(state)
        cont_actions = cont_actions.cpu().numpy()[0]
        assert_action(cont_actions)

        # choose best action (argmax from probs) for each discrete action
        disc_actions = [
            probs.argmax(dim=-1).cpu().numpy()[0]
            for probs in disc_probs_list
        ]

        return np.concatenate([cont_actions, disc_actions]), cont_entropies

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
        policy_loss, cont_entropies, disc_entropies_list = self.calc_policy_loss(states)
        update_params(self._policy_optim, policy_loss)

        # Update the entropy coefficient.
        entropy_loss_cont = torch.zeros(1, device=self._device)
        entropy_loss_disc = torch.zeros(1, device=self._device)
        if self.update_entropy:
            entropy_loss_cont, entropy_loss_disc = self.calc_entropy_loss(cont_entropies, disc_entropies_list)
            update_params(self._alpha_cont_optim, entropy_loss_cont)
            update_params(self._alpha_disc_optim, entropy_loss_disc)
            entropy_loss_cont = entropy_loss_cont.detach().item()
            entropy_loss_disc = entropy_loss_disc.detach().item()

        self._alpha_cont = self._log_alpha_cont.detach().exp()
        self._alpha_disc = self._log_alpha_disc.detach().exp()
        # calculate mean for logging, since disc_entropies is not a tensor
        mean_disc_entropy = torch.stack(disc_entropies_list).mean().item()

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
                'stats/alpha_cont', self._alpha_cont.item(),
                self._learning_steps)
            writer.add_scalar(
                'stats/alpha_disc', self._alpha_disc.item(),
                self._learning_steps)
            writer.add_scalar(
                'stats/entropy_cont', cont_entropies.detach().mean().item(),
                self._learning_steps)
            writer.add_scalar(
                'stats/entropy_disc', mean_disc_entropy, self._learning_steps)

            return {"policy_loss": policy_loss.detach().item(),
                    "entropy_loss_cont": entropy_loss_cont,
                    "entropy_loss_disc": entropy_loss_disc,
                    "alpha_cont": self._alpha_cont.item(),
                    "alpha_disc": self._alpha_disc.item(),
                    "entropy_cont": cont_entropies.detach().mean().item(),
                    "entropy_disc": mean_disc_entropy}

    def calc_policy_loss(self, states):
        # Resample actions to calculate expectations of Q.
        cont_actions, cont_entropies, _, disc_probs_list, disc_entropies_list = self._policy_net(states)

        # Expectations of Q with clipped double Q technique.
        # each qi_list has a list for all discrete heads
        # each list has a list for all possible values
        # i.e. [[(q1,a0=0), (q1,a0=1)], [(q2, a1=-1), (q2, a1=0), (q2,a1=1)]
        q1_list, q2_list = self._online_q_net(states, cont_actions)

        policy_loss = torch.zeros(1, device=self._device)
        disc_loss = torch.zeros(1, device=self._device)
        # iterate over all discrete actions
        for q1, q2, disc_probs, disc_entropy in zip(q1_list, q2_list, disc_probs_list, disc_entropies_list):
            q_min = torch.min(q1, q2)  # (B, Ki). min(q1_d, q2_d) for each discrete action
            # expected Q under policy + discrete entropy
            qs_expected = (disc_probs * q_min).sum(dim=-1, keepdim=True)  # (B, 1). sum(pi*q)

            # maximize q_min + alpha_d * H_disc + alpha_c * H_cont
            assert qs_expected.shape == disc_entropy.shape == cont_entropies.shape
            disc_loss += torch.mean(
                -qs_expected
                - self._alpha_disc * disc_entropy
            )

        disc_loss /= len(disc_probs_list)
        policy_loss = disc_loss - torch.mean(self._alpha_cont * cont_entropies)

        return policy_loss, cont_entropies.detach(), [e.detach() for e in disc_entropies_list]

    def calc_entropy_loss(self, cont_entropies, disc_entropies_list) -> tuple[torch.Tensor, torch.Tensor]:
        assert not cont_entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss_cont = -torch.mean(
            self._log_alpha_cont * (self._target_entropy_cont - cont_entropies))

        # discrete alphas, for each discrete head (discrete action)
        # same as continuous but each discrete head has his own target entropy, that's why we iterate
        entropy_loss_disc = torch.zeros(1, device=self._device)
        for i, entropy in enumerate(disc_entropies_list): # iterate over different target entropies
            assert not entropy.requires_grad
            entropy_loss_disc += -torch.mean(
                self._log_alpha_disc * (self._target_entropy_disc[i] - entropy))
        entropy_loss_disc /= len(disc_entropies_list)

        return entropy_loss_cont, entropy_loss_disc

    def update_q_functions(self, batch, writer, imp_ws1=None, imp_ws2=None):
        states, actions, rewards, next_states, dones = batch

        # Calculate current and target Q values.
        disc_actions = actions[:, self._action_cont_dim:] # keep batch dimension, select cont actions
        cont_actions = actions[:, :self._action_cont_dim]
        disc_actions_list = [disc_actions[:, i] for i in range(disc_actions.shape[1])]  # list of N tensors (128,)
        curr_qs1_list, curr_qs2_list = self.calc_current_qs(states, cont_actions, disc_actions_list)
        target_qs_list = self.calc_target_qs(rewards, next_states, dones)

        # Update Q functions.
        q_loss, mean_q1, mean_q2 = \
            self.calc_q_loss(curr_qs1_list, curr_qs2_list, target_qs_list, imp_ws1, imp_ws2)
        update_params(self._q_optim, q_loss)

        if self._learning_steps % self._log_interval == 0:
            writer.add_scalar(
                'loss/Q', q_loss.detach().item(), self._learning_steps)
            writer.add_scalar(
                'stats/mean_Q1', mean_q1, self._learning_steps)
            writer.add_scalar(
                'stats/mean_Q2', mean_q2, self._learning_steps)

        # Return there values for DisCor algorithm.
        return curr_qs1_list, curr_qs2_list, target_qs_list

    def calc_current_qs(self, states, actions, disc_actions_list):
        q1_list, q2_list = self._online_q_net(states, actions)

        curr_qs1, curr_qs2 = [], []
        for i, (q1, q2) in enumerate(zip(q1_list, q2_list)):
            idx = disc_actions_list[i].long().unsqueeze(-1)
            curr_qs1.append(q1.gather(1, idx))
            curr_qs2.append(q2.gather(1, idx))

        return curr_qs1, curr_qs2

    def calc_target_qs(self, rewards, next_states, dones):
        with torch.no_grad():
            next_cont_actions, next_cont_entropies, _, next_disc_probs, _ = self._policy_net(next_states)
            next_qs1_list, next_qs2_list = self._target_q_net(next_states, next_cont_actions)

            target_qs_list = []
            q_cont = self._alpha_cont * next_cont_entropies  # From original SAC
            for q1, q2, disc_probs in zip(next_qs1_list, next_qs2_list, next_disc_probs):
                q_min = torch.min(q1, q2)  # [B, K_di]. K_di: number of values of discrete action i
                disc_log_probs = torch.log(disc_probs + 1e-8)

                # expected = Q'(s, ac) + alpha_d * Hd + alpha_c * Hc
                # Q'(s, ac) = sum(probs * q_min). Hd = -sum(alpha_d * probs * log_probs)
                # fuse the sums: sum(probs * (q_min - alpha_d * log_probs))
                q_disc = torch.sum(disc_probs * (q_min - self._alpha_disc * disc_log_probs), dim=1, keepdim=True)

                q_next = q_disc + q_cont
                target_qs = rewards + (1 - dones) * self._discount * q_next
                target_qs_list.append(target_qs)

        return target_qs_list

    def calc_q_loss(self, curr_qs1_list, curr_qs2_list, target_qs_list,
                    imp_ws1=None, imp_ws2=None):
        assert len(curr_qs1_list) == len(curr_qs2_list) > 0

        n = len(curr_qs1_list)
        q_loss = torch.zeros(1, device=self._device)
        mean_q1, mean_q2 = 0.0, 0.0
        for curr_qs1, curr_qs2, target_qs in zip(curr_qs1_list, curr_qs2_list, target_qs_list):
            assert not target_qs.requires_grad
            assert curr_qs1.shape == target_qs.shape == curr_qs2.shape
            assert imp_ws1 is None or imp_ws1.shape == curr_qs1.shape
            assert imp_ws2 is None or imp_ws2.shape == curr_qs2.shape

            # Q loss is mean squared TD errors with importance weights.
            if imp_ws1 is None:
                q1_loss = torch.mean((curr_qs1 - target_qs).pow(2))
                q2_loss = torch.mean((curr_qs2 - target_qs).pow(2))
            else:
                q1_loss = torch.sum((curr_qs1 - target_qs).pow(2) * imp_ws1)
                q2_loss = torch.sum((curr_qs2 - target_qs).pow(2) * imp_ws2)

            q_loss += q1_loss + q2_loss
            # Mean Q values for logging.
            mean_q1 += curr_qs1.detach().mean().item()
            mean_q2 += curr_qs2.detach().mean().item()

        return q_loss, mean_q1 / n, mean_q2 / n

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self._policy_net.save(os.path.join(save_dir, 'policy_net.pth'))
        self._online_q_net.save(os.path.join(save_dir, 'online_q_net.pth'))
        self._target_q_net.save(os.path.join(save_dir, 'target_q_net.pth'))

    def load_models(self, load_dir):
        self._policy_net.load(os.path.join(load_dir, 'policy_net.pth'))
        self._online_q_net.load(os.path.join(load_dir, 'online_q_net.pth'))
        self._target_q_net.load(os.path.join(load_dir, 'target_q_net.pth'))

    def load_cont_weights_from_sac(self):
        sac_policy_sd = torch.load(os.path.join(self.load_sac_dir, 'policy_net.pth'), map_location=self._device)
        sac_q_sd = torch.load(os.path.join(self.load_sac_dir, 'online_q_net.pth'), map_location=self._device)
        sac_policy_keys = list(sac_policy_sd.keys())

        # Policy
        hsac_policy_sd = self._policy_net.state_dict()
        sac_last_keys = set(sac_policy_keys[-2:])  # w&b last head is out of trunk

        for sac_key, sac_val in sac_policy_sd.items():
            if sac_key in hsac_policy_sd and hsac_policy_sd[sac_key].shape == sac_val.shape:
                hsac_policy_sd[sac_key] = sac_val.clone()
                print(f"[policy] {sac_key} → {sac_key}")
            elif sac_key in sac_last_keys:
                target = 'continuous_head.weight' if sac_key.endswith('.weight') else 'continuous_head.bias'
                if hsac_policy_sd[target].shape == sac_val.shape:
                    hsac_policy_sd[target] = sac_val.clone()
                    print(f"[policy] {sac_key} → {target}")

        self._policy_net.load_state_dict(hsac_policy_sd)

        # Q nets
        for net_attr in ['_online_q_net', '_target_q_net']:
            hsac_q_sd = getattr(self, net_attr).state_dict() # self._online_q_net or self._target_q_net
            for sac_key, sac_val in sac_q_sd.items():
                if sac_key in hsac_q_sd and hsac_q_sd[sac_key].shape == sac_val.shape:
                    hsac_q_sd[sac_key] = sac_val.clone()
                    print(f"[{net_attr}] {sac_key} → {sac_key}")
            getattr(self, net_attr).load_state_dict(hsac_q_sd)
