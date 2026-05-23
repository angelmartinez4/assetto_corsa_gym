import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal


def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def create_linear_network(input_dim, output_dim, hidden_units=[],
                          hidden_activation=nn.ReLU(), output_activation=None,
                          initializer=initialize_weights_xavier):
    assert isinstance(input_dim, int) and isinstance(output_dim, int)
    assert isinstance(hidden_units, list) or isinstance(hidden_units, list)

    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units

    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers).apply(initialize_weights_xavier)


class BaseNetwork(nn.Module):

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class StateActionFunction(BaseNetwork):

    def __init__(self, state_dim, action_dim, hidden_units=[256, 256]):
        super().__init__()

        self.net = create_linear_network(
            input_dim=state_dim+action_dim,
            output_dim=1,
            hidden_units=hidden_units)

    def forward(self, x):
        return self.net(x)


class TwinnedStateActionFunction(BaseNetwork):

    def __init__(self, state_dim, action_dim, hidden_units=[256, 256]):
        super().__init__()

        self.net1 = StateActionFunction(state_dim, action_dim, hidden_units)
        self.net2 = StateActionFunction(state_dim, action_dim, hidden_units)

    def forward(self, states, actions):
        assert states.dim() == 2 and actions.dim() == 2

        x = torch.cat([states, actions], dim=1)
        value1 = self.net1(x)
        value2 = self.net2(x)
        return value1, value2


class GaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, state_dim, action_dim, hidden_units=[256, 256]):
        super().__init__()

        self.net = create_linear_network(
            input_dim=state_dim,
            output_dim=2*action_dim,
            hidden_units=hidden_units)

    def forward(self, states):
        assert states.dim() == 2

        # Calculate means and stds of actions.
        means, log_stds = torch.chunk(self.net(states), 2, dim=-1)
        log_stds = torch.clamp(
            log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        stds = log_stds.exp_()

        # Gaussian distributions.
        normals = Normal(means, stds)

        # Sample actions.
        xs = normals.rsample() # equivalente a xs = mean + ruido_normal*std
        actions = torch.tanh(xs)

        # Calculate entropies.
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + 1e-6)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)


class GaussianHybridPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, state_dim, action_cont_dim, action_disc_dims=[3], hidden_units=[256, 256]):
        super().__init__()

        self.net = create_linear_network(
            input_dim=state_dim,
            output_dim = hidden_units[-1],
            hidden_units=hidden_units[:-1],
            output_activation=nn.ReLU())

        self.continuous_head = nn.Linear(hidden_units[-1], 2 * action_cont_dim) # * 2, mean/std for each cont feat
        self.discrete_heads = nn.ModuleList([
            nn.Linear(hidden_units[-1], dim) for dim in action_disc_dims
        ])

        self.apply(initialize_weights_xavier)

    def apply_heuristic_logits(self):
        for head in self.discrete_heads:
            nn.init.constant_(head.bias, 0.0)
            head.bias.data[0] = 2.0  # keep gear highest logit

    def forward(self, states):
        assert states.dim() == 2
        features_hidden = self.net(states) # feature output of hidden layer

        # --- Continuous head ---
        # Calculate means and stds of actions.
        means, log_stds = torch.chunk(self.continuous_head(features_hidden), 2, dim=-1)
        log_stds = torch.clamp(
            log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        stds = log_stds.exp_()

        # Gaussian distributions.
        normals = Normal(means, stds)

        # Sample actions.
        xs = normals.rsample()  # equivalent to xs = mean + Normal(0,1).sample()*std
        cont_actions = torch.tanh(xs)

        # Calculate entropies.
        cont_log_probs = normals.log_prob(xs) - torch.log(1 - cont_actions.pow(2) + 1e-6)
        cont_entropies = -cont_log_probs.sum(dim=1, keepdim=True)

        # --- Discrete head ---
        disc_probs = []
        disc_entropies = []
        for head in self.discrete_heads: # have to iterate since heads can have different size
            logits = head(features_hidden)
            probs = F.softmax(logits, dim=-1)
            disc_probs.append(probs)

            # Discrete entropies: -sum(probs * log(probs))
            disc_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)
            disc_entropies.append(disc_entropy)

        return cont_actions, cont_entropies, torch.tanh(means), disc_probs, disc_entropies


class HybridStateActionFunction(BaseNetwork): # Critic NN

    def __init__(self, state_dim, action_cont_dim, action_disc_dims=[3], hidden_units=[256,256]):
        super().__init__()

        self.net = create_linear_network(
            input_dim=state_dim+action_cont_dim, # Critic input: state and continuous actions. NOT discrete
            output_dim=hidden_units[-1],
            hidden_units=hidden_units[:-1],
            output_activation=nn.ReLU())

        self.discrete_q_heads = nn.ModuleList([
            nn.Linear(hidden_units[-1], dim) for dim in action_disc_dims
        ])

    def forward(self, x):
        features = self.net(x)
        return [head(features) for head in self.discrete_q_heads] # q_value for each discrete action/head


class HybridTwinnedStateActionFunction(BaseNetwork): # Critic has two NN for Double Q Clipping technique

    def __init__(self, state_dim, action_cont_dim, action_disc_dims=[3], hidden_units=[256, 256]):
        super().__init__()

        self.net1 = HybridStateActionFunction(state_dim, action_cont_dim, action_disc_dims, hidden_units)
        self.net2 = HybridStateActionFunction(state_dim, action_cont_dim, action_disc_dims, hidden_units)

    def forward(self, states, cont_actions):
        assert states.dim() == 2 and cont_actions.dim() == 2

        x = torch.cat([states, cont_actions], dim=1)
        q1_list = self.net1(x) # q_value for each discrete action
        q2_list = self.net2(x) # q_value for each discrete action

        return q1_list, q2_list
