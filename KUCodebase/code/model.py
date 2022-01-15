import torch
import torch.nn as nn
from torch.distributions import Normal
from rltorch.network import create_linear_network
#
import random

class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(torch.load(path))

class QNetwork(BaseNetwork):
    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256],
                 initializer='xavier', layer_norm=0, drop_rate=0.0):
        super(QNetwork, self).__init__()

        # https://github.com/ku2482/rltorch/blob/master/rltorch/network/builder.py
        self.Q = create_linear_network(
            num_inputs+num_actions, 1, hidden_units=hidden_units,
            initializer=initializer)

        # override network architecture (dropout and layer normalization). TH 20210731
        new_q_networks = []
        for i, mod in enumerate(self.Q._modules.values()):
            new_q_networks.append(mod)
            if ((i % 2) == 0) and (i < (len(list(self.Q._modules.values()))) - 1):
                if drop_rate > 0.0:
                    new_q_networks.append(nn.Dropout(p=drop_rate))  # dropout
                if layer_norm:
                    new_q_networks.append(nn.LayerNorm(mod.out_features))  # layer norm
            i += 1
        self.Q = nn.Sequential(*new_q_networks)

    def forward(self, x):
        q = self.Q(x)
        return q

class TwinnedQNetwork(BaseNetwork):

    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256],
                 initializer='xavier', layer_norm=0, drop_rate=0.0):
        super(TwinnedQNetwork, self).__init__()

        self.Q1 = QNetwork(
            num_inputs, num_actions, hidden_units, initializer, layer_norm=layer_norm, drop_rate=drop_rate)
        self.Q2 = QNetwork(
            num_inputs, num_actions, hidden_units, initializer, layer_norm=layer_norm, drop_rate=drop_rate)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2


class RandomizedEnsembleNetwork(BaseNetwork):

    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256],
                 initializer='xavier', layer_norm=0, drop_rate=0.0, N=10):
        super(RandomizedEnsembleNetwork, self).__init__()

        self.N = N
        self.indices = list(range(N))
        for i in range(N):
            setattr(self, "Q"+str(i),
                    QNetwork(num_inputs, num_actions, hidden_units, initializer,
                             layer_norm=layer_norm, drop_rate=drop_rate))

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)

        random.shuffle(self.indices)
        q1 = getattr(self, "Q" + str(self.indices[0]))(x)
        q2 = getattr(self, "Q" + str(self.indices[1]))(x)

        return q1, q2

    def allQs(self, states, actions):
        x = torch.cat([states, actions], dim=1)

        Qs = []
        for i in range(self.N):
            Qs.append(getattr(self, "Q" + str(i))(x))
        return Qs

    def averageQ(self, states, actions):
        x = torch.cat([states, actions], dim=1)

        q = getattr(self, "Q" + str(0))(x)
        for i in range(1, self.N):
            q = q + getattr(self, "Q" + str(i))(x)
        q = q / self.N

        return q

class GaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256],
                 initializer='xavier'):
        super(GaussianPolicy, self).__init__()

        # https://github.com/ku2482/rltorch/blob/master/rltorch/network/builder.py
        self.policy = create_linear_network(
            num_inputs, num_actions*2, hidden_units=hidden_units,
            initializer=initializer)

    def forward(self, states):
        mean, log_std = torch.chunk(self.policy(states), 2, dim=-1)
        log_std = torch.clamp(
            log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, states):
        # calculate Gaussian distribusion of (mean, std)
        means, log_stds = self.forward(states)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # sample actions
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # calculate entropies
        log_probs = normals.log_prob(xs)\
            - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)
