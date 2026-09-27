import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import numpy as np



class DQN(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        terminated
    ):
        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                terminated
            )
        )

    def sample(self, batch_size):
        return random.sample(
            self.buffer,
            batch_size
        )

    def __len__(self):
        return len(self.buffer)


def select_action(
state,
online_net,
action_space,
epsilon
):
    if random.random() < epsilon:
        action = action_space.sample()

    else:
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():
            q_values = online_net(state_tensor)

        action = q_values.argmax().item()

    return action


def train_dqn(
    online_net,
    target_net,
    replay_buffer,
    optimizer,
    batch_size,
    gamma
):

    batch = replay_buffer.sample(batch_size)

    states, actions, rewards, next_states, terminated = zip(*batch)

    states = torch.tensor(
        np.array(states),
        dtype=torch.float32
    )

    actions = torch.tensor(
        actions,
        dtype=torch.long
    )

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32
    )

    next_states = torch.tensor(
        np.array(next_states),
        dtype=torch.float32
    )

    terminated = torch.tensor(
        terminated,
        dtype=torch.float32
    )

    current_q = online_net(states).gather(
        1,
        actions.unsqueeze(1)
    ).squeeze(1)

    with torch.no_grad():

        next_q = target_net(
            next_states
        ).max(dim=1)[0]

        target_q = (
            rewards
            + gamma
            * next_q
            * (1 - terminated)
        )


    loss = F.mse_loss(
    current_q,
    target_q
)
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()

