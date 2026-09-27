import torch

from agent.dqn import DQN


net = DQN(
    state_dim=4,
    action_dim=2
)

state = torch.tensor(
    [0.5, 0.3, 0.8, 0.2],
    dtype=torch.float32
)

q_values = net(state)

print("state =", state)
print("Q values =", q_values)
print("shape =", q_values.shape)