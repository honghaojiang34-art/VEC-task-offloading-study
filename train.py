import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from env.simple_vec_env import SimpleVECEnv
from agent.dqn import DQN, ReplayBuffer, select_action, train_dqn


# =========================
# 1. 随机种子
# =========================

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# =========================
# 2. 创建环境
# =========================

env = SimpleVECEnv()


# =========================
# 3. 创建 DQN
# =========================

state_dim = 4
action_dim = 2

online_net = DQN(
    state_dim,
    action_dim
)

target_net = DQN(
    state_dim,
    action_dim
)

# 一开始两个网络参数完全相同
target_net.load_state_dict(
    online_net.state_dict()
)



# =========================
# 4. Replay Buffer
# =========================

replay_buffer = ReplayBuffer(
    capacity=10000
)



# =========================
# 5. Optimizer
# =========================

learning_rate = 1e-3

optimizer = optim.Adam(
    online_net.parameters(),
    lr=learning_rate
)


# =========================
# 6. 超参数
# =========================

num_episodes = 500

batch_size = 64

gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

target_update_freq = 100

train_step = 0


all_avg_rewards = []
all_losses = []
all_local_ratios = []
all_edge_ratios = []
for episode in range(num_episodes):

    state, info = env.reset(seed=seed + episode)
    local_count = 0
    edge_count = 0
    episode_losses = []
    episode_reward = 0

    while True:

        # 1. epsilon-greedy 选择动作
        action = select_action(
            state,
            online_net,
            env.action_space,
            epsilon
        )
        if action == 0:
            local_count += 1
        else:
            edge_count += 1
        # 2. 执行动作
        next_state, reward, terminated, truncated, info = env.step(action)

        # 3. 存入 Replay Buffer
        replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            terminated
        )

        # 4. 累计当前 episode 的 reward
        episode_reward += reward

        # 5. 更新当前状态
        state = next_state

        # 6. 经验足够以后，开始训练
        if len(replay_buffer) >= batch_size:

            loss = train_dqn(
                online_net,
                target_net,
                replay_buffer,
                optimizer,
                batch_size,
                gamma
            )
            episode_losses.append(loss)
            train_step += 1

            # 7. 周期更新 target network
            if train_step % target_update_freq == 0:

                target_net.load_state_dict(
                    online_net.state_dict()
                )

        # 8. 判断当前 episode 是否结束
        if terminated or truncated:
            break

    # 9. 保存这一局总 reward,loss,local_ratio,edge_ratio

    avg_reward = episode_reward / env.max_steps
    avg_loss = (
    np.mean(episode_losses)
    if episode_losses
    else 0
)

    local_ratio = local_count / env.max_steps
    edge_ratio = edge_count / env.max_steps
    all_avg_rewards.append(avg_reward)
    all_losses.append(avg_loss)
    all_local_ratios.append(local_ratio)
    all_edge_ratios.append(edge_ratio)
    # 10. epsilon 衰减
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    print(
    f"Episode: {episode + 1}, "
    f"avg_reward: {avg_reward :.4f}, "
    f"Loss: {avg_loss:.4f}, "
    f"Epsilon: {epsilon:.4f}, "
    f"Local: {local_ratio:.2%}, "
    f"Edge: {edge_ratio:.2%}"
)


torch.save(
    online_net.state_dict(),
    "vec_dqn.pth_v2"
)


window_size = 20

moving_avg = np.convolve(
    all_avg_rewards,
    np.ones(window_size) / window_size,
    mode="valid"
)
plt.figure(figsize=(10, 5))

plt.plot(
    all_avg_rewards,
    label="Episode Reward",
    alpha=0.5
)


plt.plot(
    range(
        window_size - 1,
        len(all_avg_rewards)
    ),
    moving_avg,
    label="Moving Average (20)"
)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("VEC DQN Training Reward")
plt.legend()
plt.grid()

plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    all_local_ratios,
    label="Local Ratio",
    alpha=0.7
)

plt.plot(
    all_edge_ratios,
    label="Edge Ratio",
    alpha=0.7
)

plt.xlabel("Episode")
plt.ylabel("Action Ratio")
plt.title("Local / Edge Action Ratio")
plt.legend()
plt.grid()

plt.show()