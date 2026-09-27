"""VEC-DQN 多随机种子训练；从现有 train_multi_seed.py 的初始化结构扩展。"""
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from env.simple_vec_env import SimpleVECEnv
from agent.dqn import DQN, ReplayBuffer, select_action, train_dqn

# 实验配置：小规模检查时可临时改为 SEEDS=[42,123], NUM_EPISODES=25
SEEDS = [42, 123, 2024, 3407, 8888]
NUM_EPISODES = 500
# SEEDS = [42, 123]
# NUM_EPISODES = 25
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-3
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 100  # 按优化器更新次数同步 target network
WINDOW_SIZE = 20

MODELS_DIR = Path('models')
RESULTS_DIR = Path('results')


def train_one_seed(seed):
    # 1. 每个种子独立初始化随机数、环境、网络、经验池和优化器
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = SimpleVECEnv()
    env.action_space.seed(seed)
    online_net = DQN(4, 2)
    target_net = DQN(4, 2)
    target_net.load_state_dict(online_net.state_dict())
    replay_buffer = ReplayBuffer(capacity=10000)
    optimizer = optim.Adam(online_net.parameters(), lr=LEARNING_RATE)

    epsilon = EPSILON_START
    train_step = 0
    train_rewards = []
    train_losses = []
    local_ratios = []
    edge_ratios = []

    try:
        for episode in range(NUM_EPISODES):
            # 同一种子内，各 episode 的环境任务序列可复现
            state, _ = env.reset(seed=seed + episode)
            episode_reward = 0.0
            episode_losses = []
            local_count = 0
            edge_count = 0
            steps = 0

            while True:
                action = select_action(state, online_net, env.action_space, epsilon)
                if action == 0:
                    local_count += 1
                else:
                    edge_count += 1

                next_state, reward, terminated, truncated, _ = env.step(action)
                # 用 terminated 而不是 truncated 屏蔽 TD bootstrap：
                # 本环境通常只有时间限制结束，截断并不等于真正的终止状态。
                replay_buffer.push(state, action, reward, next_state, terminated)
                episode_reward += reward
                steps += 1
                state = next_state

                if len(replay_buffer) >= BATCH_SIZE:
                    loss = train_dqn(
                        online_net, target_net, replay_buffer,
                        optimizer, BATCH_SIZE, GAMMA
                    )
                    episode_losses.append(float(loss))
                    train_step += 1
                    if train_step % TARGET_UPDATE_FREQ == 0:
                        target_net.load_state_dict(online_net.state_dict())

                if terminated or truncated:
                    break

            # 以实际步数为分母，兼容将来修改 episode 长度
            train_rewards.append(episode_reward / steps)
            train_losses.append(float(np.mean(episode_losses)) if episode_losses else 0.0)
            local_ratios.append(local_count / steps)
            edge_ratios.append(edge_count / steps)
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

            if (episode + 1) % 50 == 0 or episode == 0:
                print(
                    f'Seed {seed} | Episode {episode + 1}/{NUM_EPISODES} | '
                    f'Avg Reward {train_rewards[-1]:.4f} | '
                    f'Loss {train_losses[-1]:.4f} | Epsilon {epsilon:.4f}'
                )

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(online_net.state_dict(), MODELS_DIR / f'vec_dqn_seed_{seed}.pth')
        return train_rewards, train_losses, local_ratios, edge_ratios
    finally:
        env.close()


def main():
    all_train_rewards = {}
    all_train_losses = {}
    all_local_ratios = {}
    all_edge_ratios = {}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        print(f'\n开始训练 Seed: {seed}')
        rewards, losses, local, edge = train_one_seed(seed)
        all_train_rewards[seed] = rewards
        all_train_losses[seed] = losses
        all_local_ratios[seed] = local
        all_edge_ratios[seed] = edge
        # 每个种子完成后保存一次，避免后续训练中断导致已有曲线数据丢失
        np.savez(
            RESULTS_DIR / 'vec_dqn_multi_seed_partial.npz',
            seeds=np.array(list(all_train_rewards)),
            rewards=np.array(list(all_train_rewards.values())),
            losses=np.array(list(all_train_losses.values())),
            local_ratios=np.array(list(all_local_ratios.values())),
            edge_ratios=np.array(list(all_edge_ratios.values()))
        )
        print(f'Seed {seed} 训练完成')

    reward_array = np.array(list(all_train_rewards.values()), dtype=float)
    print('Reward 矩阵形状:', reward_array.shape)
    np.savez(
        RESULTS_DIR / 'vec_dqn_multi_seed.npz',
        seeds=np.array(SEEDS),
        rewards=reward_array,
        losses=np.array(list(all_train_losses.values())),
        local_ratios=np.array(list(all_local_ratios.values())),
        edge_ratios=np.array(list(all_edge_ratios.values()))
    )

    window = min(WINDOW_SIZE, reward_array.shape[1])
    smoothed = np.array([
        np.convolve(row, np.ones(window) / window, mode='valid')
        for row in reward_array
    ])
    mean_rewards = smoothed.mean(axis=0)
    std_rewards = smoothed.std(axis=0)  # 描述五次训练间离散程度
    episodes = np.arange(window, reward_array.shape[1] + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, mean_rewards, label=f'Mean Reward ({len(SEEDS)} seeds)')
    plt.fill_between(
        episodes, mean_rewards - std_rewards, mean_rewards + std_rewards,
        alpha=0.2, label='Mean ± 1 Std'
    )
    plt.xlabel('Episode')
    plt.ylabel('Average Reward per Step')
    plt.title('VEC DQN Multi-Seed Training')
    plt.legend()
    plt.grid()
    plt.savefig(RESULTS_DIR / 'vec_dqn_multi_seed.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
