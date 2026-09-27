import random
import numpy as np
import torch
import torch.optim as optim

from pathlib import Path

from env.simple_vec_env import SimpleVECEnv
from agent.dqn import (
    DQN,
    ReplayBuffer,
    select_action,
    train_dqn
)


# 实验配置
SEEDS = [42, 123, 2024, 3407, 8888]

NUM_EPISODES = 500
BATCH_SIZE = 64
GAMMA = 0.99

LEARNING_RATE = 1e-3

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

TARGET_UPDATE_FREQ = 100


def train_one_seed(seed):

    # 1. 固定随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2. 创建全新环境
    env = SimpleVECEnv()
    env.action_space.seed(seed)

    # 3. 创建全新网络
    online_net = DQN(4, 2)
    target_net = DQN(4, 2)

    target_net.load_state_dict(
        online_net.state_dict()
    )

    # 4. 创建全新经验池
    replay_buffer = ReplayBuffer(
        capacity=10000
    )

    # 5. 创建全新优化器
    optimizer = optim.Adam(
        online_net.parameters(),
        lr=LEARNING_RATE
    )

    # 6. 初始化训练变量
    epsilon = EPSILON_START
    train_step = 0

    train_rewards = []

    return (
        env,
        online_net,
        target_net,
        replay_buffer,
        optimizer,
        epsilon,
        train_step,
        train_rewards
    )


if __name__ == "__main__":

    results = train_one_seed(42)

    env = results[0]
    online_net = results[1]
    replay_buffer = results[3]

    print("环境时隙长度:", env.slot_duration)
    print("网络:", online_net)
    print("经验池长度:", len(replay_buffer))

    env.close()