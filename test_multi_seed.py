
import random
from pathlib import Path

import numpy as np
import torch

from env.simple_vec_env import SimpleVECEnv
from agent.dqn import DQN


# =========================
# 1. 实验配置
# =========================

SEEDS = [42, 123, 2024, 3407, 8888]

NUM_TEST_EPISODES = 100

# 测试种子与训练种子分开
TEST_SEED_START = 10000


# =========================
# 2. 加载训练好的模型
# =========================

def load_model(seed):

    model_path = Path(
        f"models/vec_dqn_seed_{seed}.pth"
    )

    # 检查模型文件是否存在
    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到模型文件: {model_path}"
        )

    # 创建与训练时相同的网络
    online_net = DQN(
        state_dim=4,
        action_dim=2
    )

    # 加载对应随机种子的参数
    online_net.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )
    )

    # 切换到评估模式
    online_net.eval()

    return online_net


# =========================
# 3. 测试单个模型
# =========================

def evaluate_model(online_net):

    env = SimpleVECEnv()

    episode_rewards = []
    episode_delays = []
    episode_energies = []

    for episode in range(NUM_TEST_EPISODES):

        # 所有模型使用相同的测试种子
        state, info = env.reset(
            seed=TEST_SEED_START + episode
        )

        total_reward = 0.0
        total_delay = 0.0
        total_energy = 0.0
        step_count = 0

        while True:

            # 将状态转换为 Tensor
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32
            )

            # 关闭梯度计算
            with torch.no_grad():

                q_values = online_net(
                    state_tensor
                )

                # 完全根据 Q 值选择动作
                action = q_values.argmax().item()

            # 与环境交互
            (
                next_state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            # 累计本轮指标
            total_reward += reward
            total_delay += info["delay"]
            total_energy += info["energy"]

            step_count += 1
            state = next_state

            if terminated or truncated:
                break

        # 每个 episode 的平均指标
        episode_rewards.append(
            total_reward / step_count
        )

        episode_delays.append(
            total_delay / step_count
        )

        episode_energies.append(
            total_energy / step_count
        )

    env.close()

    return {
        "reward": np.mean(episode_rewards),
        "delay": np.mean(episode_delays),
        "energy": np.mean(episode_energies)
    }


# =========================
# 4. 测试五个模型
# =========================

if __name__ == "__main__":

    all_results = []

    for seed in SEEDS:

        print(f"\n开始测试 Seed {seed}")

        online_net = load_model(seed)

        result = evaluate_model(online_net)

        all_results.append([
            result["reward"],
            result["delay"],
            result["energy"]
        ])

        print(
            f"Reward: {result['reward']:.4f}, "
            f"Delay: {result['delay']:.4f}, "
            f"Energy: {result['energy']:.4f}"
        )

    # 转换为 NumPy 数组
    results_array = np.array(all_results)

    # 每一列分别计算均值和标准差
    mean_results = np.mean(
        results_array,
        axis=0
    )

    std_results = np.std(
        results_array,
        axis=0,
        ddof=1
    )

    # =========================
    # 5. 打印最终统计结果
    # =========================

    print("\n===== 五模型测试结果 =====")

    metric_names = [
        "Reward",
        "Delay",
        "Energy"
    ]

    for i, name in enumerate(metric_names):

        print(
            f"{name}: "
            f"{mean_results[i]:.4f} "
            f"± {std_results[i]:.4f}"
        )

    # =========================
    # 6. 保存测试数据
    # =========================

    Path("results").mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        "results/vec_dqn_multi_seed_test.npz",
        seeds=np.array(SEEDS),
        results=results_array,
        mean=mean_results,
        std=std_results
    )

    print("\n测试结果已保存")
