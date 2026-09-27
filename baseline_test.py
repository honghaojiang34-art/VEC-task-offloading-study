import random
import numpy as np
import torch

from env.simple_vec_env import SimpleVECEnv
from agent.dqn import DQN


env = SimpleVECEnv()

online_net = DQN(
    state_dim=4,
    action_dim=2
)

online_net.load_state_dict(
    torch.load(
        "vec_dqn.pth_v2",
        weights_only=True
    )
)

online_net.eval()


def evaluate_policy(policy, seed=10000, num_episodes=100):

    episode_rewards = []
    all_delays = []
    all_energies = []
    all_queue_time = []
    all_upload_delay = []
    all_computer_delay = []
    for episode in range(num_episodes):
        step_count = 0
        current_seed = seed + episode
        state, info = env.reset(
            seed = current_seed
        )
        env.action_space.seed(current_seed)
        episode_reward = 0

        while True:

            # 根据不同策略选择动作
            if policy == "random":

                action = env.action_space.sample()

            elif policy == "local":

                action = 0

            elif policy == "edge":

                action = 1

            elif policy == "dqn":

                state_tensor = torch.tensor(
                    state,
                    dtype=torch.float32
                )

                with torch.no_grad():

                    q_values = online_net(
                        state_tensor
                    )

                action = q_values.argmax().item()

            # 与环境交互
            next_state, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            episode_reward += reward

            state = next_state
            all_delays.append(
                info["delay"]
            )


            all_energies.append(
                info["energy"]
            )

            all_queue_time.append(info["wait_delay"])
            all_upload_delay.append(info["upload_delay"])
            all_computer_delay.append(info["compute_delay"])
            if terminated or truncated:
                break

        # 转换成每 step 平均 Reward
        avg_reward = (
            episode_reward
            /  step_count
        )

        episode_rewards.append(
            avg_reward
        )

    return {
    "reward": np.mean(episode_rewards),
    "delay": np.mean(all_delays),
    "energy": np.mean(all_energies),
    "queue_delay": np.mean(all_queue_time),
    "upload_delay": np.mean(all_upload_delay),
    "compute_delay": np.mean(all_computer_delay)
}


random_tset = evaluate_policy(
    "random"
)
print(random_tset)
local = evaluate_policy(
    "local"
)
print(local)

edge = evaluate_policy(
    "edge"
)
print(edge)


