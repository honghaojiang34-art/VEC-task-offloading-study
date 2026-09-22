import random
import numpy as np
import matplotlib.pyplot as plt

from simple_vec_env import SimpleVECEnv


# =========================
# 测试一个策略
# =========================

def evaluate_policy(policy, num_tasks=1000, seed=42):

    # 固定随机种子
    # 保证不同策略面对相同的一组任务
    np.random.seed(seed)
    random.seed(seed)

    env = SimpleVECEnv()

    state = env.reset()

    total_reward = 0
    total_delay = 0
    total_energy = 0

    local_count = 0
    edge_count = 0

    reward_history = []
    delay_history = []
    energy_history = []
    action_history = []
    task_size_history = []
    cycles_history = []
    rate_history = []
    queue_history = []

    for step in range(num_tasks):

        # ---------------------
        # 随机策略
        # ---------------------
        if policy == "random":

            action = random.choice([0, 1])


        # ---------------------
        # Rule策略
        # ---------------------
        elif policy == "rule":

            # 同一个任务分别计算两个动作
            _, _, local_reward = (
                env.calculate(0)
            )

            _, _, edge_reward = (
                env.calculate(1)
            )

            # 选择Reward更大的动作
            if local_reward > edge_reward:
                action = 0

            else:
                action = 1

        else:
            raise ValueError(
                "policy必须是random或rule"
            )


        # =====================
        # 真正执行动作
        # =====================
        queue_history.append(
        state[3]
        )
        next_state, reward, delay, energy = (
            env.step(action)
        )


        # 累加实验数据
        total_reward += reward
        total_delay += delay
        total_energy += energy
        
        reward_history.append(reward)
        delay_history.append(delay)
        energy_history.append(energy)
        action_history.append(action)
        task_size_history.append(
        state[0] / 1e6
        )

        cycles_history.append(
            state[1]
        )

        rate_history.append(
            state[2] / 1e6
        )  

        # 统计动作
        if action == 0:

            local_count += 1

        else:

            edge_count += 1


        # 更新state
        state = next_state


    # =========================
    # 计算平均值
    # =========================

    average_reward = (
        total_reward / num_tasks
    )

    average_delay = (
        total_delay / num_tasks
    )

    average_energy = (
        total_energy / num_tasks
    )


    return {
        "average_reward": average_reward,
        "average_delay": average_delay,
        "average_energy": average_energy,
        "local_count": local_count,
        "edge_count": edge_count,

        "reward_history": reward_history,
        "delay_history": delay_history,
        "energy_history": energy_history,
        "action_history": action_history,
        "task_size_history": task_size_history,
        "cycles_history": cycles_history,
        "rate_history": rate_history,
        "queue_history": queue_history
        
    }


def cumulative_average(data):

    data = np.array(data)

    return (
        np.cumsum(data)
        / np.arange(1, len(data) + 1)
    )
# =========================
# Random策略
# =========================

random_result = evaluate_policy(
    policy="random",
    num_tasks=1000,
    seed=42
)


# =========================
# Rule策略
# =========================

rule_result = evaluate_policy(
    policy="rule",
    num_tasks=1000,
    seed=42
)


print("\n========== Random Policy ==========")

print(
    "Average Reward:",
    random_result["average_reward"]
)

print(
    "Average Delay:",
    random_result["average_delay"]
)

print(
    "Average Energy:",
    random_result["average_energy"]
)

print(
    "Local Count:",
    random_result["local_count"]
)

print(
    "Edge Count:",
    random_result["edge_count"]
)


print("\n========== Rule Policy ==========")

print(
    "Average Reward:",
    rule_result["average_reward"]
)

print(
    "Average Delay:",
    rule_result["average_delay"]
)

print(
    "Average Energy:",
    rule_result["average_energy"]
)

print(
    "Local Count:",
    rule_result["local_count"]
)

print(
    "Edge Count:",
    rule_result["edge_count"]
)


plt.figure()

plt.plot(
    random_result["reward_history"],
    label="Random"
)

plt.plot(
    rule_result["reward_history"],
    label="Rule"
)

plt.xlabel("Task")
plt.ylabel("Reward")
plt.title("Reward Comparison")
plt.legend()

plt.show()

random_avg_reward = cumulative_average(
    random_result["reward_history"]
)

rule_avg_reward = cumulative_average(
    rule_result["reward_history"]
)


plt.figure()

plt.plot(
    random_avg_reward,
    label="Random"
)

plt.plot(
    rule_avg_reward,
    label="Rule"
)

plt.xlabel("Number of Tasks")
plt.ylabel("Cumulative Average Reward")
plt.title("Random vs Rule")
plt.legend()

plt.show()


labels = [
    "Reward",
    "Delay",
    "Energy"
]

random_values = [
    random_result["average_reward"],
    random_result["average_delay"],
    random_result["average_energy"]
]

rule_values = [
    rule_result["average_reward"],
    rule_result["average_delay"],
    rule_result["average_energy"]
]

x = np.arange(len(labels))

width = 0.35

plt.figure()

plt.bar(
    x - width / 2,
    random_values,
    width,
    label="Random"
)

plt.bar(
    x + width / 2,
    rule_values,
    width,
    label="Rule"
)

plt.xticks(x, labels)

plt.title("Policy Comparison")

plt.legend()

plt.show()


rule_actions = np.array(
    rule_result["action_history"]
)

rule_rates = np.array(
    rule_result["rate_history"]
)

rule_cycles = np.array(
    rule_result["cycles_history"]
)

local_mask = (rule_actions == 0)
edge_mask = (rule_actions == 1)


plt.figure()

plt.scatter(
    rule_rates[local_mask],
    rule_cycles[local_mask],
    label="Local",
    alpha=0.6
)

plt.scatter(
    rule_rates[edge_mask],
    rule_cycles[edge_mask],
    label="Edge",
    alpha=0.6
)

plt.xlabel("Upload Rate R (Mbps)")
plt.ylabel("Computation Density C (cycles/bit)")

plt.title("VEC Offloading Decision")

plt.legend()

plt.show()


