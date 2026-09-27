import torch
import numpy as np


from env.simple_vec_env import SimpleVECEnv
from agent.dqn import DQN


# 创建环境
env = SimpleVECEnv()


# 创建相同结构的 DQN
online_net = DQN(
    state_dim=4,
    action_dim=2
)


# 加载训练好的参数
online_net.load_state_dict(
    torch.load(
        "vec_dqn.pth",
        weights_only=True
    )
)


# 切换到评估模式
online_net.eval()

print("DQN模型加载成功")


local_states = []
edge_states = []

local_count = 0
edge_count = 0

num_test_steps = 1000

state, info = env.reset(seed=100)
for step in range(num_test_steps):

    # 转成 Tensor
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    )

    # 测试阶段不需要计算梯度
    with torch.no_grad():

        q_values = online_net(
            state_tensor
        )

        action = q_values.argmax().item()

    # 记录 DQN 在什么状态下选择了什么动作
    if action == 0:

        local_count += 1
        local_states.append(state.copy())

    else:

        edge_count += 1
        edge_states.append(state.copy())

    # 与环境交互
    next_state, reward, terminated, truncated, info = env.step(action)

    state = next_state

    # episode结束后重新开始
    if terminated or truncated:

        state, info = env.reset()


local_ratio = local_count / num_test_steps
edge_ratio = edge_count / num_test_steps

print(f"Local Ratio: {local_ratio:.2%}")
print(f"Edge Ratio: {edge_ratio:.2%}")


local_states = np.array(local_states)
edge_states = np.array(edge_states)




print("\nLocal states mean:")

print(
    np.mean(
        local_states,
        axis=0
    )
)

print("\nEdge states mean:")

print(
    np.mean(
        edge_states,
        axis=0
    )
)