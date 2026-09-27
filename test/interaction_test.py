from env.simple_vec_env import SimpleVECEnv
from agent.dqn import DQN, ReplayBuffer, select_action


env = SimpleVECEnv()

online_net = DQN(
    state_dim=4,
    action_dim=2
)

replay_buffer = ReplayBuffer(
    capacity=10000
)

epsilon = 0.9

state, info = env.reset(seed=42)


for step in range(20):

    # 选择动作
    action = select_action(
        state,
        online_net,
        env.action_space,
        epsilon
    )

    # 与 VEC 环境交互
    next_state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    # 保存经验
    replay_buffer.push(
        state,
        action,
        reward,
        next_state,
        done
    )

    print(
        f"step={step + 1}, "
        f"action={action}, "
        f"reward={reward:.4f}, "
        f"buffer={len(replay_buffer)}"
    )

    # 状态向前移动
    state = next_state

    if done:
        break