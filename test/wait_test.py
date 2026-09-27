from env.simple_vec_env import SimpleVECEnv

env = SimpleVECEnv()

state, _ = env.reset(seed=42)

for step in range(300):

    # 始终选择 Edge
    next_state, reward, terminated, truncated, info = env.step(1)

    if (step + 1) % 50 == 0:
        print(
            f"Step: {step + 1}, "
            f"Queue: {env.queue_time:.4f}, "
            f"Delay: {info['delay']:.4f}"
        )

    state = next_state

    if terminated or truncated:
        break