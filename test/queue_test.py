from env.simple_vec_env import SimpleVECEnv


env = SimpleVECEnv()


state, info = env.reset()


print("initial state:")
print(state)


print("\n======== Always Edge ========")


for i in range(10):

    action = 0   # 一直边缘卸载


    next_state, reward, terminated, truncated, info = (
        env.step(action)
    )


    print(
        f"step {i+1}:",
        "queue=",
        next_state[3],
        "reward=",
        reward
    )