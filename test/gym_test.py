from env.simple_vec_env import SimpleVECEnv


env = SimpleVECEnv()


state, info = env.reset()


print("state:")
print(state)


action = env.action_space.sample()


print("action:")
print(action)


next_state, reward, terminated, truncated, info = (
    env.step(action)
)


print("next_state:")
print(next_state)

print("reward:")
print(reward)

print("info:")
print(info)