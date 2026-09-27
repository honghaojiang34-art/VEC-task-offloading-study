from env import simple_vec_env 
env = simple_vec_env.SimpleVECEnv()
state, info = env.reset()

for i in range(300):

    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)
    if not env.observation_space.contains(next_state):
        print("发现非法状态！")
        print("step =", env.current_step + 1)
        print("state =", next_state)
        break

    if terminated or truncated:
        print("Episode结束")
        print("总步数 =", env.current_step + 1)
        print("最终state =", next_state)
        break
    else:
        print("测试完成，没有发现非法状态")


