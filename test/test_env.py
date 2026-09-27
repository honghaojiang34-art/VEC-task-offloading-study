from env.simple_vec_env import SimpleVECEnv
import random
env = SimpleVECEnv()

state = env.reset()

for step in range(5):

    print("\n====================")
    print("Step:", step)

    print(
        "当前状态:",
        state
    )


    # 暂时随机选动作
    action = random.choice(
        [0, 1]
    )

    print(
        "选择动作:",
        action
    )


    next_state, reward, delay, energy = (
        env.step(action)
    )


    print(
        "Delay:",
        delay
    )

    print(
        "Energy:",
        energy
    )

    print(
        "Reward:",
        reward
    )

    print(
        "Next state:",
        next_state
    )


    # 更新状态
    state = next_state