from env.simple_vec_env import SimpleVECEnv


env = SimpleVECEnv()

state = env.reset()


for step in range(10):

    print("\n======================")
    print("Step:", step)

    print(
        "D =",
        state[0] / 1e6,
        "Mbit"
    )

    print(
        "C =",
        state[1],
        "cycles/bit"
    )

    print(
        "R =",
        state[2] / 1e6,
        "Mbps"
    )


    local_delay, local_energy, local_reward = (
        env.calculate(0)
    )

    edge_delay, edge_energy, edge_reward = (
        env.calculate(1)
    )


    if local_reward > edge_reward:

        action = 0

    else:

        action = 1

    print("\n本地计算：")
    print(
        "Delay:",
        local_delay
    )
    print(
        "Energy:",
        local_energy
    )
    print(
        "Reward:",
        local_reward
    )


    print("\n边缘卸载：")
    print(
        "Delay:",
        edge_delay
    )
    print(
        "Energy:",
        edge_energy
    )
    print(
        "Reward:",
        edge_reward
    )


    print(
        "\n最终选择 Action:",
        action
    )

    next_state, reward, delay, energy = (
        env.step(action)
    )
    state = next_state