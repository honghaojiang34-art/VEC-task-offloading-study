import numpy as np

from agent.dqn import ReplayBuffer


buffer = ReplayBuffer(capacity=1000)

state = np.array(
    [0.2, 0.4, 0.6, 0.1],
    dtype=np.float32
)

next_state = np.array(
    [0.3, 0.5, 0.7, 0.2],
    dtype=np.float32
)

buffer.push(
    state,
    1,
    -0.5,
    next_state,
    False
)

print("buffer长度：", len(buffer))
print("存进去的数据：", buffer.buffer[0])