import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SimpleVECEnv(gym.Env):

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
        low=np.array([
            0,
            0,
            0,
            0
        ],dtype = np.float32),
        
        high=np.array([
            1,
            1,
            1,
            1
        ],dtype = np.float32)
        )
        self.max_steps = 300
        self.current_step = 0
        # 车辆 CPU：1 GHz
        self.local_cpu = 1e9

        # 边缘服务器 CPU：5GHz
        self.edge_cpu = 5e9

        # 无线上行发射功率：2 W
        self.tx_power = 2.0

        # CPU 能耗系数
        self.kappa = 1e-27

        # 队列等待时机
        self.queue_time = 0
        # 每个时隙持续0.3秒
        self.slot_duration = 0.3

        self.state = None


    def generate_state(self):

        # 任务大小：0.5 ~ 2 Mbit
        task_size = self.np_random.uniform(
            0.5e6,
            2e6
        )

        # 每 bit 所需 CPU cycles
        cycles_per_bit = self.np_random.uniform(
            500,
            1500
        )

        # 上传速率：2 ~ 10 Mbps
        upload_rate = self.np_random.uniform(
            0.5e6,
            5e6
        )

        task_size_norm = (task_size - 0.5e6) / (2e6 - 0.5e6)
        cycles_per_bit_norm = (cycles_per_bit - 500) / (1500 - 500)
        upload_rate_norm = (upload_rate - 0.5e6) / (5e6 - 0.5e6)
        queue_time_norm = (
                        self.queue_time
                        / (self.queue_time + 1.0)
                    )
        state = np.array([
            task_size_norm,
            cycles_per_bit_norm,
            upload_rate_norm,
            queue_time_norm
            ],dtype=np.float32)

        return state

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        self.current_step = 0
        self.queue_time = 0

        self.state = self.generate_state()

        return self.state, {}

    def calculate(self, action):

        task_size = self.state[0] * (2e6 - 0.5e6) + 0.5e6
        cycles_per_bit = self.state[1] * (1500 - 500) + 500
        upload_rate = self.state[2] * (5e6 - 0.5e6) + 0.5e6

        # 总 CPU cycles
        total_cycles = (
            task_size * cycles_per_bit
        )

        # 本地计算
        if action == 0:

            delay = (
                total_cycles
                / self.local_cpu
            )

            energy = (
                self.kappa
                * self.local_cpu ** 2
                * total_cycles
            )
            upload_delay = 0
            edge_compute_delay = 0
            wait_delay = 0
        # 边缘卸载
        elif action == 1:
            
            upload_delay = (
                task_size
                / upload_rate
            )

            edge_compute_delay = (
                total_cycles
                / self.edge_cpu
            )

            wait_delay = self.queue_time
            
            delay = (
                upload_delay
                + edge_compute_delay
                + wait_delay
            )
            
            energy = (
                self.tx_power
                * upload_delay
            )

        else:
            raise ValueError(
                "action只能是0或1"
            )

        reward = -(delay + energy)
        

        return delay, energy, reward, wait_delay,upload_delay, edge_compute_delay

    
    def step(self, action):

        delay, energy, reward, wait_delay, upload_delay , edge_compute_delay = self.calculate(action)
        if action == 1:

            # 增加服务器负载
            self.queue_time += edge_compute_delay
            
        
        self.queue_time = max(
                        0,
                        self.queue_time - self.slot_duration
                        )
        next_state = self.generate_state()
        self.state = next_state

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return (
                next_state,
                reward,
                False,
                truncated,
                {
                    "delay": delay,
                    "energy": energy,
                    "upload_delay": upload_delay,
                    "wait_delay":  wait_delay,
                    "compute_delay":  edge_compute_delay
                }
                )

    
