import numpy as np


class SimpleVECEnv:

    def __init__(self):

        # 车辆 CPU：1 GHz
        self.local_cpu = 1e9

        # 边缘服务器 CPU：5GHz
        self.edge_cpu = 10e9

        # 无线上行发射功率：2 W
        self.tx_power = 2.0

        # CPU 能耗系数
        self.kappa = 1e-27

        # 队列等待时机
        self.queue_time = 0


        self.state = None


    def generate_state(self):

        # 任务大小：0.5 ~ 2 Mbit
        task_size = np.random.uniform(
            0.5e6,
            2e6
        )

        # 每 bit 所需 CPU cycles
        cycles_per_bit = np.random.uniform(
            500,
            1500
        )

        # 上传速率：2 ~ 10 Mbps
        upload_rate = np.random.uniform(
            0.5e6,
            5e6
        )

        state = np.array([
            task_size,
            cycles_per_bit,
            upload_rate,
            self.queue_time
        ])

        return state

    def reset(self):
        self.queue_time = 0
        self.state= self.generate_state()
        return self.state

    def calculate(self, action):

        task_size = self.state[0]
        cycles_per_bit = self.state[1]
        upload_rate = self.state[2]

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
        

        return delay, energy, reward

    
    def step(self, action):

        delay, energy, reward = self.calculate(action)
        if action == 1:

            task_size = self.state[0]
            cycles_per_bit = self.state[1]

            total_cycles = (
                task_size
                * cycles_per_bit
            )

            edge_compute_delay = (
                total_cycles
                /
                self.edge_cpu
            )

            # 增加服务器负载
            self.queue_time += edge_compute_delay
        next_state = self.generate_state()
        self.queue_time = max(
                        0,
                        self.queue_time - 0.1
                        )
        self.state = next_state
        

        return (next_state, reward, delay, energy)

    
