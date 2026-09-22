import numpy as np
import matplotlib.pyplot as plt
from compare_policies import evaluate_policy

rule_result = evaluate_policy(
    policy="rule",
    num_tasks=1000,
    seed=42
)

queue = np.array(
    rule_result["queue_history"]
)

actions = np.array(
    rule_result["action_history"]
)

bins = [
    0,
    0.5,
    1,
    1.5,
    2,
    3,
    5
]

edge_ratio = []

queue_labels = []


for i in range(len(bins)-1):

    mask = (
        (queue >= bins[i])
        &
        (queue < bins[i+1])
    )

    if np.sum(mask) > 0:

        ratio = np.mean(
            actions[mask]
        )

        edge_ratio.append(ratio)

        queue_labels.append(
            f"{bins[i]}-{bins[i+1]}"
        )

plt.figure()

plt.bar(
    queue_labels,
    edge_ratio
)

plt.xlabel(
    "Queue Time Q(s)"
)

plt.ylabel(
    "Edge Offloading Ratio"
)

plt.title(
    "Effect of Queue on Offloading Decision"
)

plt.xticks(
    rotation=45
)

plt.show()