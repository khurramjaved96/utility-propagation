import sys
import torch
import random
import numpy as np
from collections import deque


import FlexibleNN
from FlexibleNN import Metric, Database

from src_py.utils.logging_manager import LoggingManager
from src_py.envs.mnist import MNIST


def set_random_seed(seed: int) -> None:
    """
    Seed the different random generators.
    :param seed:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    seed = 0
    step_size = 3e-5
    set_random_seed(seed)

    mnist = MNIST(seed=seed)
    model = FlexibleNN.RecurrentClassifierNetwork(
        step_size,
        seed,
        28,
        10,
        1024,
        10
    )

    step = 0
    running_error = 0
    running_acc = 0
    for epochs in range(500):
        for _, inp, label in mnist.sequential_iterator(split='train'):
            step += 1
            if step %40000 == 0:
                model.replace_least_important_feature()
            model.forward(inp)
            if label is not None:
                targets = np.zeros(10)
                targets[label] = 1
            else:
                targets = model.read_output_values()
            model.backward(targets)
            model.update_parameters()

            if label is not None:
                mse = np.mean((targets - np.array(model.read_output_values()))**2)
                acc = 1 if np.argmax(model.read_output_values()) == label else 0

                running_error = running_error * 0.9995 + 0.0005 * mse
                running_acc = running_acc * 0.9995 + 0.0005 * acc
                print(f"Step: {step}, acc: {acc}, mse: {mse}, running_acc: {running_acc}, running_error: {running_error}")

                model.reset_trace()



if __name__ == "__main__":
    main()
