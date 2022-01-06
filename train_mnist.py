import sys
import torch
import random
import numpy as np
from collections import deque
from tqdm import tqdm


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
    eps = sys.float_info.epsilon
    seed = 0
    step_size = 1e-2
    set_random_seed(seed)

    mnist = MNIST(seed=seed)
    model = FlexibleNN.RecurrentClassifierNetwork(step_size, seed, 28, 10, 32, 28)

    step = 0
    running_error = None
    running_acc = None
    for epochs in range(500):
        bar = tqdm(mnist.sequential_iterator(split="train"))
        for _, inp, label in bar:
            step += 1
            # if step %100000 == 0:
            #    model.replace_least_important_feature()
            model.forward(inp)
            if label is not None:
                targets = np.zeros(10)
                targets[label] = 1
            else:
                targets = model.read_output_values()
            model.backward(targets)
            model.update_parameters()

            if label is not None:
                # mse = np.mean((targets - np.array(model.read_output_values()))**2)
                cross_entropy = -np.sum(targets * np.log(np.array(model.read_output_values()) + eps))
                acc = 1 if np.argmax(model.read_output_values()) == label else 0

                running_error = (running_error * 0.9995 + 0.0005 * cross_entropy if running_error else cross_entropy)
                running_acc = running_acc * 0.9995 + 0.0005 * acc if running_acc else acc
                bar.set_description(f"Epoch: {epochs}, Step: {step}, acc: {acc}, cross_ent: {cross_entropy}, running_acc: {running_acc}, running_error: {running_error}")

                model.reset_state()


if __name__ == "__main__":
    main()
