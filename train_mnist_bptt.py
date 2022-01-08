import sys
import random
import numpy as np
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src_py.utils.logging_manager import LoggingManager
from src_py.envs.mnist import MNIST
from src_py.models.rnn import RNN


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
    step_size = 1e-4
    n_timesteps = 320000
    set_random_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")  # its just faster ¯\_(ツ)_/¯

    mnist = MNIST(seed=seed)
    model = RNN(input_size=28, hidden_size=256, output_size=10).to(device)
    opt = optim.RMSprop(model.parameters(), lr=step_size)

    step = 0
    running_error = None
    running_acc = None
    hidden = model.reset_state().to(device)
    # loss = nn.CrossEntropyLoss()
    # cross_entropy = loss(prediction, torch.tensor([label]))
    loss = nn.MSELoss()
    for epochs in range((n_timesteps // 60000) + 1):
        iterator = tqdm(mnist.sequential_iterator(split="train"))
        for _, inp, label in iterator:
            step += 1
            if step // 28 >= n_timesteps:
                model.reset_state()
                break

            prediction, hidden = model(
                torch.FloatTensor(inp).unsqueeze(dim=0).to(device), hidden
            )

            if label is not None:
                targets = torch.FloatTensor(np.zeros(10)).to(device)
                targets[label] = 1
                # cross_entropy = -torch.sum(targets * torch.log(torch.squeeze(prediction) + eps))
                error = loss(prediction, targets.unsqueeze(dim=0))

                error.backward()
                opt.step()

                acc = 1 if torch.argmax(prediction) == label else 0

                running_error = (
                    running_error * 0.9995 + 0.0005 * error.detach()
                    if running_error
                    else error.detach()
                )
                running_acc = (
                    running_acc * 0.9995 + 0.0005 * acc if running_acc else acc
                )
                iterator.set_description(f"Epoch: {epochs}, Step: {step//28}, acc: {acc}, error: {error}, running_acc: {running_acc}, running_error: {running_error}")

                opt.zero_grad()
                hidden = model.reset_state().to(device)


if __name__ == "__main__":
    main()
