import sys
import torch
import argparse
import random
import numpy as np
from collections import deque
from tqdm import tqdm
from time import sleep
from timeit import default_timer as timer
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import FlexibleNN
from FlexibleNN import Metric, Database

from src_py.utils.utils import get_types
from src_py.utils.utils import get_types
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

    # make sure run_ids dont overlap when using parallel sweeps
    # sleep(random.random() * 10)

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument( "-r", "--run-id", help="run id (default: datetime)", default=datetime.now().strftime("%d%H%M%S%f")[:-5], type=int,)
    parser.add_argument("-s", "--seed", help="seed", default=0, type=int)
    parser.add_argument( "--db", help="database name", default="", type=str,)
    parser.add_argument( "--db-prefix", help="database name prefix", default="hshah1_", type=str,)
    parser.add_argument( "-c", "--comment", help="comment for the experiment (can be used to filter within one db)", default="", type=str,)
    parser.add_argument( "--n-timesteps", help="number of timesteps", default=320000, type=int)
    parser.add_argument( "--hidden-size", help="size of the hidden recurrent layer", default=256, type=int)

    parser.add_argument("--step-size", help="step size", default=1e-4, type=float)

    args = parser.parse_args()
    training_metrics = None
    test_metrics = None

    if args.db == "":
        print("db name not provided. Not logging results")
    else:
        args.db = args.db_prefix + args.db
        Database().create_database(args.db)
        run_metric = Metric(args.db, "runs", list(vars(args).keys()), get_types(list(vars(args).values())), ["run_id"])
        run_metric.add_value([str(v) for v in list(vars(args).values())])

        run_state_metric = Metric(
            args.db,
            "run_states",
            ["run_id", "comment", "state", "timestep", "epoch", "training_error", "training_acc", "test_error", "test_acc", "run_time"],
            ["int", "VARCHAR(80)", "VARCHAR(40)", "int", "int", "real", "real", "real", "real", "VARCHAR(60)"],
            ["run_id"],
        )
        training_metrics = Metric(
            args.db,
            "training_metrics",
            ["run_id", "epoch", "timestep", "error", "running_error", "acc", "running_acc"],
            ["int", "int", "int", "real", "real", "real", "real"],
            ["run_id", "epoch", "timestep"],
        )
        test_metrics = Metric(
            args.db,
            "test_metrics",
            ["run_id", "epoch", "timestep", "error", "acc" ],
            ["int", "int", "int", "real", "real"],
            ["run_id", "epoch", "timestep"],
        )
    # fmt: on

    logger = LoggingManager(
        log_to_db=(args.db != ""),
        run_id=args.run_id,
        model=None,
        commit_frequency=20000,
        training_metrics=training_metrics,
        test_metrics=test_metrics,
    )

    eps = sys.float_info.epsilon
    set_random_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")  # its just faster ¯\_(ツ)_/¯

    mnist = MNIST(seed=args.seed)
    model = RNN(input_size=28, hidden_size=args.hidden_size, output_size=10).to(device)
    opt = optim.RMSprop(model.parameters(), lr=args.step_size)

    step = 0
    running_error = None
    running_acc = None
    hidden = model.reset_state().to(device)
    # loss = nn.CrossEntropyLoss()
    # cross_entropy = loss(prediction, torch.tensor([label]))
    loss = nn.MSELoss()

    start = timer()
    state_comment = "finished"

    try:
        for epoch in range((args.n_timesteps // 60000) + 1):
            iterator = tqdm(mnist.sequential_iterator(split="train"))
            for _, inp, label in iterator:
                step += 1
                if step // 28 >= args.n_timesteps:
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
                        running_error * 0.9995 + 0.0005 * error.detach().item()
                        if running_error
                        else error.detach().item()
                    )
                    running_acc = (
                        running_acc * 0.9995 + 0.0005 * acc if running_acc else acc
                    )
                    iterator.set_description(
                        "Epoch: %s, Step: %s, Err: %s, Acc: %s, Run_Err: %s, Run_Acc: %s"
                        % (epoch, step // 28, error.detach(), acc, running_error, running_acc)
                    )
                    logger.log_performance_metrics(
                        "training",
                        epoch,
                        step // 28,
                        error.detach().item(),
                        running_error,
                        acc,
                        running_acc,
                    )
                    opt.zero_grad()
                    hidden = model.reset_state().to(device)
    except:
        state_comment = "killed"
        print("failed... quitting")
    finally:
        if args.db != "":
            run_state_metric.add_value(
                [
                    str(v)
                    for v in [
                        args.run_id,
                        args.comment,
                        state_comment,
                        step // 28,
                        epoch,
                        running_error,
                        running_acc,
                        0,
                        0,
                        str(timedelta(seconds=timer() - start)),
                    ]
                ]
            )
        logger.commit_logs()

if __name__ == "__main__":
    main()
