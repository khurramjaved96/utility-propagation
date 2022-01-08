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

import FlexibleNN
from FlexibleNN import Metric, Database

from src_py.utils.utils import get_types
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

    # make sure run_ids dont overlap when using parallel sweeps
    # sleep(random.random() * 10)

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument( "-r", "--run-id", help="run id (default: datetime)", default=datetime.now().strftime("%d%H%M%S%f")[:-5], type=int,)
    parser.add_argument("-s", "--seed", help="seed", default=0, type=int)
    parser.add_argument( "--db", help="database name", default="", type=str,)
    parser.add_argument( "-c", "--comment", help="comment for the experiment (can be used to filter within one db)", default="", type=str,)
    parser.add_argument( "--n-timesteps", help="number of timesteps", default=320000, type=int)
    parser.add_argument( "--num-recurrent-units", help="number of recurrent units in the first hidden layer", default=128, type=int)
    parser.add_argument( "--num-connections", help="number of connections per first hidden layer recurrent unit", default=28, type=int)

    parser.add_argument("--step-size", help="step size", default=1e-4, type=float)
    parser.add_argument( "--meta-step-size", help="tidbd step size (not used currently)", default=1e-3, type=float)

    args = parser.parse_args()
    training_metrics = None
    test_metrics = None

    if args.db == "":
        print("db name not provided. Not logging results")
    else:
        args.db = "hshah1_" + args.db
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
        commit_frequency=5000,
        training_metrics=training_metrics,
        test_metrics=test_metrics,
    )

    eps = sys.float_info.epsilon
    set_random_seed(args.seed)

    mnist = MNIST(seed=args.seed)
    model = FlexibleNN.RecurrentClassifierNetwork(
        args.step_size,
        args.seed,
        28,
        10,
        args.num_recurrent_units,
        args.num_connections,
    )

    step = 0
    running_error = None
    running_acc = None

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
                # if step %20000 == 0:
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
                    error = np.mean((targets - np.array(model.read_output_values())) ** 2)
                    # cross_entropy = -np.sum(
                    #    targets * np.log(np.array(model.read_output_values()) + eps)
                    # )
                    acc = 1 if np.argmax(model.read_output_values()) == label else 0

                    running_error = (
                        running_error * 0.9995 + 0.0005 * error
                        if running_error
                        else error
                    )
                    running_acc = (
                        running_acc * 0.9995 + 0.0005 * acc if running_acc else acc
                    )

                    iterator.set_description(
                        "Epoch: %s, Step: %s, Err: %s, Acc: %s, Run_Err: %s, Run_Acc: %s"
                        % (epoch, step // 28, error, acc, running_error, running_acc)
                    )
                    logger.log_performance_metrics(
                        "training",
                        epoch,
                        step // 28,
                        error,
                        running_error,
                        acc,
                        running_acc,
                    )
                    model.reset_state()

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
