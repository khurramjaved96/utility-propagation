import sys
import torch
import argparse
import random
import copy
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
from src_py.models.lstm import LSTM, LSTM_multilayer


def set_random_seed(seed: int) -> None:
    """
    Seed the different random generators.
    :param seed:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, test_iterator, loss, logger, epoch, step, device):
    print("")
    model.eval()
    hidden = model.reset_state()
    test_errs = []
    test_accs = []
    test_step = 0
    for _, inp, label in test_iterator:
        test_step += 1

        with torch.no_grad():
            prediction, hidden = model(
                torch.FloatTensor(inp).unsqueeze(dim=0).to(device), hidden
            )

        if label is not None:
            targets = torch.FloatTensor(np.zeros(10)).to(device)
            targets[label] = 1
            test_errs.append(loss(prediction, targets.unsqueeze(dim=0)).detach())
            test_accs.append(1 if torch.argmax(prediction) == label else 0)
            test_iterator.set_description(
                "Testing Epoch: %s, Step: %s, Err: %s, Acc: %s"
                % (epoch, test_step // 28, np.mean(test_errs), np.mean(test_accs))
            )
            hidden = model.reset_state()
    logger.log_performance_metrics(
        "test",
        epoch,
        step // 28,
        np.mean(test_errs),
        0,
        np.mean(test_accs),
        0,
    )
    print("")
    return np.mean(test_errs), np.mean(test_accs)


def main():

    # make sure run_ids dont overlap when using parallel sweeps
    sleep(random.random() * 10)

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument( "-r", "--run-id", help="run id (default: datetime)", default=datetime.now().strftime("%d%H%M%S%f")[:-5], type=int,)
    parser.add_argument("-s", "--seed", help="seed", default=0, type=int)
    parser.add_argument( "--db", help="database name", default="", type=str,)
    parser.add_argument( "--db-prefix", help="database name prefix (change with username for ComputeCanada)", default="hshah1_", type=str,)
    parser.add_argument( "-c", "--comment", help="comment for the experiment (can be used to filter within one db)", default="", type=str,)
    parser.add_argument( "--n-timesteps", help="number of timesteps", default=640000, type=int)
    parser.add_argument( "--hidden-size", help="size of the hidden recurrent layer", default=128, type=int)
    parser.add_argument( "--n_layers", help="number of layers", default=1, type=int)
    parser.add_argument( "--model", help="model to use: RNN or LSTM, LSTM_multilayer", default="LSTM", type=str,)
    parser.add_argument('--sparse', help='sparse hidden weights (0: dense - default, 1: sparse)', default=0, type=int)
    parser.add_argument('--echo-state', help='echo state (for LSTM only), (0: disable - default, 1: enable)', default=0, type=int)
    parser.add_argument('--truncation-length', help='trunctation length', default=100000, type=int)

    parser.add_argument("--step-size", help="step size", default=1e-1, type=float)

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
        weight_metrics = Metric(
            args.db,
            "weight_metrics",
            ["run_id", "epoch", "timestep", "linear_weights" ],
            ["int", "int", "int", "real"],
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
    if args.model == "RNN":
        model = RNN(
            input_size=28, hidden_size=args.hidden_size, output_size=10, device=device
        ).to(device)
    elif args.model == "LSTM":
        model = LSTM(
            input_size=28,
            hidden_size=args.hidden_size,
            output_size=10,
            n_layers=args.n_layers,
            device=device,
            seed=args.seed
        ).to(device)
    elif args.model == "LSTM_multilayer":
        model = LSTM_multilayer(
            input_size=28,
            hidden_size=args.hidden_size,
            output_size=10,
            n_layers=args.n_layers,
            device=device,
            seed=args.seed
        ).to(device)

    if args.sparse:
        mask = torch.zeros(args.hidden_size*4, args.hidden_size)
        for a in range(0, 4*args.hidden_size, args.hidden_size):
            mask[a: a + args.hidden_size, :] = torch.eye(args.hidden_size)
        for name, param in model.named_parameters():
            if("weight_hh" in name):
                param.data = param.data*mask

    if args.echo_state:
        assert args.model == "LSTM", f"impl for LSTM only"
        model.lstm.requires_grad_(False)

    opt = optim.SGD(model.parameters(), lr=args.step_size)

    step = 0
    running_error = None
    running_acc = None
    test_error = 0
    test_acc = 0
    loss = nn.MSELoss()

    start = timer()
    state_comment = "finished"

    try:
        for epoch in range((args.n_timesteps // 60000) + 1):
            model.train()
            hidden = model.reset_state()
            iterator = tqdm(mnist.sequential_iterator(split="train"))
            for _, inp, label in iterator:
                step += 1
                if step // 28 >= args.n_timesteps:
                    hidden = model.reset_state()
                    break
                if step % (280000 * 6 * 3) == 0 and args.model == "LSTM_multilayer":
                    print("Freezing LSTM 1st layer weights...")
                    model.start_stage_2_training()

                # detach except for last truncation_length steps per img
                if (not 28 - ((step-1) %28) <= args.truncation_length):
                    hidden = tuple(h.detach() for h in hidden)
                prediction, hidden = model(
                    torch.FloatTensor(inp).unsqueeze(dim=0).to(device), hidden
                )

                if label is not None:
                    targets = torch.FloatTensor(np.zeros(10)).to(device)
                    targets[label] = 1
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
                        % (
                            epoch,
                            step // 28,
                            error.detach(),
                            acc,
                            running_error,
                            running_acc,
                        )
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
                    hidden = model.reset_state()
                    if args.sparse:
                        for name, param in model.named_parameters():
                            if("weight_hh" in name):
                                param.data = param.data*mask

                if step % 280000 == 0:
                    test_iterator = tqdm(mnist.sequential_iterator(split="test"))
                    test_error, test_acc = evaluate(
                        copy.deepcopy(model),
                        test_iterator,
                        loss,
                        logger,
                        epoch,
                        step,
                        device,
                    )
                if step % (280000 * 6) == 1:
                    print("magnitude of linear weights: ")
                    print(torch.sum(torch.abs(model.linear.weight), dim=0).detach())
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
                        test_error,
                        test_acc,
                        str(timedelta(seconds=timer() - start)),
                    ]
                ]
            )
        logger.commit_logs()
        #from IPython import embed; embed()


if __name__ == "__main__":
    main()
