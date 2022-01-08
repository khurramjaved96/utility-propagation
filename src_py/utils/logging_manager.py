import copy
import numpy as np


# There is problem with handling nan and inf values
# Ideally, we should use None but it cannot be sent through
# the interface. Maybe we should adjust add_values() to handle not passing
# the value to all the fields
INVALID_PLACEHOLDER = 1e+100


class LoggingManager:
    def __init__(self, log_to_db, run_id, model, commit_frequency=1000, training_metrics=None, test_metrics=None):
        self.log_to_db = log_to_db
        self.run_id = run_id
        self.model = model
        self.commit_frequency = commit_frequency

        self.training_metrics = training_metrics
        self.test_metrics = test_metrics

        self.training_log_vec = []
        self.test_log_vec = []

    def items_to_str(self, vec):
        ret_vec = []
        for v in vec:
            if type(v) == float and (np.isnan(v) or np.isinf(v)):
                ret_vec.append(str(INVALID_PLACEHOLDER))
            else:
                ret_vec.append(str(v))
        return ret_vec

    def commit_logs(self):
        if not self.log_to_db:
            return
        try:
            if self.training_metrics:
                self.training_metrics.add_values(self.training_log_vec)
            if self.test_metrics:
                self.test_metrics.add_values(self.test_log_vec)
        except:
            print("Failed commiting logs")

        self.training_log_vec = []
        self.test_log_vec = []

    def log_performance_metrics(self, split, epoch, timestep, error, running_error, acc, running_acc):
        if not self.log_to_db:
            return

        if split == 'training' and self.training_metrics is not None:
            if timestep % 10000 == 0:
                self.training_log_vec.append(self.items_to_str([self.run_id, epoch, timestep, error, running_error, acc, running_acc]))

        if split == 'test' and self.test_metrics is not None:
            self.test_log_vec.append(self.items_to_str([self.run_id, epoch, timestep, error, acc]))

        if split == 'test' or timestep % self.commit_frequency== 0:
            self.commit_logs()
