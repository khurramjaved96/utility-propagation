import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.i2o = nn.Linear(input_size + hidden_size, output_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.sigmoid(output)
        return output, hidden

    def reset_state(self):
        return torch.zeros(1, self.hidden_size)


class RNNv2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNv2, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.i2o = nn.Linear(input_size + hidden_size, output_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.sigmoid(output)
        return output, hidden

    def reset_state(self):
        return torch.zeros(1, self.hidden_size)
