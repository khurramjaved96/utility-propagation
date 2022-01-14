import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, device):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers=n_layers,
                            dropout=0)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        out, hidden = self.lstm(input.view(1,1,-1), hidden)
        out = self.linear(out)
        #out = F.softmax(out, dim=2)
        out = self.sigmoid(out)
        return out.view((1,-1)), hidden

    def reset_state(self):
        return (torch.zeros(1, 1, self.hidden_size).to(self.device),
               torch.zeros(1, 1, self.hidden_size).to(self.device))

