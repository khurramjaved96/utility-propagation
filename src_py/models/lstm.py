import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, device):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=0,
        )

        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        out, hidden = self.lstm(input.view(1, 1, -1), hidden)
        out = self.linear(out)
        # out = F.softmax(out, dim=2)
        out = self.sigmoid(out)
        return out.view((1, -1)), hidden

    def reset_state(self):
        return (
            torch.zeros(1, 1, self.hidden_size).to(self.device),
            torch.zeros(1, 1, self.hidden_size).to(self.device),
        )


class LSTM_multilayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, device):
        super(LSTM_multilayer, self).__init__()

        assert n_layers == 1, f"dont..."
        self.hidden_size = hidden_size
        self.device = device
        self.stage_2 = False

        self.lstm_1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=0,
        )

        self.lstm_2 = nn.LSTM(
            input_size=input_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=0,
        )

        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        """
        args:
            input: [input_size]
            hidden: [2,2,1,1,hidden_size] or
                    [(lstm1,lstm2),(hidden,cell),1,1,hidden_size]

        when stage_2 false: just train lstm_1 only. The half of linear
            layer's inputs are zeros
        when stage_2 true: lstm_1 is frozen. The lstm_2 is taking
            input + lstm_1's output as its inputs. The linear layer will
            take lstm_1 output and lstm_2 output as inputs.
        """
        if not self.stage_2:
            out, hidden[0] = self.lstm_1(input.view(1, 1, -1), hidden[0])
            out = self.linear(torch.cat([out, torch.zeros((1, 1, self.hidden_size))], dim=-1))
        else:
            out_1, hidden[0] = self.lstm_1(input.view(1, 1, -1), hidden[0])
            out_2, hidden[1] = self.lstm_2(torch.cat([input.view(1, 1, -1), out_1], dim=-1), hidden[1])
            out = torch.cat([out_1, out_2], dim=-1)
        out = self.sigmoid(out)
        return out.view((1, -1)), hidden

    def start_stage_2_training(self):
        """
        Freezes the first LSTM layer and
        starts using the 2nd LSTM layer
        """
        self.stage_2 = True
        self.lstm_1.requires_grad_(False)

    def reset_state(self):
        return [
            (
                torch.zeros(1, 1, self.hidden_size).to(self.device),
                torch.zeros(1, 1, self.hidden_size).to(self.device),
            ),
            (
                torch.zeros(1, 1, self.hidden_size).to(self.device),
                torch.zeros(1, 1, self.hidden_size).to(self.device),
            ),
        ]
