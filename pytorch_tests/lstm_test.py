import torch
from torch import optim

hidden_size = 3
input_size = 3
network = torch.nn.LSTM(input_size, hidden_size, 1)
weights = torch.tensor(
    [ 0.36 ,-0.265 ,0.085 ,0.46 ,-0.165 ,0.475 ,0.015 ,-0.455 ,-0.395 ,-0.32 ,-0.065 ,-0.15 ,-0.06 ,0.2 ,-0.21 ,0.465 ,-0.305 ,-0.065 ,0.37 ,-0.06 ,-0.095 ,0.325 ,-0.375 ,-0.115 ,-0.14 ,-0.455 ,0.24 ,0.075 ,0.485 ,-0.105 ,0.375 ,0.46 ,-0.09 ,-0.005 ,0.385 ,-0.355])
weights = weights.view(4*hidden_size, input_size)
print(weights.data)
U = torch.tensor(
    [0.235 ,0.235 ,0.21 ,0.335 ,-0.34 ,0.465 ,-0.455 ,0.425 ,0.135 ,-0.34 ,-0.345 ,0.255 ,0.315 ,0.07 ,0.415 ,-0.36 ,-0.33 ,0.14 ,0.14 ,0.32 ,-0.235 ,0.165 ,-0.31 ,-0.415 ,-0.105 ,0.16 ,0.025 ,-0.29 ,0.43 ,-0.345 ,0.1 ,-0.495 ,-0.175 ,0.345 ,-0.215 ,-0.325])

U = U.view(4*hidden_size, hidden_size)
b = torch.tensor([  0.01, 0.095, -0.445, 0.37, -0.09, -0.045, 0.14, 0.21, -0.005, -0.235, 0.2, 0.105])
# b = b*0

h0 = torch.zeros(1, 1, hidden_size, requires_grad=True)
c0 = torch.zeros(1, 1, hidden_size, requires_grad=True)

prediction_weights = torch.zeros(hidden_size)
prediction_weights = prediction_weights + 0.1

list_of_inputs = []
for i in range(10):
    temp = torch.tensor([i, 20-i, i*-0.2 + 3-i*0.3]).view(1, 1, -1)
    list_of_inputs.append(temp)
    # print(temp)



for name, i in network.named_parameters():
    # print("Before")
    # print(name, i)
    # print(i.shape)
    if(name == "weight_ih_l0"):
        i.data = weights
    if(name == "weight_hh_l0"):
        i.data = U
    if(name == "bias_ih_l0"):
        i.data = b
    if(name == "bias_hh_l0"):
        i.data = i.data*0
    # print("After")
    # print(name, i)
    # print(i.shape)

for name, i in network.named_parameters():
    print(name, i.data.view(-1))


opt = optim.SGD(list(network.parameters()), 1e-3)
opt.zero_grad()
for input in list_of_inputs:
    print("Input = ", input)
    output, (h0, c0) = network(input, (h0, c0))
    output = output.squeeze()
pred = output*prediction_weights
pred = pred.sum()
pred.backward()
for name, i in network.named_parameters():
    print(name,"\n", i.grad)
    print(i.data)
print(input, h0.data, c0.data)
