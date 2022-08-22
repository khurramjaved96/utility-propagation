import torch
import lstm_model

hidden_size = 2
input_size = 3
network = lstm_model.CustomLSTM(input_size, hidden_size)
weights = torch.tensor(
    [0.36 ,-0.265 ,0.085 ,0.46 ,-0.165 ,0.475 ,0.015 ,-0.455 ,-0.395 ,-0.32 ,-0.065 ,-0.15 ,-0.06 ,0.2 ,-0.21 ,0.465 ,-0.305 ,-0.065 ,0.37 ,-0.06 ,-0.095 ,0.325 ,-0.375 ,-0.115])
weights = weights.view(4*hidden_size,  input_size).T
U = torch.tensor(
    [-0.14 ,-0.455 ,0.24 ,0.075 ,0.485 ,-0.105 ,0.375 ,0.46 ,-0.09 ,-0.005 ,0.385 ,-0.355 ,0.235 ,0.235 ,0.21 ,0.335]).T

U = U.view(4*hidden_size, hidden_size ).T
b = torch.tensor([-0.14, -0.455, 0.24, 0.075, 0.485, -0.105, 0.375, 0.46, -0.09, -0.005, 0.385, -0.355, 0.235, 0.235, 0.21, 0.335])
b = b*0

print(network.W.shape, weights.shape)
print(network.U.shape, U.shape)
print(network.bias.shape, b.shape)
network.W.data = weights;
network.U.data = U
network.bias.data = network.bias*0

h0 = torch.zeros(1, hidden_size)
c0 = torch.zeros(1, hidden_size)

list_of_inputs = []
for i in range(10):
    temp = torch.tensor([i, 20-i, i*-0.2 + 3-i*0.3]).view(1, 1, -1)
    list_of_inputs.append(temp)
    # print(temp)



for name, i in network.named_parameters():
    print(name, i)
    # print("Before")
    # print(name, i)
    # print(i.shape)
    # if(name == "weight_ih_l0"):
    #     i.data = weights
    # if(name == "weight_hh_l0"):
    #     i.data = U
    # if(name == "bias_ih_l0"):
    #     i.data = b
    # if(name == "bias_hh_l0"):
    #     i.data = b*0
    # print("After")
    # print(name, i)
    # print(i.shape)


# for name, i in network.named_parameters():
#     print(name, i.data.view(-1))

for input in list_of_inputs:
    output, (h0, c0) = network(input, (h0, c0))
    print(input, h0.data, c0.data)
