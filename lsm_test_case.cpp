//
// Created by Khurram Javed on 2022-01-08.
//

#include <iostream>
#include "include/nn/networks/recurrent_network.h"
#include "include/nn/neuron.h"
#include "include/nn/synapse.h"
#include "include/environments/animal_learning/tracecondioning.h"
#include "include/utils.h"
#include "include/nn/utils.h"
#include "include/experiment/Experiment.h"
#include "include/nn/utils.h"
#include "include/experiment/Metric.h"

#include <random>
#include <algorithm>
#include "include/environments/mnist/mnist_reader.hpp"
#include "include/environments/mnist/mnist_utils.hpp"


int main(int argc, char *argv[]) {

  static std::mt19937 gen;

  gen.seed(0);
  std::uniform_real_distribution<float> weight_sampler(-1, 1);
  float u_i =  0.3632;
  float u_f =  0.8304;
  float u_g = -0.2058;
  float u_o =  0.7483;

  float b_i = -0.1612;
  float b_f = 0.1058;
  float b_g = 0.9055;
  float b_o = -0.9277;

  LSTM* lstm_neuron = new LSTM(u_i, u_f, u_g, u_o, b_i, b_f, b_g, b_o);
  Neuron* temp_neuron = new LinearNeuron(false, false);
  std::vector<Neuron*> input_neurons;
  for(int i = 0; i < 5; i++)
    input_neurons.push_back(new LinearNeuron(true, false));
  //  Order: i|f|g|o

  std::vector<float> W_i{-0.0075,  0.5364, -0.8230, -0.7359, -0.3852};
  std::vector<float> W_f{0.2682, -0.0198,  0.7929, -0.0887,  0.2646};
  std::vector<float> W_g {-0.3022, -0.1966, -0.9553, -0.6623, -0.4122};
  std::vector<float> W_o {0.0370,  0.3953,  0.6000, -0.6779, -0.4355};
  std::vector<float> initial_values {1.4, 0.2, 1.6, 0.54, 1.02};
  std::vector<std::vector<float>> inputs;
  for(int step = 0; step < 1000; step++){
    std::vector<float> tmp;
    for(int temp = 0; temp < 5; temp ++)
      tmp.push_back(weight_sampler(gen));

    inputs.push_back(tmp);
  }

  for(int i = 0; i< 5; i++){
    lstm_neuron->add_synapse(input_neurons[i], W_i[i], W_f[i], W_g[i], W_o[i]);
  }
//
  for(int i = 0; i < 1000; i++){
    for(int j = 0; j< 5; j++){
      input_neurons[j]->value = inputs[i][j];
    }
    print_vector(inputs[i]);
    lstm_neuron->update_value_sync();
    lstm_neuron->compute_gradient_of_all_synapses();
//    lstm_neuron->fire();
    lstm_neuron->accumulate_gradient(1);

  }
  lstm_neuron->print_gradients();
}

// Ground truth gradient script
//import torch
//from torch import nn
//import random
//
//random.seed(0)
//torch.random.manual_seed(0)
//
//hidden_units = 1
//rnn = nn.LSTM(5, hidden_units, 1)
//
//
//for name, param in rnn.named_parameters():
//    print(name)
//if name == "bias_hh_l0":
//param.data = param.data*0
//print(param)
//
//
//inp = torch.zeros(1, 1,5)
//inp[0,0,0] = 1.4
//inp[0,0,1] = 0.2
//inp[0,0,2] = 1.6
//inp[0,0,3] = 0.54
//inp[0,0,4] = 1.02
//
//
//h0 = torch.zeros(1, 1, hidden_units)
//c0 = torch.zeros(1, 1, hidden_units)
//
//sum_of_h = None
//for steps in range(0, 10):
//print("H = ", h0)
//print("C = ", c0)
//output, (h0, c0) = rnn(inp, (h0, c0))
//if sum_of_h is None:
//sum_of_h = output
//else:
//sum_of_h += output
//# inp = torch.zeros(1, 1, 5)
//# print(h0)
//
//    sum_of_h.backward()
//for name, param in rnn.named_parameters():
//    print("Name", name)
//# print("Weights", param.data)
//print("Gradient", param.grad)
//
//# rnn(inp)
//# rnn.parameters()
