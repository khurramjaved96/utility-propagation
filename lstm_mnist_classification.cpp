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

#include "include/environments/mnist/mnist_reader.hpp"
#include "include/environments/mnist/mnist_utils.hpp"


int main(int argc, char *argv[]) {


  LSTM* lstm_neuron = new LSTM();
  Neuron* lstm_downcast = dynamic_cast<Neuron*>(lstm_neuron);
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
  for(int i = 0; i< 5; i++){
    Synapse* syn = new Synapse(input_neurons[i], temp_neuron, 0, 0);
    lstm_neuron->add_synapse(syn, W_i[i], W_f[i], W_g[i], W_o[i]);
    input_neurons[i]->value = initial_values[i];
  }


  for(int i = 0; i < 10; i++){
    lstm_neuron->update_value();
    lstm_neuron->fire();
    lstm_neuron->compute_gradient_of_all_synapses(W_i);
    lstm_neuron->accumulate_gradient();

  }
  lstm_neuron->print_gradients();






}
