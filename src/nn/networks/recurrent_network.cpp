//
// Created by Khurram Javed on 2021-09-28.
//

#include <execution>
#include <algorithm>
#include <iostream>
#include "../../../include/nn/networks/recurrent_network.h"

RecurrentNetwork::RecurrentNetwork(float step_size,
                                   int seed,
                                   int no_of_input_features,
                                   int total_targets,
                                   int total_recurrent_features,
                                   int connections_per_feature) {

  this->mt.seed(seed);
  std::uniform_int_distribution<int> neuron_number_sampler(0, no_of_input_features - 1);
  std::uniform_real_distribution<float> weight_sampler(-1, 1);
  std::uniform_real_distribution<float> recurrent_weight_sampler(0, 0.95);
  for (int i = 0; i < no_of_input_features; i++) {
    Neuron *n = new LinearNeuron(true, false);
    n->neuron_age = 0;
    n->drinking_age = 20000;
    n->set_layer_number(0);
    this->input_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  float step_size_val = 3e-2;
  for (int output_neurons = 0; output_neurons < total_targets; output_neurons++) {
//    Neuron *output_neuron = new SigmoidNeuron(false, true);
    Neuron *output_neuron = new LinearNeuron(false, true);
    output_neuron->set_layer_number(100);
    this->all_neurons.push_back(output_neuron);
    this->output_neurons.push_back(output_neuron);
  }

  for (int i = 0; i < total_recurrent_features; i++) {
//    std::cout << "Recurrent feature no "<< i << std::endl;
    RecurrentRelu *recurent_neuron = new RecurrentRelu(false, false);
    this->recurrent_features.push_back(recurent_neuron);
    this->all_neurons.push_back(dynamic_cast<Neuron *>(recurent_neuron));
    Neuron *recurrent_neuron_base_class = recurent_neuron;
    Synapse
        *recurrent_synapse = new Synapse(recurrent_neuron_base_class,
                                         recurrent_neuron_base_class,
                                         recurrent_weight_sampler(this->mt),
                                         step_size_val);
    this->all_synapses.push_back(recurrent_synapse);
    std::vector<int> my_vec(no_of_input_features, 0);
    for (int j = 0; j < connections_per_feature; j++) {
      int feature_index = -1;
      while (feature_index == -1) {
        int temp_index = neuron_number_sampler(this->mt);
        if (my_vec[temp_index] == 0) {
          my_vec[temp_index] = 1;
          feature_index = temp_index;
//          std::cout << "Selected incoming index "<< feature_index << std::endl;
        }
      }
      Synapse *new_synapse = new Synapse(input_neurons[feature_index],
                                         recurrent_neuron_base_class,
                                         weight_sampler(this->mt),
                                         step_size_val);
      this->all_synapses.push_back(new_synapse);
    }
  }
  this->Recurrent_neuron_layer.push_back(this->recurrent_features);

  for (auto output_neuron: this->output_neurons) {
    for (int i = 0; i < this->recurrent_features.size(); i++) {
      Synapse *output_synapse = new Synapse(this->recurrent_features[i], output_neuron, 0, step_size_val);
      this->output_synapses.push_back(output_synapse);
      this->all_synapses.push_back(output_synapse);
    }
  }
}

RecurrentNetwork::~RecurrentNetwork() {

}

void RecurrentNetwork::forward(std::vector<float> inp) {

  this->set_input_values(inp);

  std::for_each(
      std::execution::par_unseq,
      this->input_neurons.begin(),
      this->input_neurons.end(),
      [&](Neuron *n) {
        n->fire();
      });

  int counter = 0;
  for (auto RecurrentNeuronList: this->Recurrent_neuron_layer) {
    counter++;
//    std::cout << "Updating values " << counter << "\n";
    std::for_each(
        std::execution::par_unseq,
        RecurrentNeuronList.begin(),
        RecurrentNeuronList.end(),
        [&](Neuron *n) {
          n->update_value();
        });

//    std::cout << "Firing " << counter << "\n";
    std::for_each(
        std::execution::par_unseq,
        RecurrentNeuronList.begin(),
        RecurrentNeuronList.end(),
        [&](Neuron *n) {
          n->fire();
        });

  }


//  std::cout << "Updating values output \n";
  std::for_each(
      std::execution::par_unseq,
      this->output_neurons.begin(),
      this->output_neurons.end(),
      [&](Neuron *n) {
        n->update_value();
      });

//  std::cout << "Firing output \n";
  std::for_each(
      std::execution::par_unseq,
      this->output_neurons.begin(),
      this->output_neurons.end(),
      [&](Neuron *n) {
        n->fire();
      });

}

int RecurrentNetwork::least_useful_feature() {
//  std::cout << "Getting lease useful feature index\n";
  float least_useful_weight = 5000000;
  int least_useful_index = -1;
  int counter = 0;
  int total_eligible_for_replacement = 0;
  for (RecurrentRelu *neuron: this->Recurrent_neuron_layer[0]) {

    if(neuron->neuron_age > 20000){
      total_eligible_for_replacement++;
    }
    if (neuron->neuron_age > 20000 && std::abs(neuron->outgoing_synapses[0]->weight) < least_useful_weight) {
      least_useful_weight = std::abs(neuron->outgoing_synapses[0]->weight);
      least_useful_index = counter;
    }
    counter++;
  }
  if(total_eligible_for_replacement*2 > this->Recurrent_neuron_layer[0].size())
    return least_useful_index;
  return -1;
}

void RecurrentNetwork::replace_least_important_feature() {
  int index = least_useful_feature();
  std::cout << "Replacing feature = " << index << std::endl;
  if (index == -1)
    return;
  replace_feature(index);
}

void RecurrentNetwork::replace_feature(int feature_no) {
//  std::cout << "Replacing feature no " << feature_no << "\n";
  std::uniform_int_distribution<int> neuron_number_sampler(0, this->input_neurons.size() - 1);
  std::uniform_real_distribution<float> weight_sampler(-1, 1);
  std::uniform_real_distribution<float> recurrent_weight_sampler(0, 0.95);
//  std::cout << "Getting recurrent feature\n";
  RecurrentRelu *feature = this->Recurrent_neuron_layer[0][feature_no];
  std::vector<int> my_vec(this->input_neurons.size(), 0);
//  std::cout << "Iterating over inputs\n";
//  std::cout << "Incoming synapses size " << feature->incoming_synapses.size() << std::endl;
  for (auto synapse: feature->incoming_synapses) {
//    std::cout << "In the for loop\n";
    int feature_index = -1;
    while (feature_index == -1) {
//      std::cout << "In the while loop\n";
      int temp_index = neuron_number_sampler(this->mt);
      if (my_vec[temp_index] == 0) {
        my_vec[temp_index] = 1;
        feature_index = temp_index;
      }
    }
//    std::cout << "Index found " << feature_index << "\n";
    synapse->input_neuron = this->input_neurons[feature_index];
    synapse->age = 0;
    synapse->weight = weight_sampler(this->mt);
  }
  feature->recurrent_synapse->weight = recurrent_weight_sampler(this->mt);
  feature->outgoing_synapses[0]->weight = 0;
  feature->outgoing_synapses[0]->age = 0;
  feature->neuron_age = 0;
//  std::cout << "Feature replaced\n";
}

void RecurrentNetwork::backward(std::vector<float> target) {
  float prediction_error = this->output_neurons[0]->value - target[0];

  std::for_each(
      std::execution::par_unseq,
      this->Recurrent_neuron_layer[0].begin(),
      this->Recurrent_neuron_layer[0].end(),
      [&](RecurrentRelu *n) {
        n->compute_gradient_of_all_synapses();
      });

  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](Synapse *s) {
        s->assign_credit(prediction_error, 0, 0);
      });

}

void RecurrentNetwork::update_parameters() {

  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](Synapse *s) {
        s->update_weight();
      });
}



