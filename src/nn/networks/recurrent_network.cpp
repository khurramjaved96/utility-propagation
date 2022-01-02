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

  float step_size_val = 1e-2;
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
    this->all_neurons.push_back(dynamic_cast<Neuron*>(recurent_neuron));
    Neuron *recurrent_neuron_base_class = recurent_neuron;
    Synapse
        *recurrent_synapse = new Synapse(recurrent_neuron_base_class, recurrent_neuron_base_class, recurrent_weight_sampler(this->mt), step_size_val);
    this->all_synapses.push_back(recurrent_synapse);
    std::vector<int> my_vec(no_of_input_features, 0);
    for(int j = 0; j < connections_per_feature; j++ ) {
      int feature_index = -1;
      while(feature_index == -1) {
        int temp_index = neuron_number_sampler(this->mt);
        if(my_vec[temp_index] == 0){
          my_vec[temp_index] = 1;
          feature_index = temp_index;
//          std::cout << "Selected incoming index "<< feature_index << std::endl;
        }
      }
      Synapse* new_synapse = new Synapse(input_neurons[feature_index], recurrent_neuron_base_class, weight_sampler(this->mt), step_size_val);
      this->all_synapses.push_back(new_synapse);
    }
  }
  this->Recurrent_neuron_layer.push_back(this->recurrent_features);

  for(auto output_neuron: this->output_neurons) {
    for (int i = 0; i < this->recurrent_features.size(); i++) {
      Synapse* output_synapse = new Synapse(this->recurrent_features[i], output_neuron, 0, step_size_val);
      this->output_synapses.push_back(output_synapse);
      this->all_synapses.push_back(output_synapse);
    }
  }
}

RecurrentNetwork::~RecurrentNetwork() {

}

void RecurrentNetwork::forward(std::vector<float> inp) {

//  std::cout << "Set inputs\n";

  this->set_input_values(inp);

//  std::cout << "Firing\n";

  std::for_each(
      std::execution::par_unseq,
      this->input_neurons.begin(),
      this->input_neurons.end(),
      [&](Neuron *n) {
        n->fire(this->time_step);
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
          n->update_value(this->time_step);
        });

//    std::cout << "Firing " << counter << "\n";
    std::for_each(
        std::execution::par_unseq,
        RecurrentNeuronList.begin(),
        RecurrentNeuronList.end(),
        [&](Neuron *n) {
          n->fire(this->time_step);
        });

  }


//  std::cout << "Updating values output \n";
  std::for_each(
      std::execution::par_unseq,
      this->output_neurons.begin(),
      this->output_neurons.end(),
      [&](Neuron *n) {
        n->update_value(this->time_step);
      });

//  std::cout << "Firing output \n";
  std::for_each(
      std::execution::par_unseq,
      this->output_neurons.begin(),
      this->output_neurons.end(),
      [&](Neuron *n) {
        n->fire(this->time_step);
      });

//  std::cout << "Updating neuron utility \n";
//  std::for_each(
//      std::execution::par_unseq,
//      this->all_neurons.begin(),
//      this->all_neurons.end(),
//      [&](Neuron *n) {
//        n->update_utility();
//      });

  this->time_step++;
}

void RecurrentNetwork::backward(std::vector<float> target, bool update_weight) {
  this->introduce_targets(target);

  std::for_each(
      std::execution::par_unseq,
      output_neurons.begin(),
      output_neurons.end(),
      [&](Neuron *n) {
        n->forward_gradients();
      });

  for (int layer = this->Recurrent_neuron_layer.size() - 1; layer >= 0; layer--) {
    std::for_each(
        std::execution::par_unseq,
        this->Recurrent_neuron_layer[layer].begin(),
        this->Recurrent_neuron_layer[layer].end(),
        [&](Neuron *n) {
          n->propagate_error();
        });

    std::for_each(
        std::execution::par_unseq,
        this->Recurrent_neuron_layer[layer].begin(),
        this->Recurrent_neuron_layer[layer].end(),
        [&](RecurrentRelu *n) {
          n->update_TH();
        });

    std::for_each(
        std::execution::par_unseq,
        this->Recurrent_neuron_layer[layer].begin(),
        this->Recurrent_neuron_layer[layer].end(),
        [&](RecurrentRelu *n) {
          n->forward_gradients();
        });
  }
//  Calculate our credit

//  std::for_each(
//      std::execution::par_unseq,
//      output_synapses.begin(),
//      output_synapses.end(),
//      [&](Synapse *s) {
//        s->update_utility();
//      });


  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](Synapse *s) {
        s->assign_credit();
      });

  if (update_weight) {
    std::for_each(
        std::execution::par_unseq,
        all_synapses.begin(),
        all_synapses.end(),
        [&](Synapse *s) {
          s->update_weight();
        });
  }


}



