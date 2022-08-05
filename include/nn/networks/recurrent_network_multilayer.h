//
// Created by Khurram Javed on 2022-01-10.
//

#ifndef INCLUDE_NN_NETWORKS_RECURRENT_NETWORK_MULTILAYER_H_
#define INCLUDE_NN_NETWORKS_RECURRENT_NETWORK_MULTILAYER_H_

#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"
#include "./neural_network.h"

class RecurrentNetworkMultilayer : public NeuralNetwork {

 public:

  std::vector<Synapse *> active_synapses;

  std::vector<std::vector<RecurrentRelu *>> Recurrent_neuron_layer;

  RecurrentNetworkMultilayer(float step_size,
                             int seed,
                             int no_of_input_features,
                             int total_targets,
                             int total_recurrent_features_per_layer,
                             int total_layers,
                             int connections_per_feature);

  ~RecurrentNetworkMultilayer();

  void replace_feature(int feature_no);

  void print_graph(Neuron *root);

  void viz_graph();

  void set_print_bool();

  std::string get_viz_graph();

  void imprint();

  void forward(std::vector<float> inputs);

  void backward(std::vector<float> targets, int layer);

  void update_parameters();

  void add_feature(float step_size, float utility_to_keep);

  void reset_state();

  int least_useful_feature();

  void replace_least_important_feature();

};

#endif //INCLUDE_NN_NETWORKS_RECURRENT_NETWORK_MULTILAYER_H_
