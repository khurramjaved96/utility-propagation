//
// Created by Khurram Javed on 2021-09-28.
//

#ifndef INCLUDE_NN_NETWORKS_LAYERWISE_FEEDWORWARD_H_
#define INCLUDE_NN_NETWORKS_LAYERWISE_FEEDWORWARD_H_




#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"
#include "./neural_network.h"

class RecurrentNetwork : public NeuralNetwork {

 public:

  std::vector<Synapse *> active_synapses;

  std::vector<std::vector<RecurrentRelu *>> Recurrent_neuron_layer;

  RecurrentNetwork(float step_size, int seed, int no_of_input_features, int total_targets, int total_recurrent_features, int connections_per_feature);

  ~RecurrentNetwork();

  void replace_feature(int feature_no);

  void print_graph(Neuron *root);

  void viz_graph();

  void set_print_bool();

  std::string get_viz_graph();

  void imprint();

  void forward(std::vector<float> inputs);

  void backward(std::vector<float> targets);

  void update_parameters();

  void add_feature(float step_size, float utility_to_keep);

  int least_useful_feature();

  void replace_least_important_feature();

};


#endif //INCLUDE_NN_NETWORKS_LAYERWISE_FEEDWORWARD_H_
