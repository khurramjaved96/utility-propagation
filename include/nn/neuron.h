//
// Created by Khurram Javed on 2021-09-20.
//

#ifndef INCLUDE_NN_SYNCED_NEURON_H_
#define INCLUDE_NN_SYNCED_NEURON_H_


#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include "./dynamic_elem.h"
#include "./synapse.h"
#include "./message.h"
#include "./utils.h"

class Neuron : public dynamic_elem {
 public:
  bool is_recurrent_neuron;
  static int64_t neuron_id_generator;
  static std::mt19937 gen;
  bool is_input_neuron;
  bool is_bias_unit;
  int layer_number;
  float value;
  int drinking_age;
  float value_before_firing;
  float neuron_utility;
  float neuron_utility_to_distribute;
  float sum_of_utility_traces;
  bool is_output_neuron;
  bool useless_neuron;
  int64_t id;
  int neuron_age;
  float mark_useless_prob;

  void set_layer_number(int layer);

  int get_layer_number();

  void forward_gradients();

  virtual void update_value(int time_step);

  message error_gradient;
  std::vector<Synapse *> outgoing_synapses;
  std::vector<Synapse *> incoming_synapses;



  int get_no_of_syanpses_with_gradients();

  Neuron(bool is_input, bool is_output);

  virtual void fire(int time_step);

  float introduce_targets(float target, int timestep);

//  float introduce_targets(float target, int timestep, float gamma, float lambda);

  void propagate_error();

  virtual float backward(float output_grad) = 0;

  virtual float forward(float temp_value) = 0;

  void update_utility();

  void normalize_neuron();

  void mark_useless_weights();

  void prune_useless_weights();

  ~Neuron() = default;
};

class RecurrentRelu : public Neuron {

 public:
  float old_value;

  bool learning = true;

  void disable_learning();

  void enable_learning();

  void compute_gradient_of_all_synapses();

  void update_value(int time_step);

  void update_TH();

  float backward(float output_grad);

  float forward(float temp_value);

  Synapse* recurrent_synapse;

  RecurrentRelu(bool is_input, bool is_output);

  void fire(int time_step);


};

class LTUSynced : public Neuron {
 public:
  float backward(float output_grad);
  float forward(float temp_value);
  LTUSynced(bool is_input, bool is_output, float threshold);
  void set_threshold(float threshold);
  float activation_threshold;

};

class SigmoidNeuron : public Neuron {
 public:
  float backward(float output_grad);

  float forward(float temp_value);

  SigmoidNeuron(bool is_input, bool is_output);
};


class ReluNeuron : public Neuron {
 public:
  float backward(float output_grad);

  float forward(float temp_value);

  ReluNeuron(bool is_input, bool is_output);

};


class BiasNeuron : public Neuron {
 public:
  BiasNeuron();
  float backward(float output_grad);

  float forward(float temp_value);
};


class LinearNeuron : public Neuron {
 public:
  float backward(float output_grad);

  float forward(float temp_value);

  LinearNeuron(bool is_input, bool is_output);
};





#endif //INCLUDE_NN_SYNCED_NEURON_H_
