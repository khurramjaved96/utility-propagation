//
// Created by Khurram Javed on 2022-02-23.
//

#ifndef INCLUDE_NN_NETWORKS_DENSE_LSTM_H_
#define INCLUDE_NN_NETWORKS_DENSE_LSTM_H_

#include <vector>
#include <queue>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"
#include "./neural_network.h"

class DenseLSTM {
 protected:
  int64_t time_step;
  std::mt19937 mt;

 public:
//
  float step_size;
  std::vector<float> get_state();
  int truncation;
  int input_size;
  int hidden_state_size;
  std::vector<float> prediction_weights;
  std::vector<float> prediction_weights_grad;

  std::vector<std::vector<float>> x_queue;
  std::vector<std::vector<float>> h_queue;
  std::vector<std::vector<float>> c_queue;
  std::vector<std::vector<float>> i_queue;
  std::vector<std::vector<float>> g_queue;
  std::vector<std::vector<float>> f_queue;
  std::vector<std::vector<float>> o_queue;

  std::vector<float> W;
  std::vector<float> W_grad;
  std::vector<float> U;
  std::vector<float> U_grad;
  std::vector<float> b;
  std::vector<float> b_grad;

  float  get_target_without_sideeffects(std::vector<float> inputs);

  std::vector<LinearNeuron> input_neurons;

  std::vector<LSTM> LSTM_neurons;

  std::vector<float> read_output_values();

  DenseLSTM(float step_size,
            int seed,
            int hidden_size,
            int no_of_input_features,
            int truncation,
            float init_range);

  void zero_grad();

  void decay_gradient(float decay_rate);

  float forward(std::vector<float> inputs);

  void backward();

  std::vector<std::vector<float>> backward_with_future_grad(std::vector<std::vector<float>> grad_f, int time);

  void update_parameters(float error);

  void reset_state();

};

#endif //INCLUDE_NN_NETWORKS_DENSE_LSTM_H_
