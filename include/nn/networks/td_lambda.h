//
// Created by Khurram Javed on 2022-01-24.
//

#ifndef INCLUDE_NN_NETWORKS_TD_LAMBDA_H_
#define INCLUDE_NN_NETWORKS_TD_LAMBDA_H_


#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"
#include "./neural_network.h"

class TDLambda {
 protected:
  int64_t time_step;
  std::mt19937 mt;

 public:
//


  float step_size;
  float predictions;
  float bias;
  float bias_gradients;
  float layer_size;

  //  These indexes are used to do parallel computation since std::parallel does not provide rank of a thread
  std::vector<int> indexes;
  std::vector<int> indexes_lstm_cells;

  std::vector<float> prediction_weights;

  std::vector<float> prediction_weights_gradient;

  float  get_target_without_sideeffects(std::vector<float> inputs);

  std::vector<LinearNeuron> input_neurons;

  std::vector<LSTM> LSTM_neurons;

  float read_output_values();

  TDLambda(float step_size, int seed, int no_of_input_features, int total_targets, int total_recurrent_features, int layer_size);

  ~TDLambda();

  float forward(std::vector<float> inputs);

  void zero_grad();

  void decay_gradient(float decay_rate);

  void backward();

  void update_parameters(int layer, float error);

  void reset_state();

};



#endif //INCLUDE_NN_NETWORKS_TD_LAMBDA_H_
