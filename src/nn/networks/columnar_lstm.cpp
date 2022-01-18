//
// Created by Khurram Javed on 2022-01-17.
//


#include <execution>
#include <algorithm>
#include <iostream>
#include "../../../include/nn/networks/columnar_lstm.h"
#include "../../../include/nn/neuron.h"

ColumnarLSTM::ColumnarLSTM(float step_size,
                           int seed,
                           int no_of_input_features,
                           int total_targets,
                           int total_recurrent_features) {
  this->step_size = step_size;
  this->mt.seed(seed);
  std::uniform_real_distribution<float> weight_sampler(-1, 1);

  for (int i = 0; i < no_of_input_features; i++) {
    LinearNeuron n(true, false);
    this->input_neurons.push_back(n);
  }


  for (int i = 0; i < total_recurrent_features; i++) {
//    std::cout << "Recurrent feature no "<< i << std::endl;
    LSTM lstm_neuron(weight_sampler(mt),
                     weight_sampler(mt),
                     weight_sampler(mt),
                     weight_sampler(mt),
                     weight_sampler(mt),
                     weight_sampler(mt),
                     weight_sampler(mt),
                     weight_sampler(mt));
    for (int counter = 0; counter < this->input_neurons.size(); counter++) {
      Neuron *neuron_ref = &this->input_neurons[counter];
      lstm_neuron.add_synapse(neuron_ref,
                              weight_sampler(mt),
                              weight_sampler(mt),
                              weight_sampler(mt),
                              weight_sampler(mt));
    }
    indexes_lstm_cells.push_back(i);
    this->LSTM_neurons.push_back(lstm_neuron);
  }

  for(int i = 0; i < total_targets; i++){
    predictions.push_back(0);
    errors.push_back(0);
    indexes.push_back(i);
    std::vector<float> prediction_weight_cur_target;
    for(int j = 0; j < this->LSTM_neurons.size(); j++){
      prediction_weight_cur_target.push_back(0);
    }
    prediction_weights.push_back(prediction_weight_cur_target);
  }
}

void ColumnarLSTM::reset_state() {
  for(int i = 0; i < LSTM_neurons.size(); i++){
    LSTM_neurons[i].reset_state();
  }
}

void ColumnarLSTM::forward(std::vector<float> inputs) {
//  Set input neurons value
  for(int i = 0; i < inputs.size(); i++){
    this->input_neurons[i].value = inputs[i];
  }
  std::for_each(
      std::execution::par_unseq,
      this->LSTM_neurons.begin(),
      this->LSTM_neurons.end(),
      [&](LSTM &n) {
        n.update_value();
        n.compute_gradient_of_all_synapses();
      });

  std::for_each(
      std::execution::par_unseq,
      this->indexes.begin(),
      this->indexes.end(),
      [&](int index) {
        predictions[index] = 0;
        for(int i = 0; i < prediction_weights[index].size(); i++){
          predictions[index] += prediction_weights[index][i]*this->LSTM_neurons[i].value;
        }
      });
//  std::cout << "Predictions = ";
//  for(int i = 0; i < predictions.size(); i++)
//    std::cout << predictions[i] << ",";
//  std::cout << std::endl;
}

void ColumnarLSTM::backward(std::vector<float> targets) {

  for(int i = 0; i < targets.size(); i++){
    errors[i] = targets[i] - predictions[i];
  }
//  Update the prediction weights
  std::for_each(
      std::execution::par_unseq,
      this->indexes_lstm_cells.begin(),
      this->indexes_lstm_cells.end(),
      [&](int index) {
        float gradient = 0;
        for(int i = 0; i < predictions.size(); i++){
          gradient += errors[i]*prediction_weights[i][index];
        }
        LSTM_neurons[index].zero_grad();
        LSTM_neurons[index].accumulate_gradient(gradient);
      });

//  First we compute error, then we call accumulate gradient on the LSTM units


}

void ColumnarLSTM::update_parameters() {
  std::for_each(
      std::execution::par_unseq,
      this->indexes_lstm_cells.begin(),
      this->indexes_lstm_cells.end(),
      [&](int index) {
        LSTM_neurons[index].update_weights(step_size);
      });

  std::for_each(
      std::execution::par_unseq,
      this->indexes_lstm_cells.begin(),
      this->indexes_lstm_cells.end(),
      [&](int index) {
        for(int i = 0; i < predictions.size(); i++){
          prediction_weights[i][index] += LSTM_neurons[index].value*errors[i]*step_size;
        }
      });

}

std::vector<float> ColumnarLSTM::read_output_values() {

  return predictions;
}

ColumnarLSTM::~ColumnarLSTM() {};