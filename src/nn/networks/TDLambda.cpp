//
// Created by Khurram Javed on 2022-01-17.
//

#include <algorithm>
#include <iostream>
#include <cmath>
#include "../../../include/nn/networks/td_lambda.h"
#include "../../../include/nn/neuron.h"
//
TDLambda::TDLambda(float step_size,
                   int seed,
                   int no_of_input_features,
                   int total_targets,
                   int total_recurrent_features,
                   int layer_size) {
  this->layer_size = layer_size;
  this->step_size = step_size;
  this->mt.seed(seed);
  std::mt19937 second_mt(seed);
  std::uniform_real_distribution<float> weight_sampler(-0.1, 0.1);
  std::uniform_real_distribution<float> prob_sampler(0, 1);
  std::uniform_int_distribution<int> index_sampler(0, no_of_input_features + total_recurrent_features - 1);

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
//    for (int counter = 0; counter < this->input_neurons.size(); counter++) {
//      Neuron *neuron_ref = &this->input_neurons[counter];
//      lstm_neuron.add_synapse(neuron_ref,
//                              weight_sampler(mt),
//                              weight_sampler(mt),
//                              weight_sampler(mt),
//                              weight_sampler(mt));
//    }
    indexes_lstm_cells.push_back(i);
    this->LSTM_neurons.push_back(lstm_neuron);
  }

  for (int counter = 0; counter < total_recurrent_features; counter++) {
    int layer_no = counter / layer_size;
    //int max_connections = (layer_no * layer_size) + no_of_input_features; //dense
    int max_connections = no_of_input_features;
    int incoming_features = 0;
    std::vector<int> map_index(no_of_input_features + total_recurrent_features, 0);
    int counter_temp_temp = 0;
    int temp_counter = 0;
    while (temp_counter < 5000) {
      temp_counter++;
//    while (counter_temp_temp < 4000) {
      counter_temp_temp++;
      int index = index_sampler(second_mt);
      if (map_index[index] == 0) {
        map_index[index] = 1;
        if (index < no_of_input_features) {
//          std::cout << "Inp " << index << "\t" << counter << std::endl;
          incoming_features++;
          Neuron *neuron_ref = &this->input_neurons[index];
          LSTM_neurons[counter].add_synapse(neuron_ref,
                                            weight_sampler(mt),
                                            weight_sampler(mt),
                                            weight_sampler(mt),
                                            weight_sampler(mt));
        } else {
          index = index - no_of_input_features;
          int new_layer_no = index / layer_size;
          if (new_layer_no < layer_no) {
//            std::cout << index << "\t" << counter << std::endl;
            incoming_features++;
            Neuron *neuron_ref = &this->LSTM_neurons[index];
            //TODO making it single layer
            //Neuron *neuron_ref = &this->input_neurons[index];
            LSTM_neurons[counter].add_synapse(neuron_ref,
                                              weight_sampler(mt),
                                              weight_sampler(mt),
                                              weight_sampler(mt),
                                              weight_sampler(mt));
          }
        }
      }
    }
  }
  for (int counter = 0; counter < this->LSTM_neurons.size(); counter++) {
    for (int inner_counter = 0; inner_counter < this->LSTM_neurons[counter].incoming_neurons.size(); inner_counter++) {
      std::cout << this->LSTM_neurons[counter].incoming_neurons[inner_counter]->id << "\tto\t"
                << this->LSTM_neurons[counter].id << std::endl;
    }
  }
//  exit(1);

  predictions = 0;
  bias = 0;
  bias_gradients = 0;
  for (int j = 0; j < this->LSTM_neurons.size(); j++) {
    prediction_weights.push_back(0);
    prediction_weights_gradient.push_back(0);
    avg_feature_value.push_back(0);
    feature_mean.push_back(0);
    feature_std.push_back(1);
  }
}

float TDLambda::forward(std::vector<float> inputs) {
//  Set input neurons value
//  if(this->time_step%100000 == 99999)
//    this->step_size *= 0.8;
  for (int i = 0; i < inputs.size(); i++) {
    this->input_neurons[i].value = inputs[i];
  }

  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    LSTM_neurons[counter].update_value_sync();
  }

  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    LSTM_neurons[counter].compute_gradient_of_all_synapses();
  }

  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    LSTM_neurons[counter].fire();
    avg_feature_value[counter] =
        avg_feature_value[counter] * 0.99999 + 0.00001 * (LSTM_neurons[counter].value - feature_mean[counter]);
  }
//  std::cout << "Feature value = " << LSTM_neurons[0].value << "\t" <<  (LSTM_neurons[0].value - feature_mean[0])/sqrt(feature_std[0]) << std::endl;

  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    feature_mean[counter] = feature_mean[counter] * 0.99999 + 0.00001 * LSTM_neurons[counter].value;
//    std::cout << "Feature mean = " << feature_mean[counter] << std::endl;
    if (std::isnan(feature_mean[counter])) {
      std::cout << "feature value = " << LSTM_neurons[counter].value << std::endl;
      exit(1);
    }
    float temp = (feature_mean[counter] - LSTM_neurons[counter].value);
    feature_std[counter] = feature_std[counter] * 0.99999 + 0.00001 * temp * temp;
  }

  predictions = 0;
  for (int i = 0; i < prediction_weights.size(); i++) {
    predictions += prediction_weights[i] * (this->LSTM_neurons[i].value - feature_mean[i]) / sqrt(feature_std[i]);
  }
  predictions += bias;
  return predictions;
}

float TDLambda::get_target_without_sideeffects(std::vector<float> inputs) {
//  Backup old values
  std::vector<float> backup_vales;
  for (int i = 0; i < inputs.size(); i++) {
    backup_vales.push_back(this->input_neurons[i].value);
    this->input_neurons[i].value = inputs[i];
  }

//  Get hidden state without side-effects
  std::vector<float> hidden_state;
  for (int i = 0; i < LSTM_neurons.size(); i++) {
    hidden_state.push_back(LSTM_neurons[i].get_value_without_sideeffects());
  }

//  Compute prediction
  float temp_prediction = 0;
  for (int i = 0; i < prediction_weights.size(); i++) {
    temp_prediction += prediction_weights[i] * ((hidden_state[i] - feature_mean[i]) / sqrt(feature_std[i]));
  }
  temp_prediction += bias;


//  Restore values
  for (int i = 0; i < inputs.size(); i++) {
    this->input_neurons[i].value = backup_vales[i];
  }
  return temp_prediction;
}

void TDLambda::reset_state() {
  for (int i = 0; i < LSTM_neurons.size(); i++) {
    LSTM_neurons[i].reset_state();
  }
}

void TDLambda::zero_grad() {
  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    LSTM_neurons[counter].zero_grad();
  }

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    prediction_weights_gradient[index] = 0;
  }

  bias_gradients = 0;

}

void TDLambda::decay_gradient(float decay_rate) {
  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    LSTM_neurons[counter].decay_gradient(decay_rate);
  }

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    prediction_weights_gradient[index] *= decay_rate;
  }

  bias_gradients *= decay_rate;

}

std::vector<float> TDLambda::get_prediction_weights() {
  std::vector<float> my_vec;
  my_vec.reserve(prediction_weights.size());
  for (int index = 0; index < prediction_weights.size(); index++) {
    my_vec.push_back(prediction_weights[index]);
  }
  return my_vec;
}

std::vector<float> TDLambda::get_prediction_gradients() {
  std::vector<float> my_vec;
  my_vec.reserve(prediction_weights_gradient.size());
  for (int index = 0; index < prediction_weights_gradient.size(); index++) {
    my_vec.push_back(prediction_weights_gradient[index]);
  }
  return my_vec;
}

std::vector<float> TDLambda::get_state() {
  std::vector<float> my_vec;
  my_vec.reserve(LSTM_neurons.size());
  for (int index = 0; index < LSTM_neurons.size(); index++) {
    my_vec.push_back(LSTM_neurons[index].value);
  }
  return my_vec;
}

std::vector<float> TDLambda::get_normalized_state() {
  std::vector<float> my_vec;
  my_vec.reserve(LSTM_neurons.size());
  for (int i = 0; i < LSTM_neurons.size(); i++) {
    my_vec.push_back((this->LSTM_neurons[i].value - feature_mean[i]) / sqrt(feature_std[i]));
  }
  return my_vec;
}
void TDLambda::backward() {

//  Update the prediction weights
  for (int index = 0; index < LSTM_neurons.size(); index++) {
    float gradient = prediction_weights[index] / sqrt(feature_std[index]);
    LSTM_neurons[index].accumulate_gradient(gradient);
  }

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    prediction_weights_gradient[index] += (LSTM_neurons[index].value - feature_mean[index]) / sqrt(feature_std[index]);
  }

  bias_gradients += 1;

}

void TDLambda::update_parameters(int layer, float error) {

//  for(int index = 0; index < LSTM_neurons.size(); index++){
//    if(layer * layer_size > index){
//      if(LSTM_neurons[index].frozen == false) {
//        std::cout << "Freezing neuron with ID " << index << std::endl;
//        std::cout << "Mean = " << LSTM_neurons[index].running_mean << " Var " << std::sqrt(LSTM_neurons[index].running_variance) << std::endl;
//        LSTM_neurons[index].frozen = true;
//        prediction_weights[index] = prediction_weights[index]*LSTM_neurons[index].running_variance;
//      }
//    }
//  }
  for (int index = 0; index < LSTM_neurons.size(); index++) {
    if ((layer) * layer_size <= index && index < (layer + 1) * layer_size)
      //std::cout << "Training neuron = " << index << std::endl;
      LSTM_neurons[index].update_weights(step_size, error);
  }

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    if (index < (layer + 1) * layer_size) {
      float scaling = 1;
      if (fabs(prediction_weights_gradient[index]) > (step_size * 1000))
        scaling = 1/10000;
//      scaling = step_size * 10000 / prediction_weights_gradient[index];
//    if ((layer) * layer_size <= index && index < (layer + 1) * layer_size){

      scaling = 1;
      prediction_weights[index] += prediction_weights_gradient[index] * error * step_size * scaling;
    }
  }

//  bias += error * step_size * 0.001 * bias_gradients;

}
std::vector<float> TDLambda::real_all_running_mean() {
  std::vector<float> output_val;
  output_val.reserve(this->input_neurons.size() + this->LSTM_neurons.size());
//  Store input values
  for (auto n : this->input_neurons)
    output_val.push_back(n.running_mean);
//  Store other values
  for (auto n : this->LSTM_neurons)
    output_val.push_back(n.running_mean);
  return output_val;
}

std::vector<float> TDLambda::read_all_running_variance() {
  std::vector<float> output_val;
  output_val.reserve(this->input_neurons.size() + this->LSTM_neurons.size());
//  Store input values
  for (auto n : this->input_neurons)
    output_val.push_back(n.running_variance);
//  Store other values
  for (auto n : this->LSTM_neurons)
    output_val.push_back(n.running_variance);
  return output_val;
}

void TDLambda::update_parameters_no_freeze(float error) {

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    LSTM_neurons[index].update_weights(step_size, error);
  }

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    prediction_weights[index] += prediction_weights_gradient[index] * error * step_size;
  }

  bias += error * step_size * bias_gradients;

}

float TDLambda::read_output_values() {

  return predictions;
}

TDLambda::~TDLambda() {};
