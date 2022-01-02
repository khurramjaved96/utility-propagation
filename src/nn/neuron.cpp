//
// Created by Khurram Javed on 2021-03-16.
//

#include "../../include/nn/neuron.h"
#include <assert.h>
#include <iostream>
#include <utility>
#include <random>
#include <algorithm>
#include <vector>
#include "../../include/utils.h"

Neuron::Neuron(bool is_input, bool is_output) {
  value = 0;
  value_before_firing = 0;
  id = neuron_id_generator;
  useless_neuron = false;
  neuron_id_generator++;
  this->is_output_neuron = is_output;
  is_input_neuron = is_input;
  neuron_age = 0;
  references = 0;
  neuron_utility = 0;
  drinking_age = 5000;
  mark_useless_prob = 0.99;
  is_bias_unit = false;
  is_recurrent_neuron = false;
}

void Neuron::set_layer_number(int layer) {
  this->layer_number = layer;
}

int Neuron::get_layer_number() {
  return this->layer_number;
}

void Neuron::update_utility() {

  this->neuron_utility = 0;
  for (auto it: this->outgoing_synapses) {
    this->neuron_utility += it->synapse_utility_to_distribute;
  }
  if (this->is_output_neuron)
    this->neuron_utility = 1;

  this->sum_of_utility_traces = 0;
  for (auto it: this->incoming_synapses) {
    if (!it->disable_utility)
      this->sum_of_utility_traces += it->synapse_local_utility_trace;
  }
}




void Neuron::fire(int time_step) {
  this->neuron_age++;
//  std::cout << "Base class firing\n";
  this->value = this->forward(value_before_firing);
}

void RecurrentRelu::fire(int time_step) {
  this->neuron_age++;
  this->old_value = this->value;
  this->value = this->forward(value_before_firing);
//  std::cout << "Old val " << this->old_value << " New val " << this->value << std::endl;
}

void Neuron::update_value(int time_step) {
//  std::cout << "Updating value of non-recurrent neuron : " << this->id << "\n";
  this->value_before_firing = 0;

//  Age our neuron like a fine wine and set the next values of our neuron.
  for (auto &it : this->incoming_synapses) {
    it->age++;
    this->value_before_firing += it->weight * it->input_neuron->value;
  }
}

void RecurrentRelu::disable_learning() {
  this->learning = false;
  for (auto synapse: this->incoming_synapses) {
    synapse->TH=0;
  }
}

void RecurrentRelu::enable_learning() {
  this->learning = true;
}

void RecurrentRelu::update_TH() {
  if(this->learning) {
    for (auto synapse: this->incoming_synapses) {
      if (synapse->input_neuron->id == synapse->output_neuron->id) {
//      Recurrent connection
        if (this->value > 0)
          synapse->TH = this->old_value + this->recurrent_synapse->weight * synapse->TH;
        else
          synapse->TH = 0;
      } else {
//      Non-recurrent connection
        if (this->value > 0) {
          synapse->TH = synapse->input_neuron->value + this->recurrent_synapse->weight * synapse->TH;
        } else {
          synapse->TH = 0;
        }
      }
    }
  }
}
void RecurrentRelu::update_value(int time_step) {
//  std::cout << "Updating value of recurrent neuron: " << this->id << "\n";
  this->value_before_firing = 0;

//  Age our neuron like a fine wine and set the next values of our neuron.
  for (auto &it : this->incoming_synapses) {
    it->age++;
    this->value_before_firing += it->weight * it->input_neuron->value;
  }
}


bool to_delete_ss(Synapse *s) {
  return s->is_useless;
}

/**
 * For each incoming synapse of a neuron, add the gradient from the error in this
 * neuron to its grad_queue for weight assignment. If we do pass gradients backwards,
 * also pass the gradient from the error to grad_queue for use in back propagation.
 */


void Neuron::forward_gradients() {
//  If this neuron has gradients to pass back
  for (auto &it : this->incoming_synapses) {
    float message_value;

    message_value = this->error_gradient.gradient;

//          We pack our gradient into a new message and pass it back to our incoming synapse.
    message grad_temp(message_value, this->error_gradient.time_step);
    grad_temp.lambda = this->error_gradient.lambda;
    grad_temp.gamma = this->error_gradient.gamma;
    grad_temp.error = this->error_gradient.error;
    grad_temp.target = this->error_gradient.target;

    if (it->propagate_gradients)
      it->grad_queue = grad_temp;
    it->grad_queue_weight_assignment = grad_temp;
  }  //  Remove this gradient from our list of things needed to pass back
}

/**
 * NOTE: If you are not VERY familiar with the backprop algorithm, I highly recommend
 * doing some reading before going through this function.
 */


void LTUSynced::set_threshold(float threshold) {
  this->activation_threshold = threshold;
}
int Neuron::get_no_of_syanpses_with_gradients() {
  int synapse_with_gradient = 0;
  for (auto it: this->outgoing_synapses) {
    if (it->propagate_gradients)
      synapse_with_gradient++;
  }
  return synapse_with_gradient;
}
void Neuron::propagate_error() {
  float accumulate_gradient = 0;
  int time_vector;
  int distance_vector;
  float error_vector;
  message messages_q;

//  No gradient propagation required for prediction nodes

// We need a loop invariant for this function to make sure progress is always made. Things to make sure:
// 1. A queue for a certain outgoing path won't grow large indefinitely
// 2. Adding new connections or removing old connections won't cause deadlocks
// 3. We can never get in a situation in which neither activation nor the gradient is popped. Some number should strictly increase or decrease

// No need to pass gradients if there are no out-going nodes with gradients
  if (this->get_no_of_syanpses_with_gradients() > 0 && !is_input_neuron) {

    for (auto &output_synapses_iterator : this->outgoing_synapses) {
      accumulate_gradient += output_synapses_iterator->weight *
          output_synapses_iterator->grad_queue.gradient;

      error_vector = output_synapses_iterator->grad_queue.error;
      messages_q = output_synapses_iterator->grad_queue;
      time_vector = output_synapses_iterator->grad_queue.time_step;
      output_synapses_iterator->grad_queue.remove = true;

    }

//    std::cout <<
    message n_message(accumulate_gradient, time_vector);
    n_message.error = error_vector;
    n_message.gamma = messages_q.gamma;
    n_message.lambda = messages_q.lambda;
    n_message.target = messages_q.target;


    this->error_gradient = n_message;
  }
}

/**
 * Mark synapses and neurons for deletion. Synapses will only get deleted if its age is > 70k.
 * Neurons will only be deleted if there are no outgoing synapses (and it's not an output neuron of course!)
 */
void Neuron::mark_useless_weights() {
//  if (this->is_output_neuron || this->is_input_neuron)
//    return;
  std::uniform_real_distribution<float> dist(0, 1);

  if (this->neuron_age > this->drinking_age) {
    for (auto &it : this->outgoing_synapses) {

      if (it->output_neuron->neuron_age > it->output_neuron->drinking_age
          && it->synapse_utility < it->utility_to_keep && !it->disable_utility) {
        if (dist(gen) > this->mark_useless_prob)
          it->is_useless = true;
      }
    }
  }

//  if this current neuron has no outgoing synapses and is not an output or input neuron,
//  delete it a
//  nd its incoming synapses.
  if (this->incoming_synapses.empty() && !this->is_input_neuron && !this->is_output_neuron) {
    this->useless_neuron = true;
    for (auto it : this->outgoing_synapses)
      it->is_useless = true;
  }


//  if (this->outgoing_synapses.empty() && !this->is_output_neuron && !this->is_input_neuron) {
//    this->useless_neuron = true;
//    for (auto it : this->incoming_synapses)
//      it->is_useless = true;
//  }

  if (this->outgoing_synapses.empty() && !this->is_output_neuron && !this->is_input_neuron) {
    this->useless_neuron = true;
    for (auto it : this->incoming_synapses)
      it->is_useless = true;
  }
}

/**
 * Delete outgoing and incoming synapses that were marked earlier as is_useless.
 */
void Neuron::prune_useless_weights() {
  std::for_each(
//            std::execution::seq,
      this->outgoing_synapses.begin(),
      this->outgoing_synapses.end(),
      [&](Synapse *s) {
        if (s->is_useless) {
          s->decrement_reference();
          if (s->input_neuron != nullptr) {
            s->input_neuron->decrement_reference();
            s->input_neuron = nullptr;
          }
          if (s->output_neuron != nullptr) {
            s->output_neuron->decrement_reference();
            s->output_neuron = nullptr;
          }
        }
      });

  auto it = std::remove_if(this->outgoing_synapses.begin(), this->outgoing_synapses.end(), to_delete_ss);
  this->outgoing_synapses.erase(it, this->outgoing_synapses.end());

  std::for_each(
//            std::execution::seq,
      this->incoming_synapses.begin(),
      this->incoming_synapses.end(),
      [&](Synapse *s) {
        if (s->is_useless) {
          s->decrement_reference();
          if (s->input_neuron != nullptr) {
            s->input_neuron->decrement_reference();
            s->input_neuron = nullptr;
          }
          if (s->output_neuron != nullptr) {
            s->output_neuron->decrement_reference();
            s->output_neuron = nullptr;
          }
        }
      });
  it = std::remove_if(this->incoming_synapses.begin(), this->incoming_synapses.end(), to_delete_ss);
  this->incoming_synapses.erase(it, this->incoming_synapses.end());
}

/**
 * Introduce a target to a neuron and calculate its error.
 * In this case, target should be our TD target, and the neuron should be an outgoing neuron.
 * @param target: target gradient_activation to calculate our error.
 * @param time_step: time step that we calculate this error. Use for backprop purposes.
 * @return: squared error
 */
float Neuron::introduce_targets(float target, int time_step) {

  float error = this->value - target;

  message m(this->backward(this->value), time_step);
  m.error = error;
  m.lambda = 0;
  m.gamma = 0;
  m.target = target;
  this->error_gradient = m;
  return error * error;
}

/**
 * Introduce a target to a neuron and calculate its error.
 * In this case, target should be our TD target, and the neuron should be an outgoing neuron.
 * @param target: target gradient_activation to calculate our error.
 * @param time_step: time step that we calculate this error. Use for backprop purposes.
 * @param gamma: discount factor
 * @param lambda: eligibility trace decay parameter
 * @return: squared error
 */
//float Neuron::introduce_targets(float target, int time_step, float gamma, float lambda) {
////  Introduce a target to a neuron and calculate its error.
////  In this case, target should be our TD target.
//
////      The activation is the output of our NN.
//  float error;
//
//  error = this->value - target;
//  float error_grad = error;
//
//
////      Create our error gradient for this neuron
//  message m(1, time_step);
//  m.lambda = lambda;
//  m.gamma = gamma;
//  m.error = error_grad;
//
//  this->error_gradient = m;
//  return error;
//}

float LinearNeuron::forward(float temp_value) {
  return temp_value;
}

float LinearNeuron::backward(float post_activation) {
  return 1;
}

float ReluNeuron::forward(float temp_value) {

  if (temp_value <= 0)
    return 0;

  return temp_value;
}
//
float ReluNeuron::backward(float post_activation) {
  if (post_activation > 0)
    return 1;
  else
    return 0;
}

float SigmoidNeuron::forward(float temp_value) {

  return sigmoid(temp_value);
}

float SigmoidNeuron::backward(float post_activation) {
  return post_activation * (1 - post_activation);
}

float BiasNeuron::forward(float temp_value) {
  return 1;
}

float BiasNeuron::backward(float output_grad) {
  return 0;
}

float LTUSynced::forward(float temp_value) {
  if (temp_value >= this->activation_threshold)
    return 1;
  return 0;
}

float LTUSynced::backward(float output_grad) {
  return 0;
}

float RecurrentRelu::forward(float temp_value) {
  if (temp_value <= 0)
    return 0;
  return temp_value;
}

float RecurrentRelu::backward(float post_activation) {
  if(post_activation > 0){
    return 1;
  }
  return 0;
}


ReluNeuron::ReluNeuron(bool is_input, bool is_output) : Neuron(is_input, is_output) {}

SigmoidNeuron::SigmoidNeuron(bool is_input, bool is_output) : Neuron(is_input, is_output) {}

RecurrentRelu::RecurrentRelu(bool is_input, bool is_output) : Neuron(is_input, is_output) {
  this->old_value = 0;
  this->is_recurrent_neuron = true;
}
LTUSynced::LTUSynced(bool is_input, bool is_output, float threshold) : Neuron(is_input, is_output) {
  this->activation_threshold = threshold;
}

BiasNeuron::BiasNeuron() : Neuron(false, false) {
  this->is_bias_unit = true;
}

LinearNeuron::LinearNeuron(bool is_input, bool is_output) : Neuron(is_input, is_output) {}

std::mt19937 Neuron::gen = std::mt19937(0);

int64_t Neuron::neuron_id_generator = 0;
