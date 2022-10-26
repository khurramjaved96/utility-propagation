//
// Created by Khurram Javed on 2022-09-23.
//

#include "../../../include/nn/networks/vertex.h"
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>

// Vertex implementation
Vertex::Vertex() {
  is_output = false;
  value = 0;
  sum_of_outgoing_weights = 0;
  id = id_generator;
  id_generator++;
  this->max_value = 10;
  this->min_value = -10;
  this->utility = 0;
};

float Vertex::forward() {
  return this->value;
}

float Vertex::forward_with_val(float value) {
  return value;
}
float Vertex::backward(float val) {
  return 1;
}

float Vertex::get_value() {
  return this->value;
}

int Vertex::id_generator = 0;

ReluVertex::ReluVertex() : Vertex() {
  this->max_value = 10;
  this->min_value = 0;
}

SigmoidVertex::SigmoidVertex() : Vertex() {
  this->max_value = 0.5;
  this->min_value = -0.5;
}

float ReluVertex::forward_with_val(float val) {
  if (val > 0)
    return val;
  return 0;
}
float ReluVertex::forward() {
  if (this->value > 0)
    return this->value;
  return 0;
}

float ReluVertex::backward(float val) {
  if (val > 0)
    return 1;
  return 0;
}

float SigmoidVertex::sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}

float SigmoidVertex::forward_with_val(float val) {
  return sigmoid(val) - 0.5;
}

float SigmoidVertex::forward() {
  return sigmoid(this->value) - 0.5;
}

float SigmoidVertex::backward(float val) {
  float temp = sigmoid(val);
  return temp * (1 - temp);
}

Vertex *VertexFactory::get_vertex(const std::string& type) {
  if (type == "linear") {
    return new Vertex();
  } else if (type == "relu") {
    return new ReluVertex();
  } else if (type == "sigmoid") {
    return new SigmoidVertex();
  }
  return nullptr;
}

BinaryVertex::BinaryVertex() {
  this->max_value = 1;
  this->min_value = -1;
}

float BinaryVertex::forward() {
  if(this->value > 0)
    return 1;
  else
    return 0;
}

float BinaryVertex::forward_with_val(float val) {
  if(val > 0)
    return 1;
  else
    return 0;
}

float BinaryVertex::backward(float val) {
  return 0;
}