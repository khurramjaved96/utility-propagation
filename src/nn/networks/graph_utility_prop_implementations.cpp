//
// Created by Khurram Javed on 2022-09-23.
//

#include "../../../include/nn/networks/graph.h"
#include <vector>
#include <iostream>

void GraphUtilPropogation::update_utility() {
  for (int i = 0; i < this->list_of_vertices.size(); i++) {
    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
      float old_value = this->list_of_vertices[i]->value;
      float new_value = old_value - e.weight * this->list_of_vertices[e.from]->forward();
      float new_post_activation_value = this->list_of_vertices[i]->forward_with_val(new_value);
//      e.local_utility =
//          e.local_utility * utility_decay_rate
//              + (1 - utility_decay_rate) * std::abs((this->list_of_vertices[i]->forward() - new_post_activation_value)/(e.weight+1e-8));
      e.local_utility = std::abs((this->list_of_vertices[i]->forward() - new_post_activation_value)/(this->list_of_vertices[e.from]->forward()+1e-8));
//      e.local_utility =  this->list_of_vertices[e.from]->forward();
    }
  }

  this->estimate_gradient();
  for (int i = 0; i < this->list_of_vertices.size(); i++) {
    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
      e.utility = e.utility * this->utility_decay_rate + (1 - this->utility_decay_rate) * std::abs(e.gradient * e.local_utility);
    }
  }
//
//  for (int i = list_of_vertices.size() - 1; i >= 0; i--) {
//    list_of_vertices[i]->utility = 0;
//    if (list_of_vertices[i]->is_output) {
//      list_of_vertices[i]->utility = 1;
//    }
//  }
////  list_of_vertices[list_of_vertices.size() - 1]->utility = std::abs(list_of_vertices[list_of_vertices.size() - 1]->forward());
//  list_of_vertices[list_of_vertices.size() - 1]->utility = 1;
//  for (int i = list_of_vertices.size() - 1; i >= 0; i--) {
//    float total_util = 1e-8;
//    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
//      total_util += e.local_utility;
//    }
//    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
//      e.utility = e.utility * utility_decay_rate + (1-utility_decay_rate) * (e.local_utility) * this->list_of_vertices[i]->utility*e.weight;
//      e.gradient = (e.local_utility) * this->list_of_vertices[i]->utility;
////      std::cout << " "
//    }
//    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
//      this->list_of_vertices[e.from]->utility += e.gradient;
//    }
//  }
}

void GraphUtilPropogationRelative::update_utility() {
  for (int i = 0; i < this->list_of_vertices.size(); i++) {
    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
      float old_value = this->list_of_vertices[i]->value;
      float new_value = old_value - e.weight * this->list_of_vertices[e.from]->forward();
      float new_post_activation_value = this->list_of_vertices[i]->forward_with_val(new_value);
      e.local_utility =
          e.local_utility * utility_decay_rate
              + (1 - utility_decay_rate) * std::abs(this->list_of_vertices[i]->forward() - new_post_activation_value);
    }
  }
  for (int i = list_of_vertices.size() - 1; i >= 0; i--) {
    list_of_vertices[i]->utility = 0;
    if (list_of_vertices[i]->is_output) {
      list_of_vertices[i]->utility = 1;
    }
  }
//  list_of_vertices[list_of_vertices.size() - 1]->utility = 1;
  list_of_vertices[list_of_vertices.size() - 1]->utility = std::abs(list_of_vertices[list_of_vertices.size() - 1]->forward());

  for (int i = list_of_vertices.size() - 1; i >= 0; i--) {
    float total_util = 1e-8;
    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
      total_util += e.local_utility;
    }
    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
      e.utility = (e.local_utility / total_util) * this->list_of_vertices[i]->utility;
//      std::cout << " "
    }
    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
      this->list_of_vertices[e.from]->utility += e.utility;
    }
  }
}

void GraphActivationTraceUtility::update_utility() {
  for (int i = 0; i < this->list_of_vertices.size(); i++) {
    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
      e.local_utility =
          e.local_utility * utility_decay_rate
              + (1 - utility_decay_rate) * std::abs(e.weight * this->list_of_vertices[e.from]->forward());
      e.utility = e.local_utility;
    }
  }
}

void GraphWeightUtility::update_utility() {
  for (int i = 0; i < this->list_of_vertices.size(); i++) {
    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
      e.utility = std::abs(e.weight);
    }
  }
}

void GraphRandomUtility::update_utility() {

}

void GraphLinearAssumptionUtility::update_utility() {

  this->estimate_gradient();
  for (int i = 0; i < this->list_of_vertices.size(); i++) {
    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
      e.utility =
          e.utility * this->utility_decay_rate + (1 - this->utility_decay_rate) * std::abs(e.gradient * e.weight);
    }
  }
}

void GraphGradientUtility::update_utility() {

  this->estimate_gradient();
  for (int i = 0; i < this->list_of_vertices.size(); i++) {
    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
      e.utility =
          e.utility * this->utility_decay_rate + (1 - this->utility_decay_rate) * std::abs(e.gradient);
    }
  }
}

void GraphLocalUtility::update_utility() {
  for (int i = 0; i < this->list_of_vertices.size(); i++) {
    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
      float old_value = this->list_of_vertices[i]->value;
      float new_value = old_value - e.weight * this->list_of_vertices[e.from]->forward();
      float new_post_activation_value = this->list_of_vertices[i]->forward_with_val(new_value);
      e.local_utility =
          e.local_utility * utility_decay_rate
              + (1 - utility_decay_rate) * std::abs(this->list_of_vertices[i]->forward() - new_post_activation_value);
      e.utility = e.local_utility;
    }
  }
}