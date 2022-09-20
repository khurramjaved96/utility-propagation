//
// Created by Khurram Javed on 2022-08-30.
//

#include "../../../include/nn/networks/graph.h"
#include <random>
#include <iostream>
#include <string>
#include <vector>

//Edge implementation
Edge::Edge(float weight, int from, int to) {
  this->edge_id = edge_id_generator++;
  this->from = from;
  this->to = to;
  this->gradient = 0;
  this->weight = weight;
  this->utility_propagation = 1;
  this->activation_trace_utility = 1;
  this->real_utility = 1;
}

int Edge::edge_id_generator = 0;

// Vertex implementation
Vertex::Vertex() {
  is_output = false;
  value = 0;
  sum_of_outgoing_weights = 0;
  id = id_generator;
  id_generator++;
};

int Vertex::id_generator = 0;

//Graph implementation

void Graph::add_edge(float weight, int from, int to) {
  Edge my_edge = Edge(weight, from, to);
  list_of_vertices[to].incoming_edges.push_back(my_edge);
  list_of_vertices[from].sum_of_outgoing_weights += std::abs(weight);
//  list_of_vertices[from].outgoing_edges.push_back(my_edge);
}

void Graph::normalize_weights() {
  for (int i = list_of_vertices.size() - 1; i >= 0; i--) {
    for (auto &e: this->list_of_vertices[i].incoming_edges) {
      e.weight = e.weight/this->list_of_vertices[e.from].sum_of_outgoing_weights;
    }
  }
}

void Graph::remove_weight_real_util() {
  std::pair<int, int> address_min;
  float value_min = 1000000;
  for (int i = list_of_vertices.size() - 1; i >= 0; i--) {
    int counter = 0;
    for (auto &e: this->list_of_vertices[i].incoming_edges) {
      if(std::abs(e.weight) > 0 && e.real_utility < value_min){
        value_min = e.real_utility;
        address_min.first = i;
        address_min.second = counter;
      }
      counter++;
    }
  }
  list_of_vertices[address_min.first].incoming_edges[address_min.second].weight = 0;
}

void Graph::remove_weight_activate_trace()  {
  std::pair<int, int> address_min;
  float value_min = 1000000;
  for (int i = list_of_vertices.size() - 1; i >= 0; i--) {
    int counter = 0;
    for (auto &e: this->list_of_vertices[i].incoming_edges) {
      if(std::abs(e.weight) > 0 && e.activation_trace_utility < value_min){
        value_min = e.activation_trace_utility;
        address_min.first = i;
        address_min.second = counter;
      }
      counter++;
    }
  }
  list_of_vertices[address_min.first].incoming_edges[address_min.second].weight = 0;
}

void Graph::remove_weight_util_prop()  {
  std::pair<int, int> address_min;
  float value_min = 1000000;
  for (int i = list_of_vertices.size() - 1; i >= 0; i--) {
    int counter = 0;
    for (auto &e: this->list_of_vertices[i].incoming_edges) {
      if(std::abs(e.weight) > 0 && e.utility_propagation < value_min){
        value_min = e.utility_propagation;
        address_min.first = i;
        address_min.second = counter;
      }
      counter++;
    }
  }
  list_of_vertices[address_min.first].incoming_edges[address_min.second].weight = 0;
}

std::vector<std::pair<std::string, float>>  Graph::print_utilities() {
//  std::cout << "Real\t\tActivation\t\tUtil\n";
  float max_real = 0;
  float max_trace = 0;
  float max_util = 0;
  float max_weight = 0;
  float avg_error_trace = 0;
  float avg_error_util = 0;
  float avg_error_weight = 0;
  for (auto &e: this->list_of_vertices[list_of_vertices.size() - 1].incoming_edges) {
    max_real += e.real_utility;
    max_trace += e.activation_trace_utility;
    max_util += e.utility_propagation;
    max_weight += e.weight_utility;
  }
  int wins = 0;
  int total = 0;
  std::vector<std::pair<std::string, float>> return_val;
  for (int i = 0; i < this->list_of_vertices.size(); i++) {
    for (auto &e: this->list_of_vertices[i].incoming_edges) {
//      std::cout << e.real_utility / max_real << "\t\t" << e.activation_trace_utility / max_trace << "\t\t"
//                << e.utility_propagation
//                << std::endl;
      if (std::abs(e.real_utility / max_real - e.activation_trace_utility / max_trace)
          > std::abs(e.real_utility / max_real - e.utility_propagation / max_util))
        wins++;
      total++;
      avg_error_trace += std::abs(e.real_utility / max_real - e.activation_trace_utility / max_trace);
      avg_error_util += std::abs(e.real_utility / max_real - e.utility_propagation/max_util);
      avg_error_weight += std::abs(e.real_utility / max_real - e.weight_utility / max_weight);
    }
  }
  return_val.push_back(std::pair<std::string, float>("Activation_trace", avg_error_trace));
  return_val.push_back(std::pair<std::string, float>("Utility_prop", avg_error_util));
  return_val.push_back(std::pair<std::string, float>("Weight", avg_error_weight));
  return return_val;
}

void Graph::print_graph() {
  int counter = 0;
  std::cout << "digraph g {\n";
  for (auto v: this->list_of_vertices) {
    std::cout << v.id << " [label=\"" << std::to_string(v.value) << "\"];" << std::endl;
//    std::cout << "Vertex no = " << v.id << " Value = " << v.value << std::endl;
    for (auto incoming: v.incoming_edges) {
      std::cout << incoming.from << " -> " << incoming.to << "[label=\"" << std::to_string(incoming.weight) << " "
                << std::to_string(incoming.gradient) << "\"];" << std::endl;
//      std::cout << "Edge from " << list_of_vertices[incoming.from].id << " to " << list_of_vertices[incoming.to].id
//                << " weight " << incoming.weight << " Gradient " << incoming.gradient << std::endl;
    }
    counter++;
  }
  std::cout << "}\n";
}

Graph::Graph(int total_vertices, int total_edges, int input_vertices, int seed) : mt(seed) {
  this->input_vertices = input_vertices;
  for (int i = 0; i < total_vertices; i++) {
    this->list_of_vertices.push_back(Vertex());
  }

  list_of_vertices[list_of_vertices.size() - 1].is_output = true;
  std::uniform_int_distribution<int> distrib(0, total_vertices - 1);
  std::uniform_real_distribution<float> weight_sampler(-0.2, 0.2);
  std::vector<std::vector<int>> temp_matrix;
  for (int i = 0; i < total_vertices; i++) {
    std::vector<int> row;
    for (int j = 0; j < total_vertices; j++) {
      row.push_back(0);
    }
    temp_matrix.push_back(row);
  }
  int edges_counter = 0;
//  std::cout << "From\tTo\n";
  int total_steps = 0;
  while (edges_counter < total_edges && total_steps < 50000) {
    total_steps++;
    int from_vertex = distrib(this->mt);
    if (from_vertex != total_vertices - 1) {
      int to_vertex = -10;
      while (to_vertex <= from_vertex || to_vertex < input_vertices) {
        to_vertex = distrib(this->mt);
      }
      if (temp_matrix[from_vertex][to_vertex] == 0) {
        temp_matrix[from_vertex][to_vertex] = 1;
        edges_counter += 1;
        add_edge(weight_sampler(this->mt), from_vertex, to_vertex);
//        std::cout << from_vertex << " -> " << to_vertex << std::endl;
      }
    }
  }
}

void Graph::set_input_values(std::vector<float> inp) {
  for (int i = 0; i < input_vertices; i++) {
    this->list_of_vertices[i].value = inp[i];
  }
}

float Graph::update_values() {
  for (int i = input_vertices; i < list_of_vertices.size(); i++) {
    list_of_vertices[i].value = 0;
    for (auto &e: list_of_vertices[i].incoming_edges) {
      e.gradient = 0;
      list_of_vertices[i].value += list_of_vertices[e.from].value * e.weight;
    }
  }
  return list_of_vertices[list_of_vertices.size()-1].value;
}

void Graph::estimate_gradient() {
  for (int i = 0; i < list_of_vertices.size(); i++) {
    this->list_of_vertices[i].d_out_d_vertex = 0;
  }

//  Back-prop implementation
  this->list_of_vertices[list_of_vertices.size() - 1].d_out_d_vertex = 1;
  for (int i = list_of_vertices.size() - 1; i >= 0; i--) {
    for (auto &e: this->list_of_vertices[i].incoming_edges) {
      e.gradient = this->list_of_vertices[e.from].value * this->list_of_vertices[e.to].d_out_d_vertex;
      this->list_of_vertices[e.from].d_out_d_vertex += this->list_of_vertices[e.to].d_out_d_vertex * e.weight;
    }
  }
}

void Graph::update_utility() {

  for (int i = 0; i < this->list_of_vertices.size(); i++) {
    for (auto &e: this->list_of_vertices[i].incoming_edges) {
      e.real_utility = e.real_utility * 0.999 + 0.001 * std::abs(e.gradient * e.weight);
      e.activation_trace_utility =
          e.activation_trace_utility * 0.999 + 0.001 * std::abs(e.weight * this->list_of_vertices[e.from].value);
      e.weight_utility = e.weight;
    }
  }
  for (int i = list_of_vertices.size() - 1; i >= 0; i--) {
    list_of_vertices[i].utility = 0;
    if (list_of_vertices[i].is_output) {
      list_of_vertices[i].utility = 1;
    }
  }

  for (int i = list_of_vertices.size() - 1; i >= 0; i--) {
    float total_util = 0;
    for (auto &e: this->list_of_vertices[i].incoming_edges) {
      total_util += e.activation_trace_utility;
    }
    for (auto &e: this->list_of_vertices[i].incoming_edges) {
      e.utility_propagation = (e.activation_trace_utility / total_util * this->list_of_vertices[i].utility);
    }
    for (auto &e: this->list_of_vertices[i].incoming_edges) {
      this->list_of_vertices[e.from].utility += e.utility_propagation;
    }
  }
}