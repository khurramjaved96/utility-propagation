//
// Created by Khurram Javed on 2022-08-30.
//

#include "../../../include/nn/networks/graph.h"
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include "../../../include/nn/networks/vertex.h"


//Graph implementation

void Graph::add_edge(float weight, int from, int to) {
  Edge my_edge = Edge(weight, from, to);
  list_of_vertices[to]->incoming_edges.push_back(my_edge);
  list_of_vertices[from]->sum_of_outgoing_weights += std::abs(weight);
//  list_of_vertices[from].outgoing_edges.push_back(my_edge);
}

void Graph::prune_weight() {
  std::pair<int, int> address_min;
  float value_min = 1000000;
  for (int i = list_of_vertices.size() - 1; i >= 0; i--) {
    int counter = 0;
    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
      if (std::abs(e.weight) > 0 && e.utility < value_min) {
        value_min = e.utility;
        address_min.first = i;
        address_min.second = counter;
      }
      counter++;
    }
  }
  list_of_vertices[address_min.first]->incoming_edges[address_min.second].weight = 0;
}

void Graph::print_graph() {
  int counter = 0;
  std::cout << "digraph g {\n";
  for (auto v: this->list_of_vertices) {
    std::cout << v->id << " [label=\"" << std::to_string(v->value) << "\"];" << std::endl;
//    std::cout << "Vertex no = " << v.id << " Value = " << v.value << std::endl;
    for (auto incoming: v->incoming_edges) {
      std::cout << incoming.from << " -> " << incoming.to << "[label=\"" << std::to_string(incoming.weight) << " "
                << std::to_string(incoming.gradient) << "\"];" << std::endl;
//      std::cout << "Edge from " << list_of_vertices[incoming.from].id << " to " << list_of_vertices[incoming.to].id
//                << " weight " << incoming.weight << " Gradient " << incoming.gradient << std::endl;
    }
    counter++;
  }
  std::cout << "}\n";
}

float Graph::get_prediction() {
  return this->prediction;
}

void Graph::print_utility() {
  for (int i = 0; i < this->list_of_vertices.size(); i++) {
    std::cout << "Vertex " << i << std::endl;
//    std::cout << "Vertex value = " << this->list_of_vertices[i]->forward() << std::endl;
    for (auto incoming: this->list_of_vertices[i]->incoming_edges) {
      std::cout << "Incoming edge util = " << incoming.utility << std::endl;
//      std::cout << "Incoming edge grad = " << incoming.gradient << std::endl;
    }
  }
}

Graph::Graph(int total_vertices, int total_edges, int input_vertices, int seed, std::string vertex_type) : mt(seed) {
  this->input_vertices = input_vertices;

//  This convoluted way of initializing the vertices assures that all vertices are stored sequentially in memory; we could call new to create one vertex at a time, but we don't get the guarantee of sequential storage in that case which can cause memory fragmentation

//    Handcrafted graph for debugging
//
//  this->input_vertices = 2;
//  Vertex *i1 = VertexFactory::get_vertex("linear");
//  Vertex *i2 = VertexFactory::get_vertex("linear");
//  Vertex *middle = VertexFactory::get_vertex("sigmoid");
//  Vertex *output = VertexFactory::get_vertex("linear");
//  this->list_of_vertices.push_back(i1);
//  this->list_of_vertices.push_back(i2);
//  this->list_of_vertices.push_back(middle);
//  this->list_of_vertices.push_back(output);
//
//  add_edge(1000, 0, 2);
//  add_edge(1000, 1, 2);
//  add_edge(1.0, 2, 3);
//  add_edge(0.2, 1, 3);
//  std::cout << "Graph crated\n";

  Vertex *mem = new Vertex[input_vertices];
  for (int i = 0; i < input_vertices; i++) {
    this->list_of_vertices.push_back(&mem[i]);
  }

  if (vertex_type == "linear") {
    Vertex *mem = new Vertex[total_vertices - input_vertices - 1];
    for (int i = 0; i < total_vertices - 1 - input_vertices; i++) {
      this->list_of_vertices.push_back(&mem[i]);
    }
  } else if (vertex_type == "relu") {
    ReluVertex *mem = new ReluVertex[total_vertices - input_vertices - 1];
    for (int i = 0; i < total_vertices - 1 - input_vertices; i++) {
      this->list_of_vertices.push_back(&mem[i]);
    }
  } else if (vertex_type == "sigmoid") {
    SigmoidVertex *mem = new SigmoidVertex[total_vertices - input_vertices - 1];
    for (int i = 0; i < total_vertices - 1 - input_vertices; i++) {
      this->list_of_vertices.push_back(&mem[i]);
//      this->list_of_vertices.push_back(new SigmoidVertex());
    }
  } else if (vertex_type == "binary") {
    BinaryVertex *mem = new BinaryVertex[total_vertices - input_vertices - 1];
    for (int i = 0; i < total_vertices - 1 - input_vertices; i++) {
      this->list_of_vertices.push_back(&mem[i]);
    }
  }


  Vertex *prediction_vertex = VertexFactory::get_vertex("linear");
  this->list_of_vertices.push_back(prediction_vertex);

  list_of_vertices[list_of_vertices.size() - 1]->is_output = true;
  std::uniform_int_distribution<int> distrib(0, total_vertices - 1);
  std::uniform_real_distribution<float> weight_sampler(-0.2, 5);
//  std::uniform_real_distribution<float> weight_sampler(-5, 2);
  std::vector<std::vector<int>> temp_matrix;
  for (int i = 0; i < total_vertices; i++) {
    std::vector<int> row;
    for (int j = 0; j < total_vertices; j++) {
      row.push_back(0);
    }
    temp_matrix.push_back(row);
  }
  int edges_counter = 0;
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
    this->list_of_vertices[i]->value = inp[i];
  }
}

std::vector<int> Graph::get_distribution_of_values() {
  std::vector<int> distributon_of_values;
  for (int i = 0; i < 11; i++) {
    distributon_of_values.push_back(0);
  }
  for (int i = input_vertices; i < list_of_vertices.size(); i++) {
    float temp_val = this->list_of_vertices[i]->forward();
//    std::cout << temp_val << std::endl;
    if (temp_val > this->list_of_vertices[i]->max_value)
      temp_val = this->list_of_vertices[i]->max_value;
    if (temp_val < this->list_of_vertices[i]->min_value)
      temp_val = this->list_of_vertices[i]->min_value;

    float range = this->list_of_vertices[i]->max_value - this->list_of_vertices[i]->min_value;
    float bin_size = range / 10;
    int bin_number = int(temp_val / bin_size);
    distributon_of_values[bin_number]++;
  }
  return distributon_of_values;
}
float Graph::update_values() {
  for (int i = input_vertices; i < list_of_vertices.size(); i++) {
    list_of_vertices[i]->value = 0;
    for (auto &e: list_of_vertices[i]->incoming_edges) {
      e.gradient = 0;
      list_of_vertices[i]->value += list_of_vertices[e.from]->forward() * e.weight;
    }
  }
  this->prediction = list_of_vertices[list_of_vertices.size() - 1]->forward();
  return this->prediction;
}

void Graph::estimate_gradient() {
  for (int i = 0; i < list_of_vertices.size(); i++) {
    this->list_of_vertices[i]->d_out_d_vertex = 0;
    this->list_of_vertices[i]->d_out_d_vertex_before_non_linearity = 0;
  }

//  Back-prop implementation
  this->list_of_vertices[list_of_vertices.size() - 1]->d_out_d_vertex = 1;
  this->list_of_vertices[list_of_vertices.size() - 1]->d_out_d_vertex_before_non_linearity = 1;
  for (int i = list_of_vertices.size() - 1; i >= 0; i--) {
    this->list_of_vertices[i]->d_out_d_vertex_before_non_linearity = this->list_of_vertices[i]->d_out_d_vertex;
    this->list_of_vertices[i]->d_out_d_vertex =
        this->list_of_vertices[i]->backward(this->list_of_vertices[i]->value)
            * this->list_of_vertices[i]->d_out_d_vertex;
    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
      e.gradient = this->list_of_vertices[e.from]->forward() * this->list_of_vertices[e.to]->d_out_d_vertex;
      this->list_of_vertices[e.from]->d_out_d_vertex += this->list_of_vertices[e.to]->d_out_d_vertex * e.weight;
    }
  }
}

GraphLinearAssumptionUtility::GraphLinearAssumptionUtility(int total_vertices,
                                                           int total_edges,
                                                           int input_vertices,
                                                           int seed, std::string vertex_type,
                                                           float utility_decay_rate)
    : Graph(total_vertices, total_edges, input_vertices, seed, vertex_type) {
  this->utility_decay_rate = utility_decay_rate;
}

GradientUtility::GradientUtility(int total_vertices,
                                 int total_edges,
                                 int input_vertices,
                                 int seed, std::string vertex_type,
                                 float utility_decay_rate)
    : Graph(total_vertices, total_edges, input_vertices, seed, vertex_type) {
  this->utility_decay_rate = utility_decay_rate;
}

GradientLocalUtility::GradientLocalUtility(int total_vertices,
                                           int total_edges,
                                           int input_vertices,
                                           int seed, std::string vertex_type,
                                           float utility_decay_rate)
    : Graph(total_vertices, total_edges, input_vertices, seed, vertex_type) {
  this->utility_decay_rate = utility_decay_rate;
}

UtilityPropagation::UtilityPropagation(int total_vertices,
                                       int total_edges,
                                       int input_vertices,
                                       int seed, std::string vertex_type,
                                       float utility_decay_rate)
    : Graph(total_vertices, total_edges, input_vertices, seed, vertex_type) {
  this->utility_decay_rate = utility_decay_rate;
}

ActivationTrace::ActivationTrace(int total_vertices,
                                 int total_edges,
                                 int input_vertices,
                                 int seed, std::string vertex_type,
                                 float utility_decay_rate)
    : Graph(total_vertices, total_edges, input_vertices, seed, vertex_type) {
  this->utility_decay_rate = utility_decay_rate;
}

WeightUtility::WeightUtility(int total_vertices,
                             int total_edges,
                             int input_vertices,
                             int seed,
                             std::string vertex_type)
    : Graph(total_vertices, total_edges, input_vertices, seed, vertex_type) {}

RandomUtility::RandomUtility(int total_vertices,
                             int total_edges,
                             int input_vertices,
                             int seed,
                             std::string vertex_type)
    : Graph(total_vertices, total_edges, input_vertices, seed, vertex_type) {
  std::uniform_real_distribution<float> rand_gen(0, 4);
  for (int i = 0; i < this->list_of_vertices.size(); i++) {
    for (auto &e: this->list_of_vertices[i]->incoming_edges) {
      e.utility = rand_gen(this->mt);
    }
  }

}

GraphLocalUtility::GraphLocalUtility(int total_vertices,
                                     int total_edges,
                                     int input_vertices,
                                     int seed, std::string vertex_type,
                                     float utility_decay_rate) : Graph(total_vertices,
                                                                       total_edges,
                                                                       input_vertices,
                                                                       seed,
                                                                       vertex_type) {
  this->utility_decay_rate = utility_decay_rate;
}