//
// Created by Khurram Javed on 2022-08-30.
//

#ifndef INCLUDE_NN_NETWORKS_GRAPH_H_
#define INCLUDE_NN_NETWORKS_GRAPH_H_
#include <vector>
#include <random>
#include "vertex.h"
#include <string>

class Vertex;

class Edge {
 public:
  int edge_id;
  static int edge_id_generator;
  int from;
  int to;
  float gradient;
  float utility;
  float local_utility;
  float weight;
  Edge(float weight, int from, int to);
  float get_weight();
  int get_from();
  int get_to();
  bool is_recurrent();
};


//
//class Graph{
// protected:
//    int input_vertices;
//    std::mt19937 mt;
//    std::vector<Vertex> list_of_vertices;
//    int output_vertex_index;
// public:
//  Graph(int total_vertices, int total_edges, int input_vertices, int seed);
//  void add_edge(float weight, int from, int to);
//  void print_graph();
//  std::vector<std::pair<std::string, float>> print_utilities();
//  void set_input_values(std::vector<float> inp);
//  float update_values();
//  void remove_weight_real_util();
//  void remove_weight_util_prop();
//  void remove_weight_activate_trace();
//  void normalize_weights();
//  void estimate_gradient();
//  void update_utility();
//};


class Graph {
 protected:
  int input_vertices;
  std::mt19937 mt;

  int output_vertex_index;
  float prediction;
 public:
  std::vector<Vertex *> list_of_vertices;
  std::vector<int> get_distribution_of_values();
  void print_utility();
  Graph(int total_vertices, int total_edges, int input_vertices, int seed, std::string vertex_type);
  void add_edge(float weight, int from, int to);
  void print_graph();
  void estimate_gradient();
  float get_prediction();
  void set_input_values(std::vector<float> inp);
  float update_values();
  void prune_weight();
  virtual void update_utility() = 0;
};

class GraphLinearAssumptionUtility : public Graph {
 protected:
  float utility_decay_rate;
 public:
  GraphLinearAssumptionUtility(int total_vertices,
                               int total_edges,
                               int input_vertices,
                               int seed,
                               std::string vertex_type,
                               float utility_decay_rate);
  void update_utility() override;
};

class GradientUtility : public Graph {
 protected:
  float utility_decay_rate;
 public:
  GradientUtility(int total_vertices,
                  int total_edges,
                  int input_vertices,
                  int seed,
                  std::string vertex_type,
                  float utility_decay_rate);
  void update_utility() override;
};



class GradientLocalUtility : public Graph {
 protected:
  float utility_decay_rate;
 public:
  GradientLocalUtility(int total_vertices,
                       int total_edges,
                       int input_vertices,
                       int seed,
                       std::string vertex_type,
                       float utility_decay_rate);
  void update_utility() override;
};

class UtilityPropagation : public Graph {
 protected:
  float utility_decay_rate;
 public:
  UtilityPropagation(int total_vertices,
                     int total_edges,
                     int input_vertices,
                     int seed,
                     std::string vertex_type,
                     float utility_decay_rate);
  void update_utility() override;
};


class WeightUtility : public Graph {
 public:
  WeightUtility(int total_vertices,
                int total_edges,
                int input_vertices,
                int seed,
                std::string vertex_type);
  void update_utility() override;
};

class RandomUtility : public Graph {
 public:
  RandomUtility(int total_vertices,
                int total_edges,
                int input_vertices,
                int seed,
                std::string vertex_type);
  void update_utility() override;
};

class ActivationTrace : public Graph {
 protected:
  float utility_decay_rate;
 public:
  ActivationTrace(int total_vertices,
                  int total_edges,
                  int input_vertices,
                  int seed, std::string vertex_type,
                  float utility_decay_rate);
  void update_utility() override;
};

class GraphLocalUtility : public Graph {
 protected:
  float utility_decay_rate;
 public:
  GraphLocalUtility(int total_vertices,
                    int total_edges,
                    int input_vertices,
                    int seed, std::string vertex_type,
                    float utility_decay_rate);
  void update_utility() override;
};

#endif //INCLUDE_NN_NETWORKS_GRAPH_H_
