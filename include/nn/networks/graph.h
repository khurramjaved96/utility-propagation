//
// Created by Khurram Javed on 2022-08-30.
//

#ifndef INCLUDE_NN_NETWORKS_GRAPH_H_
#define INCLUDE_NN_NETWORKS_GRAPH_H_
#include <vector>
#include <random>

class Vertex;

class Edge{
 public:
  int edge_id;
  static int edge_id_generator;
  int from;
  int to;
  float gradient;
  float real_utility;
  float activation_trace_utility;
  float weight_utility;
  float utility_propagation;
  float weight;
  Edge(float weight, int from, int to);
  float get_weight();
  int get_from();
  int get_to();
  bool is_recurrent();
};

class Vertex{

 public:
  int id;
  float value;
  bool is_output;
  float sum_of_outgoing_weights;
  float utility;
  float d_out_d_vertex;
  static int id_generator;
  Vertex();
  std::vector<Edge> incoming_edges;
//  std::vector<Edge> outgoing_edges;
  void add_incoming_edge(Edge e);
  std::vector<Edge> get_incoming_edges();
};

class Graph{
 protected:
    int input_vertices;
    std::mt19937 mt;
    std::vector<Vertex> list_of_vertices;
    int output_vertex_index;
 public:
  Graph(int total_vertices, int total_edges, int input_vertices, int seed);
  void add_edge(float weight, int from, int to);
  void print_graph();
  std::vector<std::pair<std::string, float>> print_utilities();
  void set_input_values(std::vector<float> inp);
  float update_values();
  void remove_weight_real_util();
  void remove_weight_util_prop();
  void remove_weight_activate_trace();
  void normalize_weights();
  void estimate_gradient();
  void update_utility();
};


class GraphBase{
 protected:
  int input_vertices;
  std::mt19937 mt;
  std::vector<Vertex> list_of_vertices;
  int output_vertex_index;
 public:
  GraphBase(int total_vertices, int total_edges, int input_vertices, int seed);
  void add_edge(float weight, int from, int to);
  void print_graph();
  std::vector<std::pair<std::string, float>> print_utilities();
  void set_input_values(std::vector<float> inp);
  float update_values();
  virtual void prune_weight() = 0;
  virtual void update_utility() = 0;
};





#endif //INCLUDE_NN_NETWORKS_GRAPH_H_
