//
// Created by Khurram Javed on 2022-09-23.
//

#ifndef INCLUDE_NN_NETWORKS_VERTEX_H_
#define INCLUDE_NN_NETWORKS_VERTEX_H_
#include "graph.h"
#include <string>

class Edge;
class Vertex {
 protected:
 public:
  int id;
  float value;

  bool is_output;
  float sum_of_outgoing_weights;
  float utility;
  float d_out_d_vertex;
  float d_out_d_vertex_before_non_linearity;
  static int id_generator;
  Vertex();
  std::vector<Edge> incoming_edges;
  virtual float forward_with_val(float value);
  virtual float forward();
  float get_value();
  virtual float backward(float val);
  float max_value;
  float min_value;

};

class ReluVertex : public Vertex{

 public:
  ReluVertex();
  float forward_with_val(float val) override;
  float forward() override;
  float backward(float val) override;
};

class SigmoidVertex : public Vertex{
  static float sigmoid(float x);
 public:
  SigmoidVertex();
  float forward_with_val(float val) override;
  float forward() override;
  float backward(float val) override;
};

class BinaryVertex : public Vertex{
 public:
  BinaryVertex();
  float forward_with_val(float val) override;
  float forward() override;
  float backward(float val) override;
};

class VertexFactory{
 public:
  static Vertex* get_vertex(const std::string& type);
};

#endif //INCLUDE_NN_NETWORKS_VERTEX_H_
