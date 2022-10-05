//
// Created by Khurram Javed on 2022-09-24.
//

#ifndef INCLUDE_GRAPHFACTORY_H_
#define INCLUDE_GRAPHFACTORY_H_
#include "networks/graph.h"
#include <string>
#include "../../include/experiment/Experiment.h"

class GraphFactory {
 public:
  static Graph* get_graph(std::string graph_name, Experiment* myexp);
};

#endif //INCLUDE_GRAPHFACTORY_H_
