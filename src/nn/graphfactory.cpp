//
// Created by Khurram Javed on 2022-09-24.
//
#include <string>
#include "../../include/nn/graphfactory.h"
#include "../../include/nn/networks/graph.h"
#include "../../include/experiment/Experiment.h"
#include <exception>

Graph *GraphFactory::get_graph(std::string graph_name, Experiment *my_experiment) {
  if (graph_name == "linear_assumption")
    return new GraphLinearAssumptionUtility(my_experiment->get_int_param("vertices"),
                                            my_experiment->get_int_param("edges"),
                                            my_experiment->get_int_param("input_vertices"),
                                            my_experiment->get_int_param("seed"),
                                            my_experiment->get_string_param("non_linearity"),
                                            my_experiment->get_float_param("decay"));
  else if (graph_name == "util_prop")
    return new GraphUtilPropogation(my_experiment->get_int_param("vertices"),
                                    my_experiment->get_int_param("edges"),
                                    my_experiment->get_int_param("input_vertices"),
                                    my_experiment->get_int_param("seed"),
                                    my_experiment->get_string_param("non_linearity"),
                                    my_experiment->get_float_param("decay"));
  else if (graph_name == "util_prop_relative")
    return new GraphUtilPropogationRelative(my_experiment->get_int_param("vertices"),
                                    my_experiment->get_int_param("edges"),
                                    my_experiment->get_int_param("input_vertices"),
                                    my_experiment->get_int_param("seed"),
                                    my_experiment->get_string_param("non_linearity"),
                                    my_experiment->get_float_param("decay"));
  else if (graph_name == "activation_trace")
    return new GraphActivationTraceUtility(my_experiment->get_int_param("vertices"),
                                           my_experiment->get_int_param("edges"),
                                           my_experiment->get_int_param("input_vertices"),
                                           my_experiment->get_int_param("seed"),
                                           my_experiment->get_string_param("non_linearity"),
                                           my_experiment->get_float_param("decay"));
  else if (graph_name == "weight")
    return new GraphWeightUtility(my_experiment->get_int_param("vertices"),
                                  my_experiment->get_int_param("edges"),
                                  my_experiment->get_int_param("input_vertices"),
                                  my_experiment->get_int_param("seed"),
                                  my_experiment->get_string_param("non_linearity"));

  else if (graph_name == "random")
    return new GraphRandomUtility(my_experiment->get_int_param("vertices"),
                                  my_experiment->get_int_param("edges"),
                                  my_experiment->get_int_param("input_vertices"),
                                  my_experiment->get_int_param("seed"),
                                  my_experiment->get_string_param("non_linearity"));

  else if (graph_name == "gradient")
    return new GraphGradientUtility(my_experiment->get_int_param("vertices"),
                                    my_experiment->get_int_param("edges"),
                                    my_experiment->get_int_param("input_vertices"),
                                    my_experiment->get_int_param("seed"),
                                    my_experiment->get_string_param("non_linearity"),
                                    my_experiment->get_float_param("decay"));


  throw std::invalid_argument("Graph not implemented");
  Graph *temp = nullptr;
  return temp;
}
