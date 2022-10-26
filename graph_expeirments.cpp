//
// Created by Khurram Javed on 2022-08-30.
//

#include "include/nn/networks/graph.h"
#include "include/nn/networks/vertex.h"
#include <iostream>
#include <vector>
#include <random>

#include "include/utils.h"
#include "include/experiment/Experiment.h"
#include "include/nn/weight_initializer.h"
#include "include/experiment/Metric.h"
#include <string>
#include "include/nn/graphfactory.h"

int main(int argc, char *argv[]) {
  Experiment *my_experiment = new ExperimentJSON(argc, argv);

  Metric error_metric = Metric(my_experiment->database_name, "prediction_error",
                               std::vector<std::string>{"run", "step", "method", "error"},
                               std::vector<std::string>{"int", "int", "VARCHAR(30)", "real"},
                               std::vector<std::string>{"run", "step", "method"});

  Metric error_from_optimal_util = Metric(my_experiment->database_name, "distance_from_optimal_utility",
                                          std::vector<std::string>{"run", "method", "error"},
                                          std::vector<std::string>{"int", "VARCHAR(30)", "real"},
                                          std::vector<std::string>{"run", "method"});


  if (my_experiment->get_int_param("input_vertices") < my_experiment->get_int_param("vertices")
      && my_experiment->get_int_param("vertices") * 2 < my_experiment->get_int_param("edges") or true) {
    int win = 0;
    std::normal_distribution<float>
        input_sampler(my_experiment->get_float_param("input_mean"), my_experiment->get_float_param("input_std"));
    int seed = my_experiment->get_int_param("seed");
    std::mt19937 mt(seed);
    Graph *g_pruned = GraphFactory::get_graph(my_experiment->get_string_param("method"), my_experiment);
    Graph *g_unpruned = GraphFactory::get_graph(my_experiment->get_string_param("method"), my_experiment);
    WeightInitializer initializer(my_experiment->get_vector_of_floats("init")[0],my_experiment->get_vector_of_floats("init")[1], my_experiment->get_int_param("seed"));
    g_pruned = initializer.initialize_weights(g_pruned);
    initializer = WeightInitializer(my_experiment->get_vector_of_floats("init")[0],my_experiment->get_vector_of_floats("init")[1], my_experiment->get_int_param("seed"));
    g_unpruned = initializer.initialize_weights(g_unpruned);

    float pruning_error = 0;
    for (int i = 0; i < my_experiment->get_int_param("steps"); i++) {
      if (i % 100000 == 99999) {
        std::cout << "Step: " << i << std::endl;
////        std::cout << "Distribution of values\n";
////        g_pruned->print_utility();
////        g_pruned->prune_weight();
////        print_vector(g_unpruned->get_distribution_of_values());
      }

      std::vector<float> inps;
      for (int t = 0; t < my_experiment->get_int_param("input_vertices"); t++)
        inps.push_back(input_sampler(mt));
//        inps.push_back(1);

      g_pruned->set_input_values(inps);
      g_unpruned->set_input_values(inps);

      g_pruned->update_values();
      g_unpruned->update_values();
      g_pruned->update_utility();

      pruning_error =
          pruning_error * 0.999 + std::abs(g_pruned->get_prediction() - g_unpruned->get_prediction()) * 0.001;

      if (i % my_experiment->get_int_param("frequency") == my_experiment->get_int_param("frequency") - 1) {
//        g_pruned->print_utility();
        g_pruned->prune_weight();

        std::vector<std::string> val;
        val.push_back(std::to_string(my_experiment->get_int_param("run")));
        val.push_back(std::to_string(i));
        val.push_back(my_experiment->get_string_param("method"));
        val.push_back(std::to_string(pruning_error));
        error_metric.record_value(val);
      }
      if (i % 50000 == 49999) {
        error_metric.commit_values();
        error_from_optimal_util.commit_values();
      }
    }
    error_metric.commit_values();
    error_from_optimal_util.commit_values();

  }
//  std::cout << "Win percentage " << float(win) / total_seeds << std::endl;
}