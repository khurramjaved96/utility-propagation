//
// Created by Khurram Javed on 2022-08-30.
//

#include "include/nn/networks/graph.h"
#include <iostream>
#include <vector>
#include <random>

#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"
#include <string>

int main(int argc, char *argv[]) {
  Experiment *my_experiment = new ExperimentJSON(argc, argv);

  Metric error_metric = Metric(my_experiment->database_name, "prediction_error",
                               std::vector<std::string>{"run",  "step", "method", "error"},
                               std::vector<std::string>{"int", "int", "VARCHAR(30)", "real"},
                               std::vector<std::string>{"run", "step", "method"});


  Metric error_from_optimal_util = Metric(my_experiment->database_name, "distance_from_optimal_utility",
                               std::vector<std::string>{"run",  "method", "error"},
                               std::vector<std::string>{"int",  "VARCHAR(30)", "real"},
                               std::vector<std::string>{"run", "method"});

  if (my_experiment->get_int_param("input_vertices") < my_experiment->get_int_param("vertices")
      && my_experiment->get_int_param("vertices") * 2 < my_experiment->get_int_param("edges")) {
    int win = 0;
    std::normal_distribution<float>
        input_sampler(my_experiment->get_float_param("input_mean"), my_experiment->get_float_param("input_std"));
    int seed =  my_experiment->get_int_param("seed");
    std::mt19937 mt(seed);
    Graph g = Graph(my_experiment->get_int_param("vertices"),
                    my_experiment->get_int_param("edges"),
                    my_experiment->get_int_param("input_vertices"),
                    seed);

    Graph util_pruning = Graph(my_experiment->get_int_param("vertices"),
                               my_experiment->get_int_param("edges"),
                               my_experiment->get_int_param("input_vertices"),
                               seed);

    Graph gt_pruning = Graph(my_experiment->get_int_param("vertices"),
                             my_experiment->get_int_param("edges"),
                             my_experiment->get_int_param("input_vertices"),
                             seed);

    Graph activate_trace_pruning = Graph(my_experiment->get_int_param("vertices"),
                                         my_experiment->get_int_param("edges"),
                                         my_experiment->get_int_param("input_vertices"),
                                         seed);
//      g.normalize_weights();
    float gt_error = 0;
    float util_error = 0;
    float actiavation_trace_error = 0;
    for (int i = 0; i < my_experiment->get_int_param("steps"); i++) {
      if(i%100000 == 0){
        std::cout << "Step: " << i << std::endl;
      }
      std::vector<float> inps;
      for (int t = 0; t < my_experiment->get_int_param("input_vertices"); t++)
        inps.push_back(input_sampler(mt));

      g.set_input_values(inps);
      float actual_pred = g.update_values();
      g.estimate_gradient();
      g.update_utility();

      util_pruning.set_input_values(inps);
      util_error = util_error * 0.99999 + std::abs(util_pruning.update_values() - actual_pred) * 0.00001;
      util_pruning.estimate_gradient();
      util_pruning.update_utility();

      gt_pruning.set_input_values(inps);
      gt_error = gt_error * 0.99999 + std::abs(gt_pruning.update_values() - actual_pred) * 0.00001;
      gt_pruning.estimate_gradient();
      gt_pruning.update_utility();

      activate_trace_pruning.set_input_values(inps);
      actiavation_trace_error =
          actiavation_trace_error * 0.99999 + std::abs(activate_trace_pruning.update_values() - actual_pred) * 0.00001;
      activate_trace_pruning.estimate_gradient();
      activate_trace_pruning.update_utility();

      if (i % 10000 == 9999) {
        util_pruning.remove_weight_util_prop();
        gt_pruning.remove_weight_real_util();
        activate_trace_pruning.remove_weight_activate_trace();
        std::vector<std::pair<std::string, float>> key_values;
        key_values.push_back(std::pair<std::string, float>("Util_prop_pruning", util_error));
        key_values.push_back(std::pair<std::string, float>("Optimal_pruning", gt_error));
        key_values.push_back(std::pair<std::string, float>("Local_pruning", actiavation_trace_error));
        for(auto& e : key_values){
          std::vector<std::string> val;
          val.push_back(std::to_string(my_experiment->get_int_param("run")));
          val.push_back(std::to_string(i));
          val.push_back(e.first);
          val.push_back(std::to_string(e.second));
          error_metric.record_value(val);
        }
      }
    }
    auto return_val = g.print_utilities();
    for (auto v: return_val) {
      std::vector<std::string> val;
      val.push_back(std::to_string(my_experiment->get_int_param("run")));
      val.push_back(std::to_string(seed));
      val.push_back(v.first);
      val.push_back(std::to_string(v.second));
      error_from_optimal_util.record_value(val);
    }


    error_metric.commit_values();
    error_from_optimal_util.commit_values();
  }
//  std::cout << "Win percentage " << float(win) / total_seeds << std::endl;
}