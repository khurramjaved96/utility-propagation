//
// Created by Khurram Javed on 2022-07-18.
//

#include <vector>
#include "include/environments/proto_prediction_environments.h"
#include <iostream>
#include "include/utils.h"
#include <fstream>

#include <iostream>
#include "include/nn/networks/lstm_incremental_networks.h"
#include "include/nn/networks/lstm_bptt.h"
#include "include/nn/networks/network_factory.h"
#include "include/utils.h"
#include "include/nn/utils.h"
#include "include/experiment/Experiment.h"
#include "include/nn/utils.h"
#include "include/experiment/Metric.h"
#include <string>
#include "include/nn/networks/base_lstm.h"
#include "include/environments/animal_learning/tracecondioning.h"

//
int main(int argc, char *argv[]) {

  Experiment *my_experiment = new ExperimentJSON(argc, argv);

  Metric error_metric = Metric(my_experiment->database_name, "error_table",
                               std::vector<std::string>{"run", "step", "error"},
                               std::vector<std::string>{"int", "int", "real"},
                               std::vector<std::string>{"run", "step"});

  Metric avg_error = Metric(my_experiment->database_name, "predictions",
                            std::vector<std::string>{"run", "step", "pred",
                                                     "target", "reward"},
                            std::vector<std::string>{"int", "int",
                                                     "real", "real", "real"},
                            std::vector<std::string>{"run", "step"});
//  Metric network->state = Metric(my_experiment->database_name, "network->state",
//                                std::vector<std::string>{"run", "step", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
//                                                         "x8", "x9"},
//                                std::vector<std::string>{"int", "int", "real", "real", "real", "real", "real", "real",
//                                                         "real", "real", "real", "real"},
//                                std::vector<std::string>{"run", "step"});

  std::cout << "Program started \n";

  std::mt19937 mt(my_experiment->get_int_param("seed"));
  ProtoPredictionEnvironment env = ProtoPredictionEnvironment(my_experiment->get_string_param("environment_proto"),
                                                              my_experiment->get_float_param("gamma"));

  BaseLSTM *network = NetworkFactory::get_network(my_experiment);
  std::cout << "Network created\n";
  float running_error = 0.05;

//  for(int i = 0; i < 10000; i++){
//    auto features = env.step();
//    auto target = env.get_target();
//    auto reward = env.get_reward();
//
//    print_vector(features);
//    std::cout << "Features: " << features.size() << " Target: " << target << " Reward: " << reward << " Gamma " << env.get_gamma() << std::endl;
//    if(env.get_done())
//      break;
//  }

  auto x = env.step();
  int layer = 0;
  for (int i = 0; i < my_experiment->get_int_param("steps"); i++) {

    if (i % my_experiment->get_int_param("freq") == my_experiment->get_int_param("freq") - 1) {
      layer++;
      std::cout << "Increasing layer\n";
    }

    float pred = network->forward(x);
    float real_target = env.get_target();

    if (i % 2000000 < 200) {
      std::vector<std::string> cur_error;
      cur_error.push_back(std::to_string(my_experiment->get_int_param("run")));
      cur_error.push_back(std::to_string(i));
      cur_error.push_back(std::to_string(pred));
      cur_error.push_back(std::to_string(real_target));
      cur_error.push_back(std::to_string(env.get_reward()));
      avg_error.record_value(cur_error);
    }

    x = env.step();
    float target = env.get_reward() + env.get_gamma() * network->get_target_without_sideeffects(x);

    float error = target - pred;
    float real_error = (real_target - pred) * (real_target - pred);
    running_error = running_error * 0.99999 + 0.00001 * real_error;
    network->decay_gradient(my_experiment->get_float_param("lambda") * env.get_gamma());
    network->backward(layer);
    network->update_parameters(layer, error);
    if (i % 200000 == 0) {
      std::vector<std::string> cur_error;
      cur_error.push_back(std::to_string(my_experiment->get_int_param("run")));
      cur_error.push_back(std::to_string(i));
      cur_error.push_back(std::to_string(running_error));
      error_metric.record_value(cur_error);
    }

    if (i % 2000000 == 0) {
      std::cout << "Step = " << i << " Error = " << running_error << std::endl;
      error_metric.commit_values();
      avg_error.commit_values();
//      network->state.commit_values();
    }
  }
  error_metric.commit_values();
  avg_error.commit_values();

}