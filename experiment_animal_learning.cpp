//
// Created by Khurram Javed on 2022-01-08.
//

#include <iostream>
#include "include/nn/networks/td_lambda.h"
#include "include/utils.h"
#include "include/nn/utils.h"
#include "include/experiment/Experiment.h"
#include "include/nn/utils.h"
#include "include/experiment/Metric.h"
#include <string>

#include "include/environments/animal_learning/tracecondioning.h"

int main(int argc, char *argv[]) {

  Experiment my_experiment = Experiment(argc, argv);

  Metric error_metric = Metric(my_experiment.database_name, "error_table",
                               std::vector < std::string > {"run", "step", "error"},
                               std::vector < std::string > {"int", "int", "real"},
                               std::vector < std::string > {"run", "step"});

  Metric avg_error = Metric(my_experiment.database_name, "predictions",
                            std::vector < std::string > {"run", "step", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "pred", "target"},
                            std::vector < std::string > {"int", "int", "real" , "real", "real", "real", "real", "real", "real", "real", "real"},
                            std::vector < std::string > {"run", "step"});
  Metric network_state = Metric(my_experiment.database_name, "network_state",
                                std::vector < std::string > {"run", "step", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"},
                                std::vector <std::string > {"int", "int", "real", "real", "real", "real", "real", "real", "real", "real", "real", "real"},
                                std::vector <std::string> {"run", "step"});

  std::cout << "Program started \n";

  std::mt19937 mt(my_experiment.get_int_param("seed"));
  std::pair<int, int> ISI{14, 26};
  std::pair<int, int> ITI{80, 120};
  TracePatterning env = TracePatterning(ISI, ITI, 5, my_experiment.get_int_param("seed"));

  auto network = TDLambda(my_experiment.get_float_param("step_size"),
                       my_experiment.get_int_param("seed"),
                       6 + 5 + 1,
                       1,
                       my_experiment.get_int_param("features"),
                       my_experiment.get_int_param("width")
                       );

  std::cout << "Network created\n";
  float running_error = 0.05;
  auto x = env.reset();
  int layer = my_experiment.get_int_param("start");
  for (int i = 0; i < my_experiment.get_int_param("steps"); i++) {

    if(i % my_experiment.get_int_param("freq") == my_experiment.get_int_param("freq") - 1){
      layer++;
      std::cout << "Increasing layer\n";
      print_vector(network.prediction_weights);
      std::cout << "Avg feature value\n";
      print_vector(network.avg_feature_value);
      print_vector(network.feature_mean);
      print_vector(network.feature_std);
    }



    float gamma = 0.9;
    float pred = network.forward(x);
    float real_target = env.get_target(gamma);
//    std::cout << "Step = " << i << "\n";
//    print_vector(x);
//    print_vector(network.get_state());
//    std::cout << "Prediction = " << pred << std::endl;
//    print_vector(network.prediction_weights);
//    float temp_pred = 0;
//    for(int temp_counter = 0; temp_counter < network.prediction_weights.size(); temp_counter++){
//      temp_pred += network.prediction_weights[temp_counter]*network.get_state()[temp_counter];
//    }
//    std::cout << "Temp pred = " << temp_pred << std::endl;
//    std::cout << "\n\n";
    if(i%100000 < 400){
      std::vector<std::string> cur_error;
      cur_error.push_back(std::to_string(my_experiment.get_int_param("run")));
      cur_error.push_back(std::to_string(i));
      for(int inner_c = 0;  inner_c <7; inner_c++){
        cur_error.push_back(std::to_string(x[inner_c]));
      }
      cur_error.push_back(std::to_string(pred));
      cur_error.push_back(std::to_string(real_target));
      avg_error.record_value(cur_error);
    }
    if(i > 5000000 && i < 5200000){
      std::vector<float> network_state_cur = network.get_normalized_state();
      std::vector<std::string> cur_state;
      cur_state.push_back(std::to_string(my_experiment.get_int_param("run")));
      cur_state.push_back(std::to_string(i));
      for(int counter_vec = 0; counter_vec < network_state_cur.size(); counter_vec++)
        cur_state.push_back(std::to_string(network_state_cur[counter_vec]));
      network_state.record_value(cur_state);
    }
    x = env.step();
    float target = env.get_US() + gamma*network.get_target_without_sideeffects(x);

    float error = target - pred;
//    float se = error*error;
    float real_error = (real_target - pred)*(real_target - pred);
    running_error = running_error*0.99999 + 0.00001*real_error;
    network.decay_gradient(my_experiment.get_float_param("lambda")*gamma);
//    network.zero_grad();
    network.backward();
    network.update_parameters(layer, error);
    if(i%50000 == 20000){
      std::vector<std::string> cur_error;
      cur_error.push_back(std::to_string(my_experiment.get_int_param("run")));
      cur_error.push_back(std::to_string(i));
      cur_error.push_back(std::to_string(running_error));
      error_metric.record_value(cur_error);
    }


//    if(i%10000 < 300)
//      std::cout << "Prediction = " << pred << " Target " << real_target << " Bootstrapped target "<< target << " US " << env.get_US() << std::endl;
    if (i % 100000 == 0) {
      std::cout << "Step = " << i << " Error = " << running_error << std::endl;
      error_metric.commit_values();
      avg_error.commit_values();
      network_state.commit_values();
    }
  }
  error_metric.commit_values();
  avg_error.commit_values();
  network_state.commit_values();
}
