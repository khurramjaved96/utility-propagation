//
// Created by Khurram Javed on 2021-12-29.
//
#include <iostream>
#include "include/nn/networks/recurrent_network.h"
#include "include/environments/animal_learning/tracecondioning.h"
#include "include/utils.h"
#include "include/nn/utils.h"
#include "include/experiment/Experiment.h"
#include "include/nn/utils.h"
#include "include/experiment/Metric.h"

int main(int argc, char *argv[]) {
  Experiment my_experiment = Experiment(argc, argv);
  std::cout << "Program started \n";

  float GAMMA = my_experiment.get_float_param("gamma");
  TracePatterning tc = TracePatterning(std::pair<int, int>(my_experiment.get_int_param("ISI_low"),
                                                           my_experiment.get_int_param("ISI_high")),
                                       std::pair<int, int>(my_experiment.get_int_param("ISI_low"),
                                                           my_experiment.get_int_param("ISI_high")),
                                       std::pair<int, int>(my_experiment.get_int_param("ITI_low"),
                                                           my_experiment.get_int_param("ITI_high")),
                                       0,
                                       my_experiment.get_int_param("seed"));

  auto network = RecurrentNetwork(my_experiment.get_float_param("step_size"),
                                  my_experiment.get_int_param("seed"),
                                  7,
                                  1,
                                  my_experiment.get_int_param("features"),
                                  my_experiment.get_int_param("connections"));
  network.forward(tc.reset());
  std::vector<float> target_vector;
  target_vector.push_back(tc.get_target(GAMMA));
  network.backward(target_vector, true);
  print_vector(network.read_all_values());
  float running_error = 0;
  for (int i = 0; i < my_experiment.get_int_param("steps"); i++) {
    auto inp = tc.step();
    if (i % 10000 == 0) {
//      print_vector(network.read_all_weights());
      std::cout << "Step = " << i << std::endl;
      std::cout << "Running error = " << running_error << std::endl;
    }

//    std::vector<float> inp(10, 1);
//    std::cout << "Step = " << i << std::endl;

//    std::cout << "Input vector ";
//    print_vector(inp);
    network.forward(inp);
    target_vector[0] = tc.get_target(GAMMA);
    running_error = running_error * 0.999 + 0.001
        * ((target_vector[0] - network.read_output_values()[0]) * (target_vector[0] - network.read_output_values()[0]));
    network.backward(target_vector, true);

    if(i % 20000 < 100){
//      std::cout << "Input : ";
//      print_vector(inp);
      std::cout << "Prediction\t" << network.read_output_values()[0] << "\tTarget\t" << tc.get_target(GAMMA)  << std::endl;

    }
//    print_vector(network.read_all_values());
//    print_vector(network.read_output_values());
//    std::cout << "Target = " << tc.get_target(GAMMA) << std::endl;
  }

}