//
// Created by Khurram Javed on 2022-02-26.
//


#include <iostream>
#include "include/nn/networks/dense_lstm.h"
#include "include/utils.h"
#include "include/nn/utils.h"
//#include "include/experiment/Experiment.h"
#include "include/nn/utils.h"
//#include "include/experiment/Metric.h"
#include <string>

#include "include/environments/animal_learning/tracecondioning.h"

int main(int argc, char *argv[]) {
  auto network = DenseLSTM(1e-3, 0, 3, 3, 20, 1, 3, 1);
  std::pair<int, int> ISI{14, 26};
  std::pair<int, int> ITI{80, 120};
  TracePatterning env = TracePatterning(ISI, ITI, 5, 0);
  auto state = env.reset();
  auto s1 = std::vector<float>{1, 2, 3};
  auto s2 = std::vector<float>{0.1, 0.2, -2};
  auto s3 = std::vector<float>{0.3, -2.5, 1};
  std::vector<std::vector<float>> list_of_inputs;
  for (int i = 0; i < 10; i++) {
    std::vector<float> t = std::vector<float>{float(i), 20.0f - i, float(i) * -0.2f + 3 - i * 0.3f};
    list_of_inputs.push_back(t);
  }

//  for(int types = 0; types < 4; types++) {
  for (int inner = 0; inner < network.W.size(); inner++) {
      std::cout << network.W[inner] << " ,";
  }
  std::cout << "\n";

  for (int inner = 0; inner < network.U.size(); inner++) {
      std::cout << network.U[inner] << " ,";
  }
  std::cout << "\n";

//  return 0;
//  }
  print_vector(network.W);
  std::cout << "U\n";
  print_vector(network.U);
  std::cout << "bias = \n";
  print_vector(network.b);

  std::cout << "State = ";
  print_vector(network.get_state());
  for (auto vec : list_of_inputs) {
    std::cout << "Input = ";
    print_vector(vec);
    network.forward(vec);
    std::cout << "State = ";
  }
network.backward();
//print_vector(network.get_state());
std::cout << "Test\n";
}
