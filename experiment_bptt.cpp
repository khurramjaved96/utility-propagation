//
// Created by Khurram Javed on 2022-01-08.
//

#include <iostream>
#include "include/nn/networks/dense_lstm.h"
#include "include/utils.h"
#include "include/nn/utils.h"
#include <string>
#include <cmath>
#include <chrono>

#include "include/environments/animal_learning/tracecondioning.h"

int main(int argc, char *argv[]) {

  std::cout << "Program started \n";

  std::mt19937 mt(0);
  std::pair<int, int> ISI{14, 26};
  std::pair<int, int> ITI{80, 120};
  TracePatterning env = TracePatterning(ISI, ITI, 5 + 16, 0);

  auto network = DenseLSTM(1e-3,
                           0,
                           16,
                           6+5+1+16,
                           28,
                           1);

  std::cout << "Network created\n";
  float running_error = 0.05;
  auto x = env.reset();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 30000; i++) {
    if(i%1000 == 0)
      std::cout << "i = " << i << std::endl;
    float gamma = 0.9;
    float pred = network.forward(x);
    float real_target = env.get_target(gamma);

    x = env.step();
//    float target = env.get_US() + gamma * network.get_target_without_sideeffects(x);
    float target = 10;
    real_target = 10;
    float error = target - pred;
//    float se = error*error;
    float real_error = (real_target - pred) * (real_target - pred);
    running_error = running_error * 0.9999 + 0.0001 * real_error;
    network.decay_gradient(0.9 * gamma);
//    network.zero_grad();
    network.backward();
    network.update_parameters(error);
    if(i % (38*100) == 0){
      std::cout << "Step = " << i << std::endl;
      std::cout << "Error= " << running_error << std::endl;
      auto end = std::chrono::steady_clock::now();
      std::cout << "Elapsed time in milliseconds for per steps: "
                << std::chrono::duration_cast<std::chrono::duration<double>>(end- start).count()
                << " seconds" << std::endl;
      start = std::chrono::steady_clock::now();

    }
  }
}
