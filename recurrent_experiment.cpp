//
// Created by Khurram Javed on 2021-12-29.
//
#include <iostream>
#include "include/nn/networks/recurrent_network.h"
#include "include/environments/animal_learning/tracecondioning.h"
#include "include/utils.h"
#include "include/nn/utils.h"

int main(){
  std::cout << "Hello world\n";

  TracePatterning tc = TracePatterning(std::pair<int, int>(4, 4),
                                           std::pair<int, int>(4, 4),
                                           std::pair<int, int>(20, 20), 0, 0);
  auto network = RecurrentNetwork(1e-4, 0, 7, 1, 200, 6);
  network.forward(tc.reset());
  std::vector<float> target_vector;
  target_vector.push_back(tc.get_target(0.75));
  network.backward(target_vector, true);
  print_vector(network.read_all_values());
  float running_error = 0;
  for(int i = 0; i < 5000000; i++) {
    auto inp = tc.step();
    if(i % 10000 == 0){
//      print_vector(network.read_all_weights());
      std::cout << "Running error = " << running_error << std::endl;
    }

//    std::vector<float> inp(10, 1);
//    std::cout << "Step = " << i << std::endl;

//    std::cout << "Input vector ";
//    print_vector(inp);
    network.forward(inp);
    target_vector[0] = tc.get_target(0.75);
    running_error = running_error*0.999 + 0.001*((target_vector[0] - network.read_output_values()[0])*(target_vector[0] - network.read_output_values()[0]));
    network.backward(target_vector, true);

//    if(i % 20000 < 100){
//      std::cout << "Input : ";
//      print_vector(inp);
//      std::cout << "Prediction : ";
//      print_vector(network.read_output_values());
//      std::cout << "Target : " << tc.get_target(0.75) << std::endl;
//
//    }
//    print_vector(network.read_all_values());
//    print_vector(network.read_output_values());
//    std::cout << "Target = " << tc.get_target(0.75) << std::endl;
  }

}