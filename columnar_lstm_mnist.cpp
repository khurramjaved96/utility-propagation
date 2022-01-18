//
// Created by Khurram Javed on 2022-01-08.
//

#include <iostream>
#include "include/nn/networks/columnar_lstm.h"
#include "include/utils.h"
#include "include/nn/utils.h"
#include "include/experiment/Experiment.h"
#include "include/nn/utils.h"
#include "include/experiment/Metric.h"

#include "include/environments/mnist/mnist_reader.hpp"
#include "include/environments/mnist/mnist_utils.hpp"


int main(int argc, char *argv[]) {
  Experiment my_experiment = Experiment(argc, argv);
  std::cout << "Program started \n";

  std::mt19937 mt(my_experiment.get_int_param("seed"));

  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
                                                              mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data/");

  int total_data_points = 60000;
  int total_test_points = 10000;
  std::uniform_int_distribution<int> index_sampler(0, total_data_points - 1);


  std::vector<std::vector<float>> images;
  std::vector<std::vector<float>> targets;
//
  std::vector<std::vector<float>> images_test;
  std::vector<std::vector<float>> targets_test;


  auto network = ColumnarLSTM(my_experiment.get_float_param("step_size"),
                                  my_experiment.get_int_param("seed"),
                                  28,
                                  10,
                                  my_experiment.get_int_param("features"));

  for(int counter = 0; counter < total_data_points; counter++){
    std::vector<float> x_temp;
    for(auto inner: dataset.training_images[counter]){
      x_temp.push_back(float(unsigned(inner)));
    }
    std::vector<float> y_temp;
    y_temp.push_back(float(unsigned(dataset.training_labels[counter])));
    images.push_back(x_temp);
    targets.push_back(y_temp);
  }

  float accuracy = 0.1;

  std::cout << images.size() << " " << targets.size() << std::endl;

  for (int i = 0; i < my_experiment.get_int_param("steps"); i++) {
    int index = index_sampler(mt);
//    std::cout << "INdex =  " << index << std::endl;
    auto x = images[index];
    float y_index = targets[index][0];
    std::vector<float> y(10);
    y[y_index] = 1;

    for(int row = 0; row < 28; row ++){
      std::vector<float> row_x;
      for(int temp = row*28; temp < (row+1)*28; temp++){
        row_x.push_back(x[temp]/256.0);
      }
//      print_vector(row_x);
      network.forward(row_x);

      if(row == 27) {
        network.backward(y);
        network.update_parameters();
//        print_vector(network.read_output_values());
//        print_vector(y);
      }
    }

    int prediction = argmax(network.read_output_values());


    if(prediction == y_index){
      accuracy = accuracy*0.999 + 0.001;
    }
    else
      accuracy*= 0.999;
    if(i % 100 == 0){
      std::cout << "Step = " << i << std::endl;
      std::cout << "Accuracy = " << accuracy << std::endl;

    }
    network.reset_state();
  }
}
