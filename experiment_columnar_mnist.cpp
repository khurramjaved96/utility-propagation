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
#include <string>

#include "include/environments/mnist/mnist_reader.hpp"
#include "include/environments/mnist/mnist_utils.hpp"


int main(int argc, char *argv[]) {
  Experiment my_experiment = Experiment(argc, argv);

  Metric error_metric = Metric(my_experiment.database_name, "error_table",
                               std::vector < std::string > { "run", "step", "error", "accuracy"},
                               std::vector < std::string > {"int", "int", "real", "real"},
                               std::vector < std::string > {"run", "step"});


  Metric avg_error = Metric(my_experiment.database_name, "test_accuracy",
                            std::vector < std::string > {"run", "step", "accuracy"},
                            std::vector < std::string > {"int", "int", "real"},
                            std::vector < std::string > {"run", "step"});


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
                                  my_experiment.get_int_param("features"), my_experiment.get_float_param("init"));

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

  for(int counter = 0; counter < total_test_points; counter++){
    std::vector<float> x_temp;
    for(auto inner: dataset.test_images[counter]){
      x_temp.push_back(float(unsigned(inner)));
    }
    std::vector<float> y_temp;
    y_temp.push_back(float(unsigned(dataset.test_labels[counter])));
    images_test.push_back(x_temp);
    targets_test.push_back(y_temp);
  }

  float accuracy = 0.1;
  float running_error = 5;

  std::cout << images.size() << " " << targets.size() << std::endl;

  for (int i = 0; i < my_experiment.get_int_param("steps"); i++) {
    float error = 0;
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
        error = network.backward(y);
        network.update_parameters();
//        print_vector(network.read_output_values());
//        print_vector(y);
      }
    }

    int prediction = argmax(network.read_output_values());

    running_error = running_error*0.9999 + 0.0001*error;
    if(prediction == y_index){
      accuracy = accuracy*0.9999 + 0.0001;
    }
    else
      accuracy*= 0.9999;
    if(i % 1000 == 0){
      std::vector<std::string> cur_error;
      cur_error.push_back(std::to_string(my_experiment.get_int_param("run")));
      cur_error.push_back(std::to_string(i));
      cur_error.push_back(std::to_string(running_error));
      cur_error.push_back(std::to_string(accuracy));
      error_metric.record_value(cur_error);
      std::cout << "Step = " << i << std::endl;
      std::cout << "Accuracy = " << accuracy << std::endl;

    }
    if(i%60000 == 0) {
//      Evaluate test set
      network.reset_state();
      float correct_predictions = 0;
      for (int test_set_loop = 0; test_set_loop < 10000; test_set_loop++) {
        network.reset_state();
        int index = test_set_loop;
        auto x = images_test[index];
        float y_index = targets_test[index][0];

        for (int row = 0; row < 28; row++) {

          std::vector<float> row_x;

          for (int temp = row * 28; temp < (row + 1) * 28; temp++) {
            row_x.push_back(x[temp] / 256.0);
          }
          network.forward(row_x);

        }
        int prediction = argmax(network.read_output_values());
        if(prediction == y_index)
          correct_predictions+=1;
      }
      float accuracy = correct_predictions/10000.0;
      std::vector<std::string> cur_error;
      cur_error.push_back(std::to_string(my_experiment.get_int_param("run")));
      cur_error.push_back(std::to_string(i));
      cur_error.push_back(std::to_string(accuracy));
      avg_error.record_value(cur_error);
      std::cout << "Step = " << i << std::endl;
      std::cout << "Test accuracy = " << accuracy << std::endl;
    }
    if(i%10000 == 0) {
      error_metric.commit_values();
      avg_error.commit_values();
    }
    network.reset_state();
  }
}
