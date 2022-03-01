#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>
#include <chrono>

#include <algorithm>
#include <random>
#include <torch/torch.h>
#include "src/nn/networks/torch_LSTM.cpp" //make header later mby
#include "include/utils.h"
#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"

#include "include/environments/mnist/mnist_reader.hpp"
#include "include/environments/mnist/mnist_utils.hpp"

using namespace torch::indexing;

int main(int argc, char *argv[]) {
  Experiment my_experiment = Experiment(argc, argv);

  Metric error_metric = Metric(my_experiment.database_name, "error_table",
                               std::vector < std::string > {"run", "step", "error"},
                               std::vector < std::string > {"int", "int", "real"},
                               std::vector < std::string > {"run", "step"});

  Metric avg_error = Metric(my_experiment.database_name, "predictions",
                            std::vector < std::string > {"run", "global_step", "step", "pred", "target"},
                            std::vector < std::string > {"int", "int", "int", "real", "real"},
                            std::vector < std::string > {"run", "global_step"});
  std::cout << "Program started \n";

  std::mt19937 mt(my_experiment.get_int_param("seed"));
  torch::manual_seed(my_experiment.get_int_param("seed"));
  torch::set_num_threads(1);

  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
                                                              mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data/");

  int total_data_points = 60000;
  std::uniform_int_distribution<int> index_sampler(0, total_data_points - 1);


  std::vector<std::vector<float>> images;
  std::vector<std::vector<float>> targets;

  auto pred = torch::zeros({1}, torch::dtype(torch::kFloat32).requires_grad(true));
  auto h = torch::zeros({1,1,my_experiment.get_int_param("features")}, torch::kFloat32);
  auto c = torch::zeros({1,1,my_experiment.get_int_param("features")}, torch::kFloat32);
  auto state = std::make_tuple(h,c);
  auto state_temp = std::make_tuple(h,c);
  auto network = std::make_shared<LSTM>(28, my_experiment.get_int_param("features"));
  torch::optim::SGD opti(network->parameters(), my_experiment.get_int_param("step_size"));

  std::vector<torch::Tensor> list_of_observations;

  for(int counter = 0; counter < total_data_points; counter++){
    std::vector<float> x_temp;
    for(auto inner: dataset.training_images[counter]){
      x_temp.push_back(float(unsigned(inner)));
    }
    std::vector<float> y_temp;
    if (int(unsigned(dataset.training_labels[counter])) % 2 == 0)
      y_temp.push_back(1.0);
    else
      y_temp.push_back(0.0);
    y_temp.push_back(float(unsigned(dataset.training_labels[counter])));
    images.push_back(x_temp);
    targets.push_back(y_temp);
  }

  int ITI_steps = my_experiment.get_int_param("ITI");
  float running_error = 0.05;
  //float gamma = 0.8; // err @ 0 pred: 0.2 @ 20 step, 0.04 @ 14 step, 0.02 @ 0 step
  auto gamma = torch::zeros({1}, torch::dtype(torch::kFloat32).requires_grad(false));
  gamma.index({0}) = 0.8;
  float accuracy = 0.1;

  std::cout << images.size() << " " << targets.size() << std::endl;
  int layer = 0;
  int global_step = 0;
  std::vector<float> row_x;
  torch::Tensor row_x_tensor, old_row_x_tensor;
  for (int i = 0; i < my_experiment.get_int_param("steps"); i++) {
    auto start = std::chrono::steady_clock::now();

    int index = index_sampler(mt);
    auto x = images[index];
    float y = targets[index][0];

    for(int row = 0; row < 28 + ITI_steps; row ++){
      global_step += 1;
      if (global_step > 1){
        list_of_observations.push_back(row_x_tensor);
        old_row_x_tensor = row_x_tensor; //TODO verify that it is copy
        row_x.clear();
      }

      for(int temp = row*28; temp < (row+1)*28; temp++){
        if(row < 28)
          row_x.push_back(x[temp]/256.0);
        else
          row_x.push_back(0.0);
      }
      torch::Tensor inputs, cloned_inputs;
      // I think we dont need to pass this by value. Clone is copying. Would be cool if
      // it works out without copying
      row_x_tensor = torch::from_blob(row_x.data(), {row_x.size()}, torch::kF32).clone();

      if (list_of_observations.size() == my_experiment.get_int_param("truncation")){
        //std::cout << list_of_observations[0].view({1,1,-1}) << std::endl;
        auto x = list_of_observations[0].view({1,1,-1});
        std::tie(pred, state) = network->forward(x, state);
        state_temp = state; //TODO verify that it is copy by value
        list_of_observations.erase(list_of_observations.begin()); //pop(0)
        //std::cout << torch::stack(list_of_observations).unsqueeze(1) << std::endl;
        x = torch::stack(list_of_observations).unsqueeze(/*dim=*/1); //unsqueeze instead of python ver's squeeze since list is different
        auto batch_pred = torch::zeros({my_experiment.get_int_param("truncation")},
                                       torch::dtype(torch::kFloat32).requires_grad(true));
        std::tie(batch_pred, state_temp) = network->forward(x, state_temp);
        pred = batch_pred.index({-1});

        auto next_pred = torch::zeros({1}, torch::dtype(torch::kFloat32).requires_grad(false));
        auto state_ignore_var = state; //TODO make sure it is copy
        {
          torch::NoGradGuard no_grad;
          auto n_clone = network->clone();
          std::shared_ptr<LSTM> n_copy = std::dynamic_pointer_cast<LSTM>(n_clone);
          //std::cout << row_x_tensor.detach() << std::endl;
          std::tie(next_pred, state_ignore_var) = n_copy->forward(row_x_tensor.detach().view({1,1,-1}), state_temp); //no need for detach here I think
        }
        torch::Tensor target = torch::zeros({1});
        if (row == 27)
          target = torch::Scalar(y) + gamma * next_pred.detach().item();
        else
          target = 0 + gamma * next_pred.detach().item();

        torch::Tensor return_target = torch::zeros({1});
        if (row < 28)
          return_target = y*pow(gamma, 27 - row);

        auto error = target - pred;
        auto return_error = (return_target - pred) * (return_target - pred);

        running_error = running_error*0.9999 + 0.0001*return_error.detach().item<float>();
        network->decay_gradients(my_experiment.get_int_param("lambda") * gamma);
        error.backward();
        opti.step();
        state = std::make_tuple(std::get<0>(state).detach(), std::get<1>(state).detach());
      }
    }
    if(i % 100 == 0){
      std::cout << "Step = " << i << std::endl;
      std::cout << "Error= " << running_error << std::endl;
      auto end = std::chrono::steady_clock::now();
      std::cout << "Elapsed time in milliseconds for per steps: "
                << 1000000 / (1+(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                              my_experiment.get_int_param("steps")))
                << " fps" << std::endl;

    }
  }
}
