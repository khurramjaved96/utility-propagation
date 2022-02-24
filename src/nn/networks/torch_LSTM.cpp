#include <torch/torch.h>
#include <utility>
#include <functional>
#include <tuple>


struct LSTM : torch::nn::Module {
  torch::nn::LSTM recurrent_unit{nullptr};
  torch::nn::Linear predictor{nullptr};

  LSTM(int inputs, int hidden_units){
    recurrent_unit = register_module("recurrent_unit", torch::nn::LSTM(inputs, hidden_units));
    predictor = register_module("predictor", torch::nn::Linear(hidden_units, 1));
    predictor->weight.data() = predictor->weight.data() * 0;
    predictor->bias.data() = predictor->bias.data() * 0;
  }

  std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
    forward(torch::Tensor x, std::tuple<torch::Tensor, torch::Tensor> state){
    torch::Tensor features, output;
    std::tie(features, state) = recurrent_unit->forward(x, state);
    features = features.view({x.size(0), -1});
    output = predictor->forward(features);
    return std::make_tuple(output, state);
  }

  void decay_gradients(torch::Tensor decay_rate){
    for (auto& param : this->named_parameters()){
      if (param.value().grad().defined()){
        auto g = param.value().grad();
        g = param.value().grad() * decay_rate;
      }
    }
  }
};
