//
// Created by Khurram Javed on 2022-06-03.
//

#ifndef INCLUDE_NN_NETWORKS_BASE_LSTM_H_
#define INCLUDE_NN_NETWORKS_BASE_LSTM_H_


class BaseLSTM {
 public:


  virtual float  get_target_without_sideeffects(std::vector<float> inputs) = 0;

  BaseLSTM(){};


  virtual float forward(std::vector<float> inputs) = 0;

  virtual void zero_grad() = 0;

  virtual void decay_gradient(float decay_rate) = 0;

  virtual void backward() = 0;

  virtual void update_parameters(int layer, float error) = 0;

};


#endif //INCLUDE_NN_NETWORKS_BASE_LSTM_H_
