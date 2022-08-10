//
// Created by Khurram Javed on 2022-07-19.
//

#include "../../include/environments/proto_prediction_environments.h"
#include "../../proto_files/experience.pb.h"
#include <iostream>
#include <fstream>

ProtoPredictionEnvironment::ProtoPredictionEnvironment(std::string path, float gamma) {

  this->gamma = gamma;
  std::fstream input(path, std::ios::in | std::ios::binary);
  if (!buffer.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse address book." << std::endl;
  }
  time = 0;
  total = buffer.experiences_size();

  float real_target_cur = 0;
  for (int i = 0; i < total; i++) {
    real_target.push_back(0.0);
  }

  for (int i = buffer.experiences_size() - 1; i >= 0; i--) {
    if (buffer.experiences(i).done()) {
      real_target_cur = 0;
    }
//    std::cout << buffer.experiences(i).reward() << std::endl;
    real_target[i] = real_target_cur;
    float reward = buffer.experiences(i).reward();
    if (reward > 1)
      reward = 1;
    else if (reward < -1)
      reward = -1;
    real_target_cur = real_target_cur * _get_gamma(i) + reward;
  }
}

std::vector<float> ProtoPredictionEnvironment::get_state() {
  std::vector<float> sensor_data;
  for (int j = 0; j < buffer.experiences(time).sensor_reading_size(); j++) {
    sensor_data.push_back(buffer.experiences(time).sensor_reading(j) / 256);
  }
  int action =  buffer.experiences(time).action();
  for(int i = 0; i < 20; i++){
    if(i == action){
      sensor_data.push_back(1);
    }
    else{
      sensor_data.push_back(0);
    }
  }
  return sensor_data;
}

bool ProtoPredictionEnvironment::get_done() {
  return buffer.experiences(time).done();
}
std::vector<float> ProtoPredictionEnvironment::step() {
  time++;
  time = time % total;
  std::vector<float> sensor_data = this->get_state();
  return sensor_data;
}

float ProtoPredictionEnvironment::get_target() {
  return real_target[time];
}

float ProtoPredictionEnvironment::_get_gamma(int time_cur) {
  if (buffer.experiences(time_cur).done()) {
    return 0;
  } else {
    return gamma;
  }
}
float ProtoPredictionEnvironment::get_gamma() {
  if (buffer.experiences(time).done()) {
    return 0;
  } else {
    return gamma;
  }
}
float ProtoPredictionEnvironment::get_reward() {
  float reward = buffer.experiences(time).reward();
  if (reward > 1)
    reward = 1;
  else if (reward < -1)
    reward = -1;
  return reward;
}

