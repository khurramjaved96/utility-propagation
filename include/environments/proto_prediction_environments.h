//
// Created by Khurram Javed on 2022-07-19.
//

#ifndef INCLUDE_ENVIRONMENTS_PROTO_PREDICTION_ENVIRONMENTS_H_
#define INCLUDE_ENVIRONMENTS_PROTO_PREDICTION_ENVIRONMENTS_H_

#include <string>
#include <vector>
#include "../../proto_files/experience.pb.h"


class ProtoPredictionEnvironment {
 private:
  float _get_gamma(int time_cur);
 public:
  atari_prediction::ExperienceBuffer buffer;
  std::vector<float> real_target;
  int time;
  int total;
  float gamma;
  ProtoPredictionEnvironment(std::string path, float gamma);
  std::vector<float> get_state();
  std::vector<float> step();
  float get_target();
  float get_gamma();
  bool get_done();

  float get_reward();
};

#endif //INCLUDE_ENVIRONMENTS_PROTO_PREDICTION_ENVIRONMENTS_H_
