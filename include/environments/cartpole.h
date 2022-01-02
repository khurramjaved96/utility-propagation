//
// Created by Khurram Javed on 2021-10-06.
//

#ifndef INCLUDE_ENVIRONMENTS_CARTPOLE_H_
#define INCLUDE_ENVIRONMENTS_CARTPOLE_H_

#include <vector>
#include <random>

class CartPole {
  std::mt19937 mt;
  std::vector<float> state;
  float gravity, masscart, masspole, total_mass, length,
  polemass_length, force_mag, tau, x_threshold,
    theta_threshold_radians;
  int total_actions;
 public:
  CartPole();
  std::vector<float> step(int action);
  std::vector<float> get_state();
  std::vector<float> reset();
  void seed(int seed);
  int get_no_of_actions();
};
#endif  // INCLUDE_ENVIRONMENTS_CARTPOLE_H_
