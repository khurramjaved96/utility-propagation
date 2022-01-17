//
// Created by Khurram Javed on 2022-01-16.
//


#include "../../include/nn/neuron.h"
#include <assert.h>
#include <iostream>
#include <utility>
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include "../../include/utils.h"
#include "../../include/nn/utils.h"


LSTM::LSTM() : Neuron(false, false) {
  Hu_i = Hu_f = Hu_g = Hu_o = 0;
  Cu_i = Cu_f = Cu_g = Cu_o = 0;
  Hb_i = Hb_f = Hb_g = Hb_o = 0;
  Cb_i = Cb_f = Cb_g = Cb_o = 0;
//  Gradients
  Gb_i = Gb_f = Gb_g = Gb_o = 0;
  Gu_i = Gu_f = Gu_g = Gu_o = 0;

//  Order: i|f|g|o
  u_i =  0.3632;
  u_f =  0.8304;
  u_g = -0.2058;
  u_o =  0.7483;

  b_i = -0.1612;
  b_f = 0.1058;
  b_g = 0.9055;
  b_o = -0.9277;
}

float LSTM::forward(float temp_value) {
  return temp_value;
}

float LSTM::backward(float output_grad) {
  return 1;
}

void LSTM::print_gradients() {
  std::cout << "W_i gradients\n";
  for(int i = 0; i < this->w_i.size(); i++){
    std::cout << Gw_i[i] << ", ";
  }
  std::cout << "\n";

  std::cout << "W_f gradients\n";
  for(int i = 0; i < this->w_f.size(); i++){
    std::cout << Gw_f[i] << ", ";
  }
  std::cout << "\n";

  std::cout << "W_g gradients\n";
  for(int i = 0; i < this->w_g.size(); i++){
    std::cout << Gw_g[i] << ", ";
  }
  std::cout << "\n";

  std::cout << "W_o gradients\n";
  for(int i = 0; i < this->w_g.size(); i++){
    std::cout << Gw_o[i] << ", ";
  }
  std::cout << "\n";

  std::cout << "Bias \n";
  //  Order: i|f|g|o
  std::cout << Gb_i << " " << Gb_f << " " << Gb_g << " " << Gb_o << std::endl;

  std::cout << "Recurrent connection \n";
  //  Order: i|f|g|o
  std::cout << Gu_i << " " << Gu_f << " " << Gu_g << " " << Gu_o << std::endl;
 }

void LSTM::fire() {
  this->neuron_age++;
  this->old_value = this->value;
  this->value = this->forward(value_before_firing);
}

void LSTM::compute_gradient_of_all_synapses(std::vector<float> prediction_error_list) {
  std::vector<float> x;
  for(auto &it: this->incoming_synapses){
    x.push_back(it->input_neuron->value);
  }

  int n = this->incoming_synapses.size();

//  Computing terms that are reused to save flops

  float LHS_g = (1-(g*g));
  float LHS_f = f*(1-f);
  float LHS_i = i_val*(1-i_val);
  float LHS_o = o*(1-o);
  float tanh_of_c = tanh(c);

  
//  Section 1.1

  for(int i = 0; i < n; i++){
    float dg_dWi =  LHS_g * (u_g*Hw_i[i]);
    float df_dWi = LHS_f * (u_f*Hw_i[i]);
    float di_dWi = LHS_i * (x[i] + u_i*Hw_i[i]);
    float do_dWi = LHS_o * (u_o*Hw_i[i]);
    Cw_i[i] = f * Cw_i[i] + old_c * df_dWi + i_val * dg_dWi + g*di_dWi;
    Hw_i[i] = o * (1-tanh_of_c*tanh_of_c)* Cw_i[i] +  tanh_of_c*do_dWi;
  }

//  Section 1.4
  for(int i = 0; i < n; i++){
    float dg_dWF =  LHS_g * (u_g*Hw_f[i]);
    float df_dWF = LHS_f * (x[i] + u_f*Hw_f[i]);
    float di_dWF = LHS_i * (u_i*Hw_f[i]);
    float do_dWF = LHS_o * (u_o*Hw_f[i]);
    Cw_f[i] = f * Cw_f[i] + old_c * df_dWF + i_val * dg_dWF + g*di_dWF;
    Hw_f[i] = o * (1-tanh_of_c*tanh_of_c)* Cw_f[i] +  tanh_of_c*do_dWF;
  }

//  Section 1.5
  for(int i = 0; i < n; i++){
    float dg_dWo =  LHS_g * (u_g*Hw_o[i]);
    float df_dWo = LHS_f * (u_f*Hw_o[i]);
    float di_dWo = LHS_i * (u_i*Hw_o[i]);
    float do_dWo = LHS_o * (x[i] + u_o*Hw_o[i]);
    Cw_o[i] = f * Cw_o[i] + old_c * df_dWo + i_val * dg_dWo + g*di_dWo;
    Hw_o[i] = o * (1-tanh_of_c*tanh_of_c)* Cw_o[i] +  tanh_of_c*do_dWo;
  }

//  Section 1.6
  for(int i = 0; i < n; i++){
    float dg_dWg = LHS_g * (x[i] + u_g*Hw_g[i]);
    float df_dWg = LHS_f * (u_f*Hw_g[i]);
    float di_dWg = LHS_i * (u_i*Hw_g[i]);
    float do_dWg = LHS_o * (u_o*Hw_g[i]);
    Cw_g[i] = f * Cw_g[i] + old_c * df_dWg + i_val * dg_dWg + g*di_dWg;
    Hw_g[i] = o * (1-tanh_of_c*tanh_of_c)* Cw_g[i] +  tanh_of_c*do_dWg;
  }


//  Section 1.11
  {
    float dg_dbf = LHS_g * (u_g * Hb_f);
    float df_dbf = LHS_f * (1 + u_f * Hb_f);
    float di_dbf = LHS_i * (u_i * Hb_f);
    float do_dbf = LHS_o * (u_o * Hb_f);
    Cb_f = f * Cb_f + old_c * df_dbf + i_val * dg_dbf + g * di_dbf;
    Hb_f = o * (1 - tanh_of_c * tanh_of_c) * Cb_f + tanh_of_c * do_dbf;
  }

//  Section 1.10
  {
    float dg_dbg = LHS_g * (1 + u_g * Hb_g);
    float df_dbg = LHS_f * (u_f * Hb_g);
    float di_dbg = LHS_i * (u_i * Hb_g);
    float do_dbg = LHS_o * (u_o * Hb_g);
    Cb_g = f * Cb_g + old_c * df_dbg + i_val * dg_dbg + g * di_dbg;
    Hb_g = o * (1 - tanh_of_c * tanh_of_c) * Cb_g + tanh_of_c * do_dbg;
  }

// Section 1.12
  {
    float dg_dbo = LHS_g * (u_g * Hb_o);
    float df_dbo = LHS_f * (u_f * Hb_o);
    float di_dbo = LHS_i * (u_i * Hb_o);
    float do_dbo = LHS_o * (1 + u_o * Hb_o);
    Cb_o = f * Cb_o + old_c * df_dbo + i_val * dg_dbo + g * di_dbo;
    Hb_o = o * (1 - tanh_of_c * tanh_of_c) * Cb_o + tanh_of_c * do_dbo;
  }

//  Section 1.3
  {
    float dg_dbi = LHS_g * (u_g * Hb_i);
    float df_dbi = LHS_f * (u_f * Hb_i);
    float di_dbi = LHS_i * (1 + u_i * Hb_i);
    float do_dbi = LHS_o * (u_o * Hb_i);
    Cb_i = f * Cb_i + old_c * df_dbi + i_val * dg_dbi + g * di_dbi;
    Hb_i = o * (1 - tanh_of_c * tanh_of_c) * Cb_i + tanh_of_c * do_dbi;
  }


//  Section 1.8
  {
    float dg_duf = LHS_g * (u_g * Hu_f);
    float df_duf = LHS_f * (old_h + u_f * Hu_f);
    float di_duf = LHS_i * (u_i * Hu_f);
    float do_duf = LHS_o * (u_o * Hu_f);
    Cu_f = f * Cu_f + old_c * df_duf + i_val * dg_duf + g * di_duf;
    Hu_f = o * (1 - tanh_of_c * tanh_of_c) * Cu_f + tanh_of_c * do_duf;
  }

//  Section 1.7
  {
    float dg_duo = LHS_g * (u_g * Hu_o);
    float df_duo = LHS_f * (u_f * Hu_o);
    float di_duo = LHS_i * (u_i * Hu_o);
    float do_duo = LHS_o * (old_h + u_o * Hu_o);
    Cu_o = f * Cu_o + old_c * df_duo + i_val * dg_duo + g * di_duo;
    Hu_o = o * (1 - tanh_of_c * tanh_of_c) * Cu_o + tanh_of_c * do_duo;
  }


//  Section 1.9
  {
    float dg_dug = LHS_g * (old_h + u_g * Hu_g);
    float df_dug = LHS_f * (u_f * Hu_g);
    float di_dug = LHS_i * (u_i * Hu_g);
    float do_dug = LHS_o * (u_o * Hu_g);
    Cu_g = f * Cu_g + old_c * df_dug + i_val * dg_dug + g * di_dug;
    Hu_g = o * (1 - tanh_of_c * tanh_of_c) * Cu_g + tanh_of_c * do_dug;
  }


//  Section 1.2
  {
    float dg_dui = LHS_g * (u_g * Hu_i);
    float df_dui = LHS_f * (u_f * Hu_i);
    float di_dui = LHS_i * (old_h + u_i * Hu_i);
    float do_dui = LHS_o * (u_o * Hu_i);
    Cu_i = f * Cu_i + old_c * df_dui + i_val * dg_dui + g * di_dui;
    Hu_i = o * (1 - tanh_of_c * tanh_of_c) * Cu_i + tanh_of_c * do_dui;
  }
}

void LSTM::accumulate_gradient() {
  int n = this->w_f.size();
  for(int counter = 0; counter < n; counter++) {
    Gw_f[counter] += Hw_f[counter];
    Gw_i[counter] += Hw_i[counter];
    Gw_o[counter] += Hw_o[counter];
    Gw_g[counter] += Hw_g[counter];
  }
  Gb_f += Hb_f;
  Gb_i += Hb_i;
  Gb_o += Hb_o;
  Gb_g += Hb_g;
  Gu_f += Hu_f;
  Gu_i += Hu_i;
  Gu_o += Hu_o;
  Gu_g += Hu_g;
};

void LSTM::update_value() {
  std::cout << "H = " << h << std::endl;
  std::cout << "C = " << c << std::endl;
  this->value_before_firing = 0;
  old_h = h;
  old_c = c;
//  Adding bias value
  i_val = b_i;
  g = b_g;
  f = b_f;
  o = b_o;

  int counter = 0;
//  Non-bias terms connections except the self connection
//  for (auto &it: this->incoming_synapses) {
//    std::cout << it->input_neuron->value << std::endl;
//  }
//  exit(0);
  for (auto &it: this->incoming_synapses) {
//    std::cout << w_i[counter] << "*" << it->input_neuron->value << std::endl;
    i_val += this->w_i[counter] * it->input_neuron->value;
    f += this->w_f[counter] * it->input_neuron->value;
    g += this->w_g[counter] * it->input_neuron->value;
    o += this->w_o[counter] * it->input_neuron->value;
    counter++;
  }

//  Self connection
  i_val += u_i * old_h;
  g += u_g * old_h;
  f += u_f * old_h;
  o += u_o * old_h;

//  Applying non-linear transformations and updating the hidden state;
  i_val = sigmoid(i_val);
  f = sigmoid(f);
  o = sigmoid(o);
  g = tanh(g);
  c = f * old_c + i_val * g;
  h = o * tanh(c);
  this->value_before_firing = h;

}

void LSTM::add_synapse(Synapse *s, float w_i, float w_f, float w_g, float w_o) {
  this->incoming_synapses.push_back(s);
  this->w_i.push_back(w_i);
  this->w_f.push_back(w_f);
  this->w_g.push_back(w_g);
  this->w_o.push_back(w_o);

  this->Hw_i.push_back(0);
  this->Hw_f.push_back(0);
  this->Hw_g.push_back(0);
  this->Hw_o.push_back(0);

  this->Cw_i.push_back(0);
  this->Cw_f.push_back(0);
  this->Cw_g.push_back(0);
  this->Cw_o.push_back(0);

  this->Gw_i.push_back(0);
  this->Gw_f.push_back(0);
  this->Gw_g.push_back(0);
  this->Gw_o.push_back(0);

}