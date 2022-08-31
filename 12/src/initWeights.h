#include <algorithm>
#include <vector>
#include <string>
#include <array>
#include <iostream>
#include <complex>
#include "Layer.h"
#include "npy.hpp"


template <typename Scalar> struct npy_data {
  std::vector<unsigned long> shape;
  bool fortran_order;
  std::vector<Scalar> data;
};

template <typename Scalar> npy_data<Scalar> test_load(const char * path); 


  
void readKernel(Layer::ArrOfVols const &target, const char * path, int dim[]);

void init_weights(Layer * layers);

void init_biases(Layer * layers);

void init_kernels(Layer * layers);

