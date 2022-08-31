#include <cmath>
#include <memory>
#include "Volume.h"

#pragma once

class Layer{
	int layer_num; // current layer number(index)
	int batchsize;
	int num_of_features;
	int padded_imgsize;
	
    public:
        typedef std::shared_ptr<Volume[]> ArrOfVols;
	
    ArrOfVols Aofind, Aof1d, Aof2d, Aofinu, Aof1u, Aof2u, Aof_final;	// Aof1d[num of images in the batch](num of features, x, y)
	ArrOfVols Aoeind, Aoe1d, Aoe2d, Aoeinu,	Aoe1u, Aoe2u, Aoe_final;	

	ArrOfVols Aok1d, 			Aok2d, 			Aok1u1,			Aok1u2, 		Aok2u, 			Aok_uc, 		Aok_final; 			// Aok1d[num of kernels / ouput channels](depth / number of input features, x, y)
	ArrOfVols Aok1d_gradient, 	Aok2d_gradient,	Aok1u1_gradient,Aok1u2_gradient,Aok2u_gradient,	Aok_uc_gradient,Aok_final_gradient;
	ArrOfVols Aok1d_back, 		Aok2d_back, 	Aok1u1_back,	Aok1u2_back, 	Aok2u_back, 					Aok_final_back;

    public:
        Layer()=default;
        Layer(int n, int input_imgsize, int batchsize);
};

