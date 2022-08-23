#ifndef LAYER_H
#define LAYER_H
#include <cmath>
#include <memory>
#include "Volume.h"


struct Layer{
	Layer()=default;
	Layer(int n, int input_imgsize, int batchsize);
	int layer_num; // current layer number(index)
	int batchsize;
	int num_of_features;
	int padded_imgsize;

	typedef std::shared_ptr<Volume[]> ArrOfVols;
	
	ArrOfVols Aofind, Aof1d, Aof2d, Aofinu, Aof1u, Aof2u, Aof_final;	// Aof1d[num of images in the batch](num of features, x, y)
	ArrOfVols Aoeind, Aoe1d, Aoe2d, Aoeinu,	Aoe1u, Aoe2u, Aoe_final;	

	ArrOfVols Aok1d, 			Aok2d, 			Aok1u1,			Aok1u2, 		Aok2u, 			Aok_uc, 		Aok_final; 			// Aok1d[num of kernels / ouput channels](depth / number of input features, x, y)
	ArrOfVols Aok1d_gradient, 	Aok2d_gradient,	Aok1u1_gradient,Aok1u2_gradient,Aok2u_gradient,	Aok_uc_gradient,Aok_final_gradient;
	ArrOfVols Aok1d_back, 		Aok2d_back, 	Aok1u1_back,	Aok1u2_back, 	Aok2u_back, 					Aok_final_back;

};

Layer::Layer(int n, int input_imgsize, int batchsize) : 
	layer_num(n),
	batchsize(batchsize),
	num_of_features(pow(2,6+layer_num)),
	padded_imgsize(input_imgsize/pow(2,layer_num)+2), // includes padding

	// ---------------------------- array of feature channels --------------------
	Aofind(new Volume[batchsize]),
	Aof1d(new Volume[batchsize]), 
	Aof2d(new Volume[batchsize]), 
	Aofinu(new Volume[batchsize]),
	Aof1u(new Volume[batchsize]),  
	Aof2u(new Volume[batchsize]),
	Aof_final(new Volume[batchsize]),

	// ---------------------------- array of error tensors --------------------
	Aoeind(new Volume[batchsize]),
	Aoe1d(new Volume[batchsize]), 
	Aoe2d(new Volume[batchsize]), 
	Aoeinu(new Volume[batchsize]),
	Aoe1u(new Volume[batchsize]),  
	Aoe2u(new Volume[batchsize]),
	Aoe_final(new Volume[batchsize]),

	// ---------------------------- array of conv kernels --------------------
	Aok1d(new Volume[num_of_features]),
	Aok2d(new Volume[num_of_features]),
	Aok1u1(new Volume[num_of_features]),
	Aok1u2(new Volume[num_of_features]),
	Aok2u(new Volume[num_of_features/2]),
	Aok_uc(new Volume[num_of_features/2]),
	Aok_final(new Volume[3]), // 3 kernels for 3 outputs

	// ---------------------------- array of conv kernel gradients (avg for whole batch) --------------------
	Aok1d_gradient(new Volume[num_of_features]),
	Aok2d_gradient(new Volume[num_of_features]),
	Aok1u1_gradient(new Volume[num_of_features]),
	Aok1u2_gradient(new Volume[num_of_features]),
	Aok2u_gradient(new Volume[num_of_features/2]),
	Aok_uc_gradient(new Volume[num_of_features/2]),
	Aok_final_gradient(new Volume[3]),

	// ---------------------------- array of conv kernels in backward pass for finding error tensors --------------------
	Aok1d_back(new Volume[num_of_features/2]),
	Aok2d_back(new Volume[num_of_features]),
	Aok1u1_back(new Volume[num_of_features]),
	Aok1u2_back(new Volume[num_of_features]),
	Aok2u_back(new Volume[num_of_features]),
	Aok_final_back(new Volume[num_of_features])
	{
		for (int i=0; i<batchsize; ++i){
			Aofind[i]=Volume(num_of_features/2, padded_imgsize);
			Aof1d[i]=Volume(num_of_features, padded_imgsize);
			Aof2d[i]=Volume(num_of_features, padded_imgsize);
			Aofinu[i]=Volume(num_of_features, padded_imgsize);
			Aof1u[i]=Volume(num_of_features, padded_imgsize);
			Aof2u[i]=Volume(num_of_features/2, padded_imgsize);
			Aof_final[i]=Volume(3, padded_imgsize);	// Final Array of features (3 features) (output segmentation map) (not padded, so can be directly compared with Annot)

			Aoeind[i]=Volume(num_of_features/2, padded_imgsize);
			Aoe1d[i]=Volume(num_of_features, padded_imgsize);
			Aoe2d[i]=Volume(num_of_features, padded_imgsize);
			Aoeinu[i]=Volume(num_of_features, padded_imgsize);
			Aoe1u[i]=Volume(num_of_features, padded_imgsize);
			Aoe2u[i]=Volume(num_of_features/2, padded_imgsize);
			Aoe_final[i]=Volume(3, padded_imgsize);
		}
		for (int i=0; i<num_of_features; ++i){

			Aok1d[i]=Volume(num_of_features/2, 3); 	// depth of first kernel = num of features in Aofind
			Aok2d[i]=Volume(num_of_features, 3);
			Aok1u1[i]=Volume(num_of_features, 3);
			Aok1u2[i]=Volume(num_of_features, 3);

			Aok1d_gradient[i]=Volume(num_of_features/2, 3); 	// depth of first kernel = num of features in Aofind
			Aok2d_gradient[i]=Volume(num_of_features, 3);
			Aok1u1_gradient[i]=Volume(num_of_features, 3);
			Aok1u2_gradient[i]=Volume(num_of_features, 3);
			
			Aok2d_back[i]=Volume(num_of_features, 3);
			Aok1u1_back[i]=Volume(num_of_features, 3);
			Aok1u2_back[i]=Volume(num_of_features, 3);
			Aok2u_back[i]=Volume(num_of_features/2, 3);

			Aok_final_back[i]=Volume(3, 1);
		}
		for (int i=0; i<3; ++i){
			Aok_final[i]=Volume(num_of_features, 1);
			Aok_final_gradient[i]=Volume(num_of_features, 1);
		}
		for (int i=0; i<num_of_features/2; ++i){
			Aok2u[i]=Volume(num_of_features, 3);
			Aok2u_gradient[i]=Volume(num_of_features, 3);
			Aok_uc[i]=Volume(num_of_features/2, 2);
			Aok_uc_gradient[i]=Volume(num_of_features/2, 2);

			Aok1d_back[i]=Volume(num_of_features, 3);
		}
	}

#endif