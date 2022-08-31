#include <cmath>
#include <memory>
#include "Volume.h"
#include "Layer.h"


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

