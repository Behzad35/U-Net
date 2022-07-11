#ifndef LAYER_H
#define LAYER_H
#include <cmath>
#include <memory>
#include "Volume.h"

struct Layer{
	Layer()=default;
	Layer(int n, int input_imgsize, int batchsize) : 
		layer_num(n),
		batchsize(batchsize),
		num_of_features(pow(2,6+layer_num)),
		imgsize(input_imgsize/pow(2,layer_num)+2), // includes padding

		Aofind(new Volume[batchsize]),
		Aof1d(new Volume[batchsize]), 
		Aof2d(new Volume[batchsize]), 
		Aofinu(new Volume[batchsize]),
		Aof1u(new Volume[batchsize]),  
		Aof2u(new Volume[batchsize]),

		Aok1d(new Volume[num_of_features]),
		Aok2d(new Volume[num_of_features]),
		Aok1u(new Volume[num_of_features]),
		Aok2u(new Volume[num_of_features]),
		Aok_uc(new Volume[num_of_features/2])
		{
			for (int i=0; i<batchsize; ++i){
				Aofind[i]=Volume(num_of_features/2, imgsize);
				Aof1d[i]=Volume(num_of_features, imgsize);
				Aof2d[i]=Volume(num_of_features, imgsize);
				Aofinu[i]=Volume(num_of_features, imgsize);
				Aof1u[i]=Volume(num_of_features, imgsize);
				Aof2u[i]=Volume(num_of_features, imgsize);
			}
			for (int i=0; i<num_of_features; ++i){

				Aok1d[i]=Volume(num_of_features/2, 3);
				Aok2d[i]=Volume(num_of_features, 3);
				Aok1u[i]=Volume(num_of_features, 3);
				Aok2u[i]=Volume(num_of_features, 3);
			}
			for (int i=0; i<num_of_features/2; ++i){
				Aok_uc[i]=Volume(num_of_features, 2);
			}
		}

	int layer_num; // current layer number(index)
	int batchsize;
	int num_of_features;
	int imgsize;

	typedef std::shared_ptr<Volume[]> ArrOfVols;
	
	ArrOfVols Aofind, Aof1d, Aof2d, Aofinu, Aof1u, Aof2u;	// Aof1d[num of images in the batch].(num of features, x, y)
	ArrOfVols Aok1d, Aok2d, Aok1u, Aok2u, Aok_uc; 			// Aok1d[num of kernels / ouput channels].(depth / number of input features, x, y)

};

#endif
