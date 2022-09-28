#include "ConvStruct.h"

ConvStruct::ConvStruct(int num_of_inputs, int num_of_outputs, int batchsize, int padded_imgsize, int kernel_size) : 
	in(num_of_inputs), 
	out(num_of_outputs),
	batchsize(batchsize),
	padded_imgsize(padded_imgsize),
	kernel_size(kernel_size),
	Aof(new Volume[batchsize]),
	Aoe(new Volume[batchsize]),
	Aok(new Volume[out]),
	Aom_adam(new Volume[out]),
	Aov_adam(new Volume[out]),
	Aok_gradient(new Volume[out]),
	Aok_back(new Volume[in])
	{
		for (int i=0; i<batchsize; ++i){
			Aof[i]=Volume(in, padded_imgsize);
			Aoe[i]=Volume(in, padded_imgsize);
		}
		for (int i=0; i<out; ++i){
			Aok[i]=Volume(in, kernel_size);
			Aok_gradient[i]=Volume(in, kernel_size);
			Aom_adam[i]=Volume(in, kernel_size);
			Aov_adam[i]=Volume(in, kernel_size);
		}
		for (int i=0; i<in; ++i){
			Aok_back[i]=Volume(out, kernel_size);
		}
	}
