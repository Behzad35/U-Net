#include "unetFuncs.h"

int batchsize;
int input_imgsize;

void relu(Layer::ArrOfVols &input){
	int num_of_features = input[0].d;
	int imgsize = input[0].w; // includes padding (doesnt matter but just in case)
	for (int b=0; b<batchsize; ++b){
		for(int i = 0; i < num_of_features; ++i){
			for(int x = 1; x < imgsize-1; ++x){
				for(int y = 1; y < imgsize-1; ++y){
					if(input[b](i,x,y) < 0) input[b](i,x,y) = 0;
				}
			}
		}
	}
}
void conv(Layer::ArrOfVols const &input, Layer::ArrOfVols const &kernel, Layer::ArrOfVols &output){
	int num_of_kernels = output[0].d;
	int depth_of_kernels = input[0].d;
	int imgsize = input[0].w; // imgsize includes padding (both input and output are padded)
	int width_of_kernels = kernel[0].w;

	for (int b=0; b<batchsize; ++b){
		for(int i = 0; i < num_of_kernels; ++i){
			for(int x = 0; x < imgsize-2; ++x){
				for(int y = 0; y < imgsize-2; ++y){
					float tmp = 0;
					for(int j = 0; j < depth_of_kernels; ++j){							
						for(int n = 0; n < width_of_kernels; ++n){
							for(int m = 0; m < width_of_kernels; ++m){
								tmp += input[b](j,x+n,y+m) * kernel[i](j,n,m); // i-th kernel applied to j-th input is added to i-th output							
							}
						}
					}
					output[b](i,x+1,y+1) = tmp;
				}
			}
		}
	}
}

void fullconv(Layer::ArrOfVols const &input, Layer::ArrOfVols const &kernel, Layer::ArrOfVols &output){
	int num_of_kernels = output[0].d;
	int depth_of_kernels = input[0].d;
	int imgsize = input[0].w; // imgsize is not padded

	for (int b=0; b<batchsize; ++b){
		for(int i = 0; i < num_of_kernels; ++i){
			for(int x = 1; x < imgsize-1; ++x){
				for(int y = 1; y < imgsize-1; ++y){
					float tmp = 0;
					for(int j = 0; j < depth_of_kernels; ++j){
						tmp += input[b](j,x,y) * kernel[i](j,0,0); // i-th kernel applied to j-th input is added to i-th output	
					}
					output[b](i,x-1,y-1) = tmp;
				}
			}
		}
	}
}

void avgpool(Layer::ArrOfVols const &input, Layer::ArrOfVols &output){ 
	int num_of_features = input[0].d;
	int imgsize = input[0].w; // includes padding
	for (int b=0; b<batchsize; ++b){
		for (int i=0; i<num_of_features; ++i){
			for (int x=1; x<imgsize-1; x+=2){
				for(int y=1; y<imgsize-1; y+=2){
					output[b](i,x/2+1,y/2+1) = 0.25*(input[b](i,x,y) + input[b](i,x+1,y) + input[b](i,x,y+1)+ input[b](i,x+1,y+1));      
				}
			}
		}
	}
}

void avgpool_backward(Layer::ArrOfVols const &input, Layer::ArrOfVols &output){ 
	int num_of_features = input[0].d;
	int imgsize = input[0].w; // includes padding
	for (int b=0; b<batchsize; ++b){
		for (int i=0; i<num_of_features; ++i){
			for (int x=1; x<imgsize-1; ++x){
				for(int y=1; y<imgsize-1; ++y){
					output[b](i,2*x-1,2*y-1)= 0.25*(input[b](i,x,y));
					output[b](i,2*x,2*y-1) 	= 0.25*(input[b](i,x+1,y));
					output[b](i,2*x-1,2*y) 	= 0.25*(input[b](i,x,y+1));
					output[b](i,2*x,2*y) 	= 0.25*(input[b](i,x+1,y+1));      
				}
			}
		}
	}
}

void upconv(Layer::ArrOfVols const &input, Layer::ArrOfVols const &kernel, Layer::ArrOfVols &output){
	int num_of_kernels = output[0].d; //depth of output or number of kernels should be half of depth of input
	int depth_of_kernels = input[0].d;
	int imgsize = input[0].w; // includes padding
	for (int b=0; b<batchsize; ++b){
		for (int i=0; i<num_of_kernels; ++i){
			for (int x=1; x<imgsize-1; ++x){
				for (int y=1; y<imgsize-1; ++y){
					for (int j=0; j<depth_of_kernels; ++j){
						output[b](i,2*x-1,2*y-1)	= kernel[i](j,0,0) * input[b](j,x,y);
						output[b](i,2*x,2*y-1)		= kernel[i](j,1,0) * input[b](j,x,y);
						output[b](i,2*x-1,2*y)		= kernel[i](j,0,1) * input[b](j,x,y);
						output[b](i,2*x,2*y)		= kernel[i](j,1,1) * input[b](j,x,y);
					}
				}
			}
		}
	}
}

void concat(Layer::ArrOfVols const &input1, Layer::ArrOfVols const &input2, Layer::ArrOfVols &output){
	int num_of_features = input1[0].d;
	int imgsize = input1[0].w; // includes padding
	for (int b=0; b<batchsize; ++b){
		for (int i=0; i<num_of_features; ++i){
			for (int x=0; x<imgsize; ++x){
				for (int y=0; y<imgsize; ++y){
					output[b](i,x,y) = input1[b](i,x,y);
					output[b](i+num_of_features,x,y) = input2[b](i,x,y);
				}
			}
		}
	}
}

void create_Aok_backward(Layer &layer){
	int num_of_features = layer.num_of_features;

	for (int j=0; j<num_of_features/2; ++j){ // depth of forward kernels eg. 64
		for (int i=0; i<num_of_features; ++i){ 	// num of forward kernels eg. 128
			for (int x=0; x<3; ++x){
				for (int y=0; y<3; ++y){
					layer.Aok1d_back[j](i, x, y) = layer.Aok1d[i](j, x, y);
				}
			}
		}
	}

	for (int j=0; j<num_of_features; ++j){ 	// depth of forward kernels
		for (int i=0; i<num_of_features; ++i){ 	// num of forward kernels
			for (int x=0; x<3; ++x){
				for (int y=0; y<3; ++y){
					layer.Aok2d_back[j](i, x, y) = layer.Aok2d[i](j, x, y);
					layer.Aok1u_back[j](i, x, y) = layer.Aok1u[i](j, x, y);
					layer.Aok2u_back[j](i, x, y) = layer.Aok2u[i](j, x, y);
				}
			}
		}
	}
}

void create_Aok_backward(Layer::ArrOfVols const &Aok, Layer::ArrOfVols &Aok_back){
	for (int j=0; j<Aok[0].d; ++j){ 	// depth of forward kernels
		for (int i=0; i<Aok_back[0].d; ++i){ 	// num of forward kernels
			for (int x=0; x<Aok[0].w; ++x){
				for (int y=0; y<Aok[0].w; ++y){
					Aok_back[j](i, x, y) = Aok[i](j, x, y);
				}
			}
		}
	}
}

void create_all_Aok_backward(Layer *layers, int num_of_layers){
	//fist layer
	create_Aok_backward(layers[0].Aok_final, layers[0].Aok_final_back);

	for (int i=0; i<num_of_layers-1; ++i){
		create_Aok_backward(layers[i].Aok2u, layers[i].Aok2u_back);
		create_Aok_backward(layers[i].Aok1u, layers[i].Aok1u_back);
		create_Aok_backward(layers[i].Aok2d, layers[i].Aok2d_back);
		create_Aok_backward(layers[i].Aok1d, layers[i].Aok1d_back);
	}

	// last layer
	create_Aok_backward(layers[num_of_layers-1].Aok2u, layers[num_of_layers-1].Aok2u_back);
	create_Aok_backward(layers[num_of_layers-1].Aok1u, layers[num_of_layers-1].Aok1u_back);
}

void compute_Aoloss(Layer::ArrOfVols &Aoloss, Layer::ArrOfVols const &Aof_final, Layer::ArrOfVols const &Ao_annots){
	for (int b=0; b<batchsize; ++b){
		for(int x = 1; x < Aof_final[0].w-1; ++x){
			for(int y = 1; y < Aof_final[0].w-1; ++y){
				float sum;
				for(int i = 0; i < 3; ++i){ // (class indices - 1)
					sum += exp(Aof_final[b](i,x,y));
				}
				Aoloss[b](0,x-1,y-1) = -log(exp(Aof_final[b](Ao_annots[b](0,x-1,y-1)-1, x, y))/sum);
			}
		}
	}
}

Layer::ArrOfVols create_ArrOfVols(int num_of_arrs, int depth, int width){
	Layer::ArrOfVols output(new Volume[num_of_arrs]);
	for (int i=0; i<num_of_arrs; ++i){
		output[i]=Volume(depth, width);
	}
	return output;
}

//each input image with a error tensor will generate several gradient kernels. (the number depends on the depth of the error tensor)
//so for each batch, there will be (batchsize * error tensor depth) gradient kernels. 

void compute_Aok_gradient(Layer::ArrOfVols const &input, Layer::ArrOfVols const &error_tensor, Layer::ArrOfVols &Aok_gradient){
    for (int n = 0; n < error_tensor[0].d; ++n){ // the number of gradient kernels for each input = error_tensor[0].d
        for (int d = 0; d < input[0].d; ++d){ // the depth of each gradient kernel = input[0].d
            for (int h = 0; h < Aok_gradient[0].w; ++h){ 
                for (int w = 0; w < Aok_gradient[0].w; ++w){
                    float tmp = 0;
                    for (int b = 0; b < batchsize; ++b){
	                    for (int i = 1; i < error_tensor[0].w-1; ++i){
	                        for (int j = 1; j < error_tensor[0].w-1; ++j){
	                        	tmp += error_tensor[b](n,i,j) * input[b](d,i+w-1,j+h-1); //apply interior of error_tensor to padded input
	                        }
	                    }
	                }
                    Aok_gradient[n](d,w,h) = tmp/batchsize; // this is the "average" Aok gradient for the whole batch
                }
            }
        }
    }
}

void create_all_Aok_gradient(Layer *layers, int num_of_layers){
	// first layer has an additional kernel
	compute_Aok_gradient(layers[0].Aof2u, layers[0].Aoe_final, layers[0].Aok_final_gradient);

	for (int i=0; i<num_of_layers-1; ++i){
		compute_Aok_gradient(layers[i].Aof1u, layers[i].Aoe2u, layers[i].Aok2u_gradient);
		compute_Aok_gradient(layers[i].Aof_concat, layers[i].Aoe1u, layers[i].Aok1u_gradient);
		compute_Aok_gradient(layers[i].Aof1d, layers[i].Aoe2d, layers[i].Aok2d_gradient);
		compute_Aok_gradient(layers[i].Aofind, layers[i].Aoe1d, layers[i].Aok1d_gradient);
	}

	// last layer
	compute_Aok_gradient(layers[num_of_layers-1].Aof1d, layers[num_of_layers-1].Aoe2d, layers[num_of_layers-1].Aok2d_gradient);
	compute_Aok_gradient(layers[num_of_layers-1].Aofind, layers[num_of_layers-1].Aoe1d, layers[num_of_layers-1].Aok1d_gradient);

}
