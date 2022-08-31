#include "unetFuncs.h"

int batchsize;
int input_imgsize;
float learning_rate;

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
void conv(Layer::ArrOfVols const &input, Layer::ArrOfVols const &kernel, Layer::ArrOfVols &output, bool dont_wipe_before_adding){
	int num_of_kernels = output[0].d;
	int depth_of_kernels = input[0].d;
	int imgsize = input[0].w; // imgsize includes padding (both input and output are padded)
	int width_of_kernels = kernel[0].w;

	if (dont_wipe_before_adding){
		for (int b=0; b<batchsize; ++b){
			for(int i = 0; i < num_of_kernels; ++i){	// just add to old output
				for(int x = 0; x < imgsize-2; ++x){
					for(int y = 0; y < imgsize-2; ++y){
						for(int j = 0; j < depth_of_kernels; ++j){							
							for(int n = 0; n < width_of_kernels; ++n){
								for(int m = 0; m < width_of_kernels; ++m){
									output[b](i,x+1,y+1) += input[b](j,x+n,y+m) * kernel[i](j,n,m); // i-th kernel applied to j-th input is added to i-th output
								}
							}
						}
					}
				}
			}
		}
	}
	else{
		for (int b=0; b<batchsize; ++b){
			for(int i = 0; i < num_of_kernels; ++i){	// zero out old output before adding
				std::cout<< i<< std::endl;
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

void avgpool_backward(Layer::ArrOfVols const &input, Layer::ArrOfVols &output){ // adds new error to the one found from concat
	int num_of_features = input[0].d;
	int imgsize = input[0].w; // includes padding
	for (int b=0; b<batchsize; ++b){
		for (int i=0; i<num_of_features; ++i){
			for (int x=1; x<imgsize-1; ++x){
				for(int y=1; y<imgsize-1; ++y){
					output[b](i,2*x-1,2*y-1)+= 0.25*(input[b](i,x,y));
					output[b](i,2*x,2*y-1) 	+= 0.25*(input[b](i,x+1,y));
					output[b](i,2*x-1,2*y) 	+= 0.25*(input[b](i,x,y+1));
					output[b](i,2*x,2*y) 	+= 0.25*(input[b](i,x+1,y+1));      
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

void upconv_backward(Layer::ArrOfVols const &input, Layer::ArrOfVols const &kernel, Layer::ArrOfVols &output){
   
    for(int i=0; i<output[0].d; ++i){
        for(int x=1; x<output[0].w-1; ++x){
            for(int y=1; y<output[0].w-1; ++y){
                for(int b=0; b<batchsize; ++b){
                    for(int j=0; j<input[0].d; ++j){
                        output[b](i,x,y) = kernel[i](j,0,0) * input[b](j, 2*x-1, 2*y-1) +
                                           kernel[i](j,1,0) * input[b](j, 2*x, 2*y-1) +
                                           kernel[i](j,0,1) * input[b](j, 2*x-1, 2*y) +
                                           kernel[i](j,1,1) * input[b](j, 2*x, 2*y);
                    }
                }
            }
        }
    }
}

void create_Aok_backward(Layer::ArrOfVols const &Aok, Layer::ArrOfVols &Aok_back){ // create conv kernel for finding error tensors in the backpass
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
		create_Aok_backward(layers[i].Aok1u1, layers[i].Aok1u1_back);
		create_Aok_backward(layers[i].Aok1u2, layers[i].Aok1u2_back);
		create_Aok_backward(layers[i].Aok2d, layers[i].Aok2d_back);
		create_Aok_backward(layers[i].Aok1d, layers[i].Aok1d_back);
	}

	// last layer
	create_Aok_backward(layers[num_of_layers-1].Aok2u, layers[num_of_layers-1].Aok2u_back);
	create_Aok_backward(layers[num_of_layers-1].Aok1d, layers[num_of_layers-1].Aok1d_back);
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

void compute_Aok_gradient(Layer::ArrOfVols const &input, Layer::ArrOfVols const &old_error_tensor, Layer::ArrOfVols &Aok_gradient){
    for (int n = 0; n < old_error_tensor[0].d; ++n){ // the number of gradient kernels for each input = old_error_tensor[0].d
        for (int d = 0; d < input[0].d; ++d){ // the depth of each gradient kernel = input[0].d
            for (int h = 0; h < Aok_gradient[0].w; ++h){ 
                for (int w = 0; w < Aok_gradient[0].w; ++w){
                    float tmp = 0;
                    for (int b = 0; b < batchsize; ++b){
	                    for (int x = 1; x < old_error_tensor[0].w-1; ++x){
	                        for (int y = 1; y < old_error_tensor[0].w-1; ++y){
	                        	tmp += old_error_tensor[b](n,x,y) * input[b](d,x+w-1,y+h-1); //apply interior of error_tensor to padded input
	                        }
	                    }
	                }
                    Aok_gradient[n](d,w,h) = tmp/batchsize; // this is the "average" Aok gradient for the whole batch
                }
            }
        }
    }
}

void compute_Aok_uc_gradient(Layer::ArrOfVols const &input, Layer::ArrOfVols const &old_error_tensor, Layer::ArrOfVols &Aok_gradient){
    for (int n = 0; n < old_error_tensor[0].d; ++n){ // the number of kernels
        for (int d = 0; d < input[0].d; ++d){ // the depth of each kernel = input[0].d
            float tmp1 = 0;
            float tmp2 = 0;
            float tmp3 = 0;
            float tmp4 = 0;
            for (int b = 0; b < batchsize; ++b){
                for (int x = 1; x < input[0].w-1; x+=2){
                    for (int y = 1; y < input[0].w-1; y+=2){
                    	tmp1 += old_error_tensor[b](n,2*x-1,2*y-1) * input[b](d,x,y); 
                    	tmp2 += old_error_tensor[b](n,2*x,2*y-1) * input[b](d,x+1,y);
                    	tmp3 += old_error_tensor[b](n,2*x-1,2*y) * input[b](d,x,y+1);
                    	tmp4 += old_error_tensor[b](n,2*x,2*y) * input[b](d,x+1,y+1);
                    }
                }
            }
            Aok_gradient[n](d,0,0) = tmp1/batchsize; // this is the "average" Aok gradient for the whole batch
			Aok_gradient[n](d,1,0) = tmp2/batchsize;
			Aok_gradient[n](d,0,1) = tmp3/batchsize;
			Aok_gradient[n](d,1,1) = tmp4/batchsize;        
        }
    }
}

void create_all_Aok_gradient(Layer *layers, int num_of_layers){
	
	compute_Aok_gradient(layers[0].Aof2u, layers[0].Aoe_final, layers[0].Aok_final_gradient);	// first layer has an additional kernel
	for (int i=0; i<num_of_layers-1; ++i){
		compute_Aok_gradient(layers[i].Aof1u, layers[i].Aoe2u, layers[i].Aok2u_gradient);
		compute_Aok_gradient(layers[i].Aofinu, layers[i].Aoe1u, layers[i].Aok1u1_gradient);
		compute_Aok_gradient(layers[i].Aof2d, layers[i].Aoe1u, layers[i].Aok1u2_gradient);
		compute_Aok_gradient(layers[i].Aof1d, layers[i].Aoe2d, layers[i].Aok2d_gradient);
		compute_Aok_gradient(layers[i].Aofind, layers[i].Aoe1d, layers[i].Aok1d_gradient);
		compute_Aok_uc_gradient(layers[i+1].Aof2u, layers[i].Aoeinu, layers[i+1].Aok_uc_gradient);
	}
	// last layer
	compute_Aok_gradient(layers[num_of_layers-1].Aof1d, layers[num_of_layers-1].Aoe2d, layers[num_of_layers-1].Aok2d_gradient);
	compute_Aok_gradient(layers[num_of_layers-1].Aofind, layers[num_of_layers-1].Aoe1d, layers[num_of_layers-1].Aok1d_gradient);

}

void compute_Aoe_final(Layer::ArrOfVols const &Aof_final, Layer::ArrOfVols &Aoe_final, const Layer::ArrOfVols &Ao_annots){
    for (int b = 0; b < batchsize; ++b){
    	for (int x = 1; x < Aof_final[0].w-1; ++x){
        	for (int y = 1; y < Aof_final[0].w-1; ++y){
                float sum = 0;
                for (int c = 0; c < Aof_final[0].d; ++c){
                    sum += exp(Aof_final[b](c,x,y));
                }
                for (int i=0; i<Aof_final[0].d; ++i){	// i = 0,1,2
                	if (i==Ao_annots[0](0,x-1,y-1)-1){ 	// Ao_annots = 1,2,3
                		Aoe_final[b](i, x, y) = exp(Aof_final[b](i, x, y)) / sum -1;
                	}
                	else{
                		Aoe_final[b](i, x, y) = exp(Aof_final[b](i, x, y)) / sum;
                	}
                }
            }
        }
    }
}

void update_Aok(Layer::ArrOfVols &kernel, Layer::ArrOfVols const &kernel_gradient){
	for (int b=0; b<batchsize; ++b){
		for (int d=0; d<kernel[0].d; ++d){
			for (int x=0; x<kernel[0].w; ++x){
				for (int y=0; y<kernel[0].w; ++y){
					kernel[b](d,x,y) = kernel[b](d,x,y) - learning_rate*kernel_gradient[b](d,x,y);
				}
			}
		}
	}
}

void update_all_Aok(Layer *layers, int num_of_layers){ // update all kernels from their gradient

	update_Aok(layers[0].Aok_final, layers[0].Aok_final_gradient);
	for (int i=0; i<num_of_layers-1; ++i){
		update_Aok(layers[i].Aok1d, layers[i].Aok1d_gradient);
		update_Aok(layers[i].Aok2d, layers[i].Aok2d_gradient);
		update_Aok(layers[i].Aok1u1, layers[i].Aok1u1_gradient);
		update_Aok(layers[i].Aok1u2, layers[i].Aok1u2_gradient);
		update_Aok(layers[i].Aok2u, layers[i].Aok2u_gradient);
		update_Aok(layers[i].Aok_uc, layers[i].Aok_uc_gradient);
	}
	update_Aok(layers[num_of_layers-1].Aok1d, layers[num_of_layers-1].Aok1d_gradient);
	update_Aok(layers[num_of_layers-1].Aok2u, layers[num_of_layers-1].Aok2u_gradient);
	update_Aok(layers[num_of_layers-1].Aok_uc, layers[num_of_layers-1].Aok_uc_gradient);


}

void forward_pass(Layer *layers, int num_of_layers){

	std::cout<<"================================== Forwared Pass ========================================"<< std::endl;

	for (int i=0; i<=num_of_layers-2; ++i){
		std::cout<<"---------- Layer ("<< i <<") ----------"<<std::endl;
		std::cout<<"conv 1"<<std::endl;
		conv(layers[i].Aofind, layers[i].Aok1d, layers[i].Aof1d);
		relu(layers[i].Aof1d);

		std::cout<<"conv 2"<<std::endl;
		conv(layers[i].Aof1d, layers[i].Aok2d, layers[i].Aof2d);
		relu(layers[i].Aof2d);

		std::cout<<"avgpool to layer "<< i+1 <<std::endl;
		avgpool(layers[i].Aof2d, layers[i+1].Aofind);

	}

	// lowest layer uses features (Aofind, Aof1d and Aof2u) and kernels (Aok1d, Aok2u and Aok_uc). the rest are never used
	std::cout<<"---------- Layer ("<< num_of_layers-1 <<") ----------"<<std::endl;
	std::cout<<"conv 1"<<std::endl;
	conv(layers[num_of_layers-1].Aofind, layers[num_of_layers-1].Aok1d, layers[num_of_layers-1].Aof1d);
	relu(layers[num_of_layers-1].Aof1d);

	std::cout<<"conv 2"<<std::endl;
	conv(layers[num_of_layers-1].Aof1d, layers[num_of_layers-1].Aok2u, layers[num_of_layers-1].Aof2u);
	relu(layers[num_of_layers-1].Aof2u);

	for (int i=num_of_layers-2; i>=0; --i){
		std::cout<<"upconv to layer "<< i <<std::endl;
		upconv(layers[i+1].Aof2u, layers[i+1].Aok_uc, layers[i].Aofinu);
		relu(layers[i].Aofinu);

		std::cout<<"---------- Layer ("<< i <<") ----------"<<std::endl;
		std::cout<<"conv 1"<<std::endl;
		conv(layers[i].Aofinu, layers[i].Aok1u1, layers[i].Aof1u);
		conv(layers[i].Aof2d, layers[i].Aok1u2, layers[i].Aof1u, true); // 'true' means just add results, dont wipe previous values
		relu(layers[i].Aof1u);

		std::cout<<"conv 2"<<std::endl;
		conv(layers[i].Aof1u, layers[i].Aok2u, layers[i].Aof2u);
		relu(layers[i].Aof2u);
	}

	std::cout<<"fullconv"<<std::endl;
	fullconv(layers[0].Aof2u, layers[0].Aok_final, layers[0].Aof_final);

	std::cout<< "\nForward pass done.\n"<<std::endl;
}

void backward_pass(Layer *layers, int num_of_layers, const Layer::ArrOfVols &Ao_annots){

	std::cout<<"================================== Backward Pass ========================================"<< std::endl;
	std::cout<<"First error tensor"<<std::endl;
	compute_Aoe_final(layers[0].Aof_final, layers[0].Aoe_final, Ao_annots); // find gradient of loss wrt Aof_final to get Aoe_final

	std::cout<<"First back conv"<<std::endl;
	fullconv(layers[0].Aoe_final, layers[0].Aok_final_back, layers[0].Aoe2u);

	for (int i=0; i<=num_of_layers-2; ++i){
		std::cout<<"---------- Layer ("<< i <<") ----------"<<std::endl;
		std::cout<<"back conv 2"<<std::endl;
		conv(layers[i].Aoe2u, layers[i].Aok2u_back, layers[i].Aoe1u);

		std::cout<<"back conv 1"<<std::endl;
		conv(layers[i].Aoe1u, layers[i].Aok1u1_back, layers[i].Aoeinu);
		conv(layers[i].Aoe1u, layers[i].Aok1u2_back, layers[i].Aoe2d);

		std::cout<<"back upconv to layer "<< i+1 <<std::endl;
		upconv_backward(layers[i].Aoeinu, layers[i+1].Aok_uc, layers[i+1].Aoe2u); 
	}

	std::cout<<"---------- Layer ("<< num_of_layers-1 <<") ----------"<<std::endl;
	std::cout<<"back conv 2"<<std::endl;
	conv(layers[num_of_layers-1].Aoe2u, layers[num_of_layers-1].Aok2u_back, layers[num_of_layers-1].Aoe1d);

	std::cout<<"back conv 1"<<std::endl;
	conv(layers[num_of_layers-1].Aoe1d, layers[num_of_layers-1].Aok1d_back, layers[num_of_layers-1].Aoeind);

	for (int i=num_of_layers-2; i>=0; --i){
		std::cout<<"avgpool back to layer "<< i <<std::endl;
		avgpool_backward(layers[i+1].Aoeind, layers[i].Aoe2d); // adds error to the error found from concat (doesn't overwrite)

		std::cout<<"---------- Layer ("<< i <<") ----------"<<std::endl;
		std::cout<<"back conv 2"<<std::endl;
		conv(layers[i].Aoe2d, layers[i].Aok2d_back, layers[i].Aoe1d);

		std::cout<<"back conv 1"<<std::endl;
		conv(layers[i].Aoe1d, layers[i].Aok1d_back, layers[i].Aoeind);
	}

	std::cout<< "\nBackward pass done.\n"<<std::endl;
}
