#include "unetFuncs.h"

int batchsize;
int input_imgsize;
float learning_rate;

void relu(ArrOfVols &input){
	int num_of_features = input[0].d;
	int imgsize = input[0].w; // includes padding (doesnt matter but just in case)
	#pragma omp parallel for
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
void conv(ArrOfVols const &input, ArrOfVols const &kernel, ArrOfVols &output, bool dont_wipe_before_adding){
	int num_of_kernels = output[0].d;
	int depth_of_kernels = input[0].d;
	int imgsize = input[0].w; // imgsize includes padding (both input and output are padded)
	int width_of_kernels = kernel[0].w;

	if (dont_wipe_before_adding){
		#pragma omp parallel for
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
		#pragma omp parallel for
		for (int b=0; b<batchsize; ++b){
			for(int i = 0; i < num_of_kernels; ++i){	// zero out old output before adding
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

void fullconv(ArrOfVols const &input, ArrOfVols const &kernel, ArrOfVols &output){
	int num_of_kernels = output[0].d;
	int depth_of_kernels = input[0].d;
	int imgsize = input[0].w; // imgsize is not padded

	#pragma omp parallel for
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

void avgpool(ArrOfVols const &input, ArrOfVols &output){ 
	int num_of_features = input[0].d;
	int imgsize = input[0].w; // includes padding

	#pragma omp parallel for
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

void avgpool_backward(ArrOfVols const &input, ArrOfVols &output){ // adds new error to the one found from concat
	int num_of_features = input[0].d;
	int imgsize = input[0].w; // includes padding

	#pragma omp parallel for
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

void upconv(ArrOfVols const &input, ArrOfVols const &kernel, ArrOfVols &output){
	int num_of_kernels = output[0].d; //depth of output or number of kernels should be half of depth of input
	int depth_of_kernels = input[0].d;
	int imgsize = input[0].w; // includes padding

	#pragma omp parallel for
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

void upconv_backward(ArrOfVols const &input, ArrOfVols const &kernel, ArrOfVols &output){
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

void create_all_Aok_backward(ConvStruct *conv_struct, int num_of_convstructs){
	// std::cout<<"create_all_Aok_backward"<<std::endl;
	#pragma omp parallel for
	for (int k=0; k<num_of_convstructs; ++k){
		for (int j=0; j<conv_struct[k].in; ++j){ 	// depth of forward kernels aok[0].d
			for (int i=0; i<conv_struct[k].out; ++i){ 	// num of forward kernels 	aok_back[0].d
				for (int x=0; x<conv_struct[k].kernel_size; ++x){
					for (int y=0; y<conv_struct[k].kernel_size; ++y){
						conv_struct[k].Aok_back[j](i, x, y) = conv_struct[k].Aok[i](j, x, y);
					}
				}
			}
		}
	}

}

void compute_Aoloss(ArrOfVols &Aoloss, ArrOfVols const &Aof_final, ArrOfVols const &Ao_annots){
	#pragma omp parallel for
	for (int b=0; b<batchsize; ++b){
		for(int x = 1; x < Aof_final[0].w-1; ++x){
			for(int y = 1; y < Aof_final[0].w-1; ++y){
				float sum = 0.0;
				for(int i = 0; i < 3; ++i){ // (class indices - 1)
					sum += exp(Aof_final[b](i,x,y));
				}
				Aoloss[b](0,x-1,y-1) = -log(exp(Aof_final[b](Ao_annots[b](0,x-1,y-1)-1, x, y))/sum);
			}
		}
	}
}

float avg_batch_loss(const ArrOfVols &Aoloss){
	float sum = 0.0;

	for (int b=0; b<batchsize; ++b){
		for(int x = 0; x < input_imgsize; ++x){
			for(int y = 0; y < input_imgsize; ++y){
				sum += Aoloss[b](0,x,y);
			}
		}
	}
	return sum/(input_imgsize*input_imgsize*batchsize);
}

ArrOfVols create_ArrOfVols(int num_of_arrs, int depth, int width){
	ArrOfVols output(new Volume[num_of_arrs]);
	for (int i=0; i<num_of_arrs; ++i){
		output[i]=Volume(depth, width);
	}
	return output;
}

//each input image with a error tensor will generate several gradient kernels. (the number depends on the depth of the error tensor)
//so for each batch, there will be (batchsize * error tensor depth) gradient kernels. 

void compute_Aok_gradient(ArrOfVols const &input, ArrOfVols const &old_error_tensor, ArrOfVols &Aok_gradient){
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

void compute_Aok_uc_gradient(ArrOfVols const &input, ArrOfVols const &old_error_tensor, ArrOfVols &Aok_gradient){
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

void create_all_Aok_gradient(ConvStruct **layers, int num_of_layers){
	// std::cout<<"create_all_Aok_gradient"<< std::endl;
	compute_Aok_gradient(layers[0][5].Aof, layers[0][6].Aoe, layers[0][5].Aok_gradient);	// first layer has an additional kernel
	for (int i=0; i<num_of_layers-2; ++i){
		for (int k=0; k<5; ++k){
			compute_Aok_gradient(layers[i][k].Aof, layers[i][k+1].Aoe, layers[i][k].Aok_gradient);
		}
		if(i==num_of_layers-1){break;}
		compute_Aok_uc_gradient(layers[i+1][5].Aof, layers[i][3].Aoe, layers[i+1][5].Aok_gradient);
	}
	// last layer
	compute_Aok_gradient(layers[num_of_layers-1][0].Aof, layers[num_of_layers-1][1].Aoe, layers[num_of_layers-1][0].Aok_gradient);
	compute_Aok_gradient(layers[num_of_layers-1][1].Aof, layers[num_of_layers-1][2].Aoe, layers[num_of_layers-1][1].Aok_gradient);
	compute_Aok_uc_gradient(layers[num_of_layers-1][2].Aof, layers[num_of_layers-2][3].Aoe, layers[num_of_layers-1][2].Aok_gradient);

}

void compute_Aoe_final(ArrOfVols const &Aof_final, ArrOfVols &Aoe_final, const ArrOfVols &Ao_annots){
	#pragma omp parallel for
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

void update_all_Aok(ConvStruct *conv_struct, int num_of_convstructs){ // update all kernels from their gradient
	// std::cout<<"update_all_Aok"<< std::endl;
	#pragma omp parallel for
	for (int k=0; k<num_of_convstructs; ++k){
		for (int i=0; i<conv_struct[k].out; ++i){	//num of kernels
			for (int d=0; d<conv_struct[k].in; ++d){		// depth of kernels
				for (int x=0; x<conv_struct[k].kernel_size; ++x){
					for (int y=0; y<conv_struct[k].kernel_size; ++y){
						conv_struct[k].Aok[i](d,x,y) = conv_struct[k].Aok[i](d,x,y) - learning_rate*conv_struct[k].Aok_gradient[i](d,x,y);
					}
				}
			}
		}
	}
}

void forward_pass(ConvStruct **layers, int num_of_layers){

	// std::cout<<"================================== Forwared Pass ========================================"<< std::endl;

	for (int i=0; i<num_of_layers; ++i){
		// std::cout<<"---------- Layer ("<< i <<") ----------"<<std::endl;
		// std::cout<<"conv 1"<<std::endl;
		conv(layers[i][0].Aof, layers[i][0].Aok, layers[i][1].Aof);
		relu(layers[i][1].Aof);

		// std::cout<<"conv 2"<<std::endl;
		conv(layers[i][1].Aof, layers[i][1].Aok, layers[i][2].Aof);
		relu(layers[i][2].Aof);

		if (i==num_of_layers-1){break;}
		// std::cout<<"avgpool to layer "<< i+1 <<std::endl;
		avgpool(layers[i][2].Aof, layers[i+1][0].Aof);

	}
		// std::cout<<"upconv to layer "<< num_of_layers-2 <<std::endl;
		upconv(layers[num_of_layers-1][2].Aof, layers[num_of_layers-1][2].Aok, layers[num_of_layers-2][3].Aof);
		relu(layers[num_of_layers-2][3].Aof);

	for (int i=num_of_layers-2; i>=0; --i){
		// std::cout<<"---------- Layer ("<< i <<") ----------"<<std::endl;
		// std::cout<<"conv 3"<<std::endl;
		conv(layers[i][3].Aof, layers[i][3].Aok, layers[i][4].Aof);
		conv(layers[i][2].Aof, layers[i][2].Aok, layers[i][4].Aof, true); // 'true' means just add results, dont wipe previous values
		relu(layers[i][4].Aof);

		// std::cout<<"conv 4"<<std::endl;
		conv(layers[i][4].Aof, layers[i][4].Aok, layers[i][5].Aof);
		relu(layers[i][5].Aof);

		if (i==0){break;}
		// std::cout<<"upconv to layer "<< i <<std::endl;
		upconv(layers[i][5].Aof, layers[i][5].Aok, layers[i-1][3].Aof);
		relu(layers[i-1][3].Aof);
	}

	// std::cout<<"fullconv"<<std::endl;
	fullconv(layers[0][5].Aof, layers[0][5].Aok, layers[0][6].Aof);
	relu(layers[0][6].Aof); // ---------------------------------------- do we relu last?

	// std::cout<< "\nForward pass done.\n"<<std::endl;
}

void backward_pass(ConvStruct **layers, int num_of_layers, const ArrOfVols &Ao_annots){

	// std::cout<<"================================== Backward Pass ========================================"<< std::endl;
	// std::cout<<"First error tensor"<<std::endl;
	compute_Aoe_final(layers[0][6].Aof, layers[0][6].Aoe, Ao_annots); // find gradient of loss wrt Aof_final to get Aoe_final

	// std::cout<<"First back conv"<<std::endl;
	fullconv(layers[0][6].Aoe, layers[0][5].Aok_back, layers[0][5].Aoe);

	for (int i=0; i<=num_of_layers-2; ++i){
		// std::cout<<"---------- Layer ("<< i <<") ----------"<<std::endl;
		// std::cout<<"back conv 4"<<std::endl;
		conv(layers[i][5].Aoe, layers[i][4].Aok_back, layers[i][4].Aoe);

		// std::cout<<"back conv 3"<<std::endl;
		conv(layers[i][4].Aoe, layers[i][3].Aok_back, layers[i][3].Aoe);
		conv(layers[i][4].Aoe, layers[i][2].Aok_back, layers[i][2].Aoe);

		if(i==num_of_layers-2){		
			// std::cout<<"back upconv to layer "<< i+1 <<std::endl;
			upconv_backward(layers[i][3].Aoe, layers[i+1][2].Aok_back, layers[i+1][2].Aoe);
		}
		else{
			// std::cout<<"back upconv to layer "<< i+1 <<std::endl;
			upconv_backward(layers[i][3].Aoe, layers[i+1][5].Aok_back, layers[i+1][5].Aoe); 
		}
	}

	// std::cout<<"---------- Layer ("<< num_of_layers-1 <<") ----------"<<std::endl;
	// std::cout<<"back conv 2"<<std::endl;
	conv(layers[num_of_layers-1][2].Aoe, layers[num_of_layers-1][1].Aok_back, layers[num_of_layers-1][1].Aoe);

	// std::cout<<"back conv 1"<<std::endl;
	conv(layers[num_of_layers-1][1].Aoe, layers[num_of_layers-1][0].Aok_back, layers[num_of_layers-1][0].Aoe);

	for (int i=num_of_layers-2; i>=0; --i){
		// std::cout<<"avgpool back to layer "<< i <<std::endl;
		avgpool_backward(layers[i+1][0].Aoe, layers[i][2].Aoe); // adds error to the error found from concat (doesn't overwrite)

		// std::cout<<"---------- Layer ("<< i <<") ----------"<<std::endl;
		// std::cout<<"back conv 2"<<std::endl;
		conv(layers[i][2].Aoe, layers[i][1].Aok_back, layers[i][1].Aoe);

		// std::cout<<"back conv 1"<<std::endl;
		conv(layers[i][1].Aoe, layers[i][0].Aok_back, layers[i][0].Aoe);
	}

	// std::cout<< "\nBackward pass done.\n"<<std::endl;
}

void create_architecture(ConvStruct **layers, ConvStruct *conv_struct, int num_of_layers, int channel_size){

    for (int layer_index=0; layer_index<num_of_layers; ++layer_index){
        int num_of_features = pow(2, channel_size + layer_index);
        int padded_imgsize = input_imgsize/pow(2,layer_index) + 2;

        if (layer_index==0) {
            layers[layer_index] = &conv_struct[0];
            layers[layer_index][0] = ConvStruct(3,                 num_of_features,      batchsize, padded_imgsize, 3);
            layers[layer_index][1] = ConvStruct(num_of_features,   num_of_features,      batchsize, padded_imgsize, 3);
            layers[layer_index][2] = ConvStruct(num_of_features,   num_of_features*2,    batchsize, padded_imgsize, 3);
            layers[layer_index][3] = ConvStruct(num_of_features,   num_of_features*2,    batchsize, padded_imgsize, 3);
            layers[layer_index][4] = ConvStruct(num_of_features*2, num_of_features*2,    batchsize, padded_imgsize, 3);
            layers[layer_index][5] = ConvStruct(num_of_features*2, 3,                    batchsize, padded_imgsize, 1);
            layers[layer_index][6] = ConvStruct(3,                 0,                    batchsize, padded_imgsize, 0);

        }
        else if (layer_index==num_of_layers-1){
            layers[layer_index] = &conv_struct[layer_index*6 + 1];
            layers[layer_index][0] = ConvStruct(num_of_features/2, num_of_features,      batchsize, padded_imgsize, 3);
            layers[layer_index][1] = ConvStruct(num_of_features,   num_of_features/2,    batchsize, padded_imgsize, 3);
            layers[layer_index][2] = ConvStruct(num_of_features/2, num_of_features/2,    batchsize, padded_imgsize, 2);

        }
        else {
            layers[layer_index] = &conv_struct[layer_index*6 + 1];
            layers[layer_index][0] = ConvStruct(num_of_features/2, num_of_features,      batchsize, padded_imgsize, 3);
            layers[layer_index][1] = ConvStruct(num_of_features,   num_of_features,      batchsize, padded_imgsize, 3);
            layers[layer_index][2] = ConvStruct(num_of_features,   num_of_features,      batchsize, padded_imgsize, 3);
            layers[layer_index][3] = ConvStruct(num_of_features,   num_of_features,      batchsize, padded_imgsize, 3);
            layers[layer_index][4] = ConvStruct(num_of_features,   num_of_features/2,    batchsize, padded_imgsize, 3);
            layers[layer_index][5] = ConvStruct(num_of_features/2, num_of_features/2,    batchsize, padded_imgsize, 2);

        }
    }
}
