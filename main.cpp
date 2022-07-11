#include <iostream>
#include "Layer.h"
#include "processImages.h"

int batchsize=0;
int batchNr = 0;

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
void conv(Layer::ArrOfVols const &input, Layer::ArrOfVols const &kernel, Layer::ArrOfVols &output, bool wipe_before_adding=true){
    std::cout<<"conv 1 init";
	int num_of_kernels = output[0].d;
	int depth_of_kernels = input[0].d;
	int imgsize = input[0].w; // imgsize includes padding (both input and output are padded)
	int width_of_kernels = kernel[0].w;
	if (wipe_before_adding){
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
	else{
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

Layer::ArrOfVols create_ArrOfVols(int num_of_arrs, int depth, int width){
	Layer::ArrOfVols output(new Volume[num_of_arrs]);
	for (int i=0; i<num_of_arrs; ++i){
		output[i]=Volume(depth, width);
	}
	return output;
}

//=====================================================================
int main(){
    batchsize = 8;
    int input_imgsize = 512;
    int num_of_layers = 5;
    Layer layers[num_of_layers];
    for (int i=0; i<num_of_layers; ++i){
	    layers[i]=Layer(i, input_imgsize, batchsize);
    }
    
    ReadImages(layers[0].Aofind, 0, batchsize);
    displayImage(layers[0].Aofind,2);

    Layer::ArrOfVols Aok_final(create_ArrOfVols(3, layers[0].num_of_features, 1)); // Final Array of kernels (3 kernels) (1x1)
    Layer::ArrOfVols Aof_final(create_ArrOfVols(batchsize, 3, layers[0].imgsize));	// Final Array of features (3 features) (output segmentation map) 

    // rand_init_kernels()
    //================================== Forwared Pass ========================================
    for (int i=0; i<=num_of_layers-2; ++i){
         std::cout<<"---------- Layer ("<< i  <<") ----------"<<std::endl;
         std::cout<<"conv 1"<<std::endl;
       
        conv(layers[i].Aofind, layers[i].Aok1d, layers[i].Aof1d);
        relu(layers[i].Aof1d);

        std::cout<<"conv 2"<<std::endl;
        conv(layers[i].Aof1d, layers[i].Aok2d, layers[i].Aof2d);
        relu(layers[i].Aof2d);

        std::cout<<"avgpool to layer "<< i+1 <<std::endl;
        avgpool(layers[i].Aof2d, layers[i+1].Aofind);

    }
    // lowest layer uses features (Aofinu, Aof1u and Aof2u) and kernels (Aok1u, Aok2u and Aok_uc). the rest are never used
    std::cout<<"---------- Layer ("<< num_of_layers-1 <<") ----------"<<std::endl;
    std::cout<<"conv 1"<<std::endl;
    conv(layers[num_of_layers-1].Aofinu, layers[num_of_layers-1].Aok1u, layers[num_of_layers-1].Aof1u);
    relu(layers[num_of_layers-1].Aof1u);

    std::cout<<"conv 2"<<std::endl;
    conv(layers[num_of_layers-1].Aof1u, layers[num_of_layers-1].Aok2u, layers[num_of_layers-1].Aof2u);
    relu(layers[num_of_layers-1].Aof2u);

    for (int i=num_of_layers-2; i>=0; --i){
        std::cout<<"upconv to layer "<< i <<std::endl;
        upconv(layers[i+1].Aof2u, layers[i+1].Aok_uc, layers[i].Aofinu);
        relu(layers[i].Aofinu);

        std::cout<<"---------- Layer ("<< i <<") ----------"<<std::endl;
        std::cout<<"conv 1"<<std::endl;
        conv(layers[i].Aof2d, layers[i].Aok1u, layers[i].Aof1u);	// add white to Aof1u
        conv(layers[i].Aofinu, layers[i].Aok1u, layers[i].Aof1u, false);	// dont wipe Aof1u, just add blue to Aof1u
        relu(layers[i].Aof1u);

        std::cout<<"conv 2"<<std::endl;
        conv(layers[i].Aof1u, layers[i].Aok2u, layers[i].Aof2u);
        relu(layers[i].Aof2u);
    }

    std::cout<<"Final conv"<<std::endl;
    conv(layers[0].Aof2u, Aok_final, Aof_final);

    std::cout<<"writing output"<<std::endl;
    
return 0;
}
