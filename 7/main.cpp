#include <iostream>
#include "Layer.h"
#include "processImages.h"
#include "unetFuncs.h"

int main(){

batchsize = 2;
input_imgsize = 10;
int num_of_layers = 5;
Layer layers[num_of_layers];
for (int i=0; i<num_of_layers; ++i){
	layers[i]=Layer(i, input_imgsize, batchsize);
}

Layer::ArrOfVols Ao_annots(create_ArrOfVols(batchsize, 1, input_imgsize)); // only 1 channel
Layer::ArrOfVols Aoloss(create_ArrOfVols(batchsize, 1, input_imgsize));

std::cout<<"Read input images"<<std::endl;
ReadImages(layers[0].Aofind, 0, batchsize, input_imgsize);

std::cout<<"Read annotations"<<std::endl;
ReadAnnot(Ao_annots, 0, batchsize, input_imgsize);

// displayImage(layers[0].Aofind,0,0);

// init_kernels()
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
	concat(layers[i].Aof2d, layers[i].Aofinu, layers[i].Aof_concat);
	conv(layers[i].Aof_concat, layers[i].Aok1u, layers[i].Aof1u);
	relu(layers[i].Aof1u);

	std::cout<<"conv 2"<<std::endl;
	conv(layers[i].Aof1u, layers[i].Aok2u, layers[i].Aof2u);
	relu(layers[i].Aof2u);
}

std::cout<<"fullconv"<<std::endl;
fullconv(layers[0].Aof2u, layers[0].Aok_final, layers[0].Aof_final);

std::cout<< "\nForward pass done.\n"<<std::endl;
std::cout<<"compute Aoloss"<<std::endl;
compute_Aoloss(Aoloss, layers[0].Aof_final, Ao_annots);

std::cout<<"create all Aok's for backward pass\n"<<std::endl;
create_all_Aok_backward(layers, num_of_layers);	//create conv kernles for finding error tensor in the backward pass

std::cout<<"================================== Backward Pass ========================================"<< std::endl;
// compute_Aoe_final(layers[0].Aof_final, layers[0].Aoe_final); // find gradient of loss wrt Aof_final to get Aoe_final

std::cout<<"First back conv"<<std::endl;
fullconv(layers[0].Aoe_final, layers[0].Aok_final_back, layers[0].Aoe2u);

for (int i=0; i<=num_of_layers-2; ++i){
	std::cout<<"---------- Layer ("<< i <<") ----------"<<std::endl;
	std::cout<<"back conv 2"<<std::endl;
	conv(layers[i].Aoe2u, layers[i].Aok2u_back, layers[i].Aoe1u);

	std::cout<<"back conv 1"<<std::endl;
	conv(layers[i].Aoe1u, layers[i].Aok1u_back, layers[i].Aoe_concat);

	std::cout<<"back upconv to layer "<< i+1 <<std::endl;
	// *** upconv_back();  // use Aoe_concat and apply Aok_uc_back to get Aoe2u in lower layer
}

// lowest layer uses errors (Aoeind, Aoe1d and Aoe2d) and kernels (Aok1d_back, Aok2d_back). the rest are never used
std::cout<<"---------- Layer ("<< num_of_layers-1 <<") ----------"<<std::endl;
std::cout<<"back conv 2"<<std::endl;
conv(layers[num_of_layers-1].Aoe2d, layers[num_of_layers-1].Aok2d_back, layers[num_of_layers-1].Aoe1d);

std::cout<<"back conv 1"<<std::endl;
conv(layers[num_of_layers-1].Aoe1d, layers[num_of_layers-1].Aok1d_back, layers[num_of_layers-1].Aoeind);

for (int i=num_of_layers-2; i>=0; --i){
	std::cout<<"avgpool back to layer "<< i <<std::endl;
	avgpool_backward(layers[i+1].Aoeind, layers[i].Aoe2d);

	std::cout<<"---------- Layer ("<< i <<") ----------"<<std::endl;
	std::cout<<"back conv 2"<<std::endl;
	conv(layers[i].Aoe2d, layers[i].Aok2d_back, layers[i].Aoe1d);

	std::cout<<"back conv 1"<<std::endl;
	conv(layers[i].Aoe1d, layers[i].Aok1d_back, layers[i].Aoeind);
}

std::cout<< "\nBackward pass done.\n"<<std::endl;
std::cout<<"create all Aok gradients" << std::endl;
create_all_Aok_gradient(layers, num_of_layers);

std::cout<<"update all Aok's" << std::endl;
// update_all_Aok();

/*
*/
return 0;
}