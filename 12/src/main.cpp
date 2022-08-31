#include <iostream>
#include "Layer.h"
#include "unetFuncs.h"
#include "processImages.h"
#include "initWeights.h"

int main(){

    batchsize = 2;
    input_imgsize = 10; // should be 512
    learning_rate = 1;
    int iter = 1; // total U-net iters
    int num_of_layers = 5;
    Layer layers[num_of_layers];
    for (int i=0; i<num_of_layers; ++i){
        layers[i]=Layer(i, input_imgsize, batchsize);
    }

    Layer::ArrOfVols Ao_annots(create_ArrOfVols(batchsize, 1, input_imgsize)); // only 1 feature channel, so depth is 1
    Layer::ArrOfVols Aoloss(create_ArrOfVols(batchsize, 1, input_imgsize));

    std::cout<<"Read input images"<<std::endl;
    ReadImages(layers[0].Aofind, 0, batchsize, input_imgsize);

    std::cout<<"Read annotations"<<std::endl;
    ReadAnnot(Ao_annots, 0, batchsize, input_imgsize);

    // displayImage(layers[0].Aofind,0,0);

    init_kernels(layers);

    for (int i=0; i<iter; ++i){ // U-net iterations for one batch of images
        forward_pass(layers, num_of_layers);
        compute_Aoloss(Aoloss, layers[0].Aof_final, Ao_annots); // compute loss from annots
        create_all_Aok_backward(layers, num_of_layers);	//create all conv kernles for finding error tensors in the backward pass
        backward_pass(layers, num_of_layers, Ao_annots);
        create_all_Aok_gradient(layers, num_of_layers);
        update_all_Aok(layers, num_of_layers); // update all kernels from thier gradients
        std::cout<<"iter ("<< i << ") done." << std::endl;
    }

    /*
    */
    return 0;
}
