#include <iostream>
#include "unetFuncs.h"
#include "processImages.h"
#include "initWeights.h"
#include "ConvStruct.h"

int main(int argc, char *argv[]){
    learning_rate = 1;
    if(argc<3){std::cout<<"args: [imgsize] [batchsize] [num_of_batches]"<<std::endl; return 0;}
    input_imgsize = std::stoi(argv[1]);      // should be 512
    batchsize  = std::stoi(argv[2]);         // should be 8
    int num_of_batches = std::stoi(argv[3]); // should be 128/batchsize
    std::cout<< "imgsize: "<< input_imgsize<< ", batchsize: "<< batchsize<< ", num_of_batches: "<< num_of_batches<< std::endl;
    int epochs = 4;      
    int num_of_layers = 5;
    int channel_size = 2;

    int num_of_convstructs = 7 + (num_of_layers-2)*6 + 3; // 7 for first layer, 6 for other layers, 3 for last layer
    ConvStruct *conv_struct = new ConvStruct[num_of_convstructs];
    ConvStruct *layers[num_of_layers];

    create_architecture(layers, conv_struct, num_of_layers, channel_size);

    ConvStruct::ArrOfVols Ao_annots(create_ArrOfVols(batchsize, 1, input_imgsize)); // only 1 feature channel, so depth is 1
    ConvStruct::ArrOfVols Aoloss(create_ArrOfVols(batchsize, 1, input_imgsize));

    // displayImage(layers[0][0].Aof,0,0); // first layer, first ConvStruct, first img in batch, first channel

    // std::cout<<"init kernels"<<std::endl;
    init_kernels(layers);
    float loss_sum=0.0;

    for (int i=0; i<epochs; ++i){ 
        std::cout<<"epoch ("<< i << "): " << std::endl;
        for (int batchNr=0; batchNr<num_of_batches; ++batchNr){

            ReadImages(layers[0][0].Aof, batchNr, batchsize, input_imgsize);
            ReadAnnot(Ao_annots, batchNr, batchsize, input_imgsize);
            forward_pass(layers, num_of_layers);
            compute_Aoloss(Aoloss, layers[0][6].Aof, Ao_annots); // compute loss from annots
            loss_sum += avg_batch_loss(Aoloss);
            create_all_Aok_backward(conv_struct, num_of_convstructs);	//create all conv kernles for finding error tensors in the backward pass
            backward_pass(layers, num_of_layers, Ao_annots);
            create_all_Aok_gradient(layers, num_of_layers);
            update_all_Aok(conv_struct, num_of_convstructs); // update all kernels from thier gradients
        }
        std::cout<<"{ Loss = "<< loss_sum/num_of_batches <<" }\n-------------------------"<< std::endl;
    }

    /*
    */
    return 0;
}
