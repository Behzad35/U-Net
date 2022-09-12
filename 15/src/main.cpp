#include <iostream>
#include "unetFuncs.h"
#include "processImages.h"
#include "initWeights.h"
#include "ConvStruct.h"

int main(int argc, char *argv[]){
    learning_rate = 0.001;
    if(argc<4){std::cout<<"args: [imgsize] [batchsize] [num_of_batches]"<<std::endl; return 0;}
    input_imgsize = std::stoi(argv[1]);      // should be 512
    batchsize  = std::stoi(argv[2]);         // should be 8
    int num_of_batches = std::stoi(argv[3]); // should be 128/batchsize
    int epochs = 1000;      
    int num_of_layers = 5;
    int channel_size = 2;
    bool write_segmap = false;
    std::cout<< "\nimgsize: "<< input_imgsize<< "\t batchsize: "<< batchsize<< "\t num_of_batches: "<< num_of_batches<< std::endl;
    std::cout<< "epochs: "<< epochs<< "\t num_of_layers: "<< num_of_layers<<"\t channel_size: "<< channel_size<< "\t learning_rate: "<< learning_rate<< std::endl;

    int num_of_convstructs = 7 + (num_of_layers-2)*6 + 3; // 7 for first layer, 6 for other layers, 3 for last layer
    ConvStruct *conv_struct = new ConvStruct[num_of_convstructs];
    ConvStruct *layers[num_of_layers];

    create_architecture(layers, conv_struct, num_of_layers, channel_size);
    ConvStruct::ArrOfVols Ao_annots(create_ArrOfVols(batchsize, 1, input_imgsize)); // only 1 feature channel, so depth is 1
    ConvStruct::ArrOfVols Aoloss(create_ArrOfVols(batchsize, 1, input_imgsize));
    ConvStruct::ArrOfVols Ao_segmap(create_ArrOfVols(batchsize, 1, input_imgsize));
    // convert_all_imgs_to_csv();
  
    std::cout<<"init kernels"<<std::endl;
    init_kernels(layers);
    // init_kernel_guess(conv_struct, num_of_convstructs, 0.001); ///////// for debug

    for (int i=0; i<epochs; ++i){ 
        std::cout<<"\n========================================================="<< std::endl;
        std::cout<<"{{{{{{{{{{{{{{{{{{{{{{ epoch ("<< i << ") }}}}}}}}}}}}}}}}}}}}}}" << std::endl;
        std::cout<<"=========================================================\n"<< std::endl;
        float loss_sum=0.0;
        for (int batchNr=0; batchNr<num_of_batches; ++batchNr){
            std::cout<<"\n---------------------------------------------------------------"<< std::endl;
            std::cout<< "{{ batchNr = " << batchNr << " }}"<< std::endl;
            std::cout<<"---------------------------------------------------------------\n"<< std::endl;

            // ReadImages(layers[0][0].Aof, batchNr, batchsize, input_imgsize);
            // ReadAnnot(Ao_annots, batchNr, batchsize, input_imgsize);

            read_img_text(layers[0][0].Aof, batchNr);
            read_annot_text(Ao_annots, batchNr);

            ////////////////////////////////////////////////////////// for debug /////////////////////////////////////
        /*
            std::cout<< "|||conv 0 to 1|||" << std::endl;
            conv(layers[0][0].Aof, layers[0][0].Aok, layers[0][1].Aof);
            relu(layers[0][1].Aof);
            std::cout<< "|||fullconv 1 to 6|||" << std::endl;
            fullconv(layers[0][1].Aof, layers[0][1].Aok, layers[0][6].Aof);

            compute_segmap(layers[0][6].Aof, Ao_segmap);
            compute_Aoloss(Aoloss, layers[0][6].Aof, Ao_annots);
            loss_sum += avg_batch_loss(Aoloss);
            create_all_Aok_backward(conv_struct, num_of_convstructs);
            std::cout<< "|||compute_Aoe_final|||" << std::endl;
            compute_Aoe_final(layers[0][6].Aof, layers[0][6].Aoe, Ao_annots); 

            std::cout<< "|||fullconv back|||" << std::endl;
            fullconv(layers[0][6].Aoe, layers[0][1].Aok_back, layers[0][1].Aoe);
            std::cout<< "|||back conv 1 to 0|||" << std::endl;
            conv(layers[0][1].Aoe, layers[0][0].Aok_back, layers[0][0].Aoe);


            std::cout<< "|||compute_Aok_gradient 1|||" << std::endl;
            compute_Aok_gradient(layers[0][1].Aof, layers[0][6].Aoe, layers[0][1].Aok_gradient);
            std::cout<< "|||compute_Aok_gradient 0|||" << std::endl;
            compute_Aok_gradient(layers[0][0].Aof, layers[0][1].Aoe, layers[0][0].Aok_gradient);
            update_all_Aok(conv_struct, num_of_convstructs);
        */

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // displayImage(layers[0][0].Aof,0,0, input_imgsize); // first layer, first ConvStruct, first img in batch, first channel
            

            forward_pass(layers, num_of_layers);
            if(write_segmap) compute_segmap(layers[0][6].Aof, Ao_segmap);
            compute_Aoloss(Aoloss, layers[0][6].Aof, Ao_annots); // compute loss from annots
            loss_sum += avg_batch_loss(Aoloss);
            float min, max;
            minmax_batch_loss(Aoloss, min, max);

            create_all_Aok_backward(conv_struct, num_of_convstructs);	//create all conv kernles for finding error tensors in the backward pass
            backward_pass(layers, num_of_layers, Ao_annots);
            create_all_Aok_gradient(layers, num_of_layers);
            update_all_Aok(conv_struct, num_of_convstructs); // update all kernels from thier gradients
            std::cout<< "{{{{{{{{{{{ batch ("<< batchNr<< ") avg_loss = "<< loss_sum<< " }}}}}}}}}}}" <<std::endl;
            std::cout<< "min = "<< min<< " max = "<< max<<std::endl;
        }
        std::cout<<"-------------------------\n{{{{{{{{{{{{{{ Loss = "<< loss_sum/num_of_batches <<" }}}}}}}}}}}}}}\n-------------------------"<< std::endl;
        
        if(write_segmap){
            int range_x[2]={0, Ao_segmap[0].w};
            int range_y[2]={0, Ao_segmap[0].w};
            print_arr(Ao_segmap, 0, 0, range_x, range_y, "Ao_segmap_ep"+std::to_string(i));
        }
            
    }


    return 0;
}
