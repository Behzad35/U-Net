#include "processImages.h"

using namespace cimg_library;

//=====================================================================
// Reads batchSize images from the batchNr batch into the output array of Vols
// ====================================================================
void ReadImages(ConvStruct::ArrOfVols &output, int batchNr, int batchSize, int input_imgsize){
    int imageCounter = 0 + batchNr * batchSize;
    std::string path = "";
    for(int b = 0; b < batchSize; ++b){
        std::ostringstream string_builder;
        string_builder << "training_data/images/image" << imageCounter << ".jpg";
        path = string_builder.str();
        cimg_library::CImg<float> imageIn(path.c_str());
        
        for(int i = 0; i<input_imgsize; ++i){
            for(int j = 0; j <input_imgsize; ++j){
                for(int c=0; c<3; ++c) // R=0, G=1, B=2
                 
                output[b](c,i+1,j+1)  = *imageIn.data(i,j,0,c)/255; // read image into interior of output 
            }
        }
        // float test=0.0;
        // for(int i = 0; i<input_imgsize; ++i){
        //     for(int j = 0; j <input_imgsize; ++j){
        //         for(int c=0; c<3; ++c) // R=0, G=1, B=2
                 
        //         test+=output[b](c,i+1,j+1); // read image into interior of output 
        //     }
        // }
        // std::cout<< test<<std::endl;


        imageCounter++;
    }
}

void ReadAnnot(ConvStruct::ArrOfVols &output, int batchNr, int batchSize, int input_imgsize){
    int imageCounter = 0 + batchNr * batchSize;
    std::string path = "";
    for(int b = 0; b < batchSize; ++b){
        std::ostringstream string_builder;
        string_builder << "training_data/annotations/annotation" << imageCounter << ".png";
        path = string_builder.str();
        cimg_library::CImg<float> imageIn(path.c_str());
        
        for(int i = 0; i<input_imgsize; ++i){
            for(int j = 0; j < input_imgsize; ++j){
                output[b](0,i,j)  = *imageIn.data(i,j,0,0); // Annots are not padded, also only read the red channel
            }
        }

        imageCounter++;
    }
}

//====================================================
//Displays image
// b is the b-th image in the batch, c is the c-th channel in that ArrOfVols for that image
//===================================================
void displayImage(ConvStruct::ArrOfVols &output, int b, int c, int input_imgsize){ 
    cimg_library::CImg<float> imageOut(input_imgsize,input_imgsize,1,1,0);
    cimg_forXYC(imageOut, x, y, c){
        imageOut(x,y,c) = output[b](c,x,y);
    }
    imageOut.display();
}

void convert_all_imgs_to_csv(){ // converts all images to .csv files
    ConvStruct::ArrOfVols img(create_ArrOfVols(1, 3, input_imgsize+2));
    ConvStruct::ArrOfVols annot(create_ArrOfVols(1, 1, input_imgsize));

    for (int i=0; i<128; ++i){
        std::cout<<"converting img "<< i << std::endl;
        ReadImages(img, i, 1, input_imgsize);
        ReadAnnot(annot, i, 1, input_imgsize);

        int range_x[2]={1, img[0].w -1};
        int range_y[2]={1, img[0].w -1};
        print_arr(img, 0, 0, range_x, range_y, "img_"+std::to_string(i)+"_ch0.csv");
        print_arr(img, 0, 1, range_x, range_y, "img_"+std::to_string(i)+"_ch1.csv");
        print_arr(img, 0, 2, range_x, range_y, "img_"+std::to_string(i)+"_ch2.csv");

        range_x[0]=range_y[0]=0;
        range_x[1]=range_y[1]=annot[0].w;
        print_arr(annot, 0, 0, range_x, range_y, "annot_"+std::to_string(i)+".csv");
    }
}
