#include "processImages.h"

using namespace cimg_library;

//=====================================================================
// Reads batchSize images from the batchNr batch into the output array of Vols
// ====================================================================
void ReadImages(Layer::ArrOfVols &output, int batchNr, int batchSize, int input_imgsize){
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
                 
                output[b](c,i+1,j+1)  = *imageIn.data(i,j,0,c); // read image into interior of output 
            }
        }
        imageCounter++;
    }
}

void ReadAnnot(Layer::ArrOfVols &output, int batchNr, int batchSize, int input_imgsize){
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
void displayImage(Layer::ArrOfVols &output, int b, int c){ 
    cimg_library::CImg<float> imageOut(512,512,1,1,0);
    cimg_forXYC(imageOut, x, y, c){
        imageOut(x,y,c) = output[b](c,x,y);
    }
    imageOut.display();
}