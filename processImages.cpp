#include "processImages.h"

using namespace cimg_library;

//=====================================================================
// Reads batchSize images from the batchNr batch into the output array of Vols
// Converts the RGB images to gray scale
// ====================================================================
void ReadImages(Layer::ArrOfVols &output, int batchNr, int batchSize){
    int imageCounter = 0 + batchNr * batchSize;
    std::string path = "";
    for(int b = 0; b < batchSize; b++){
        
        std::ostringstream string_builder;
        string_builder << "training_data" << "/images/image" << imageCounter << ".jpg";
        path = string_builder.str();
        cimg_library::CImg<float> imageIn(path.c_str());
        
        for(int i = 0; i<512; i++){
            for(int j = 0; j < 512; j++){
                for(int c=0; c<3; ++c){
             
                   output[b](c,i,j) = *imageIn.data(i,j,0,c);
                }
            }
        }
        imageCounter++;
    }
}

//====================================================
//Displays the b-th Volume of a ArrofVols to the screen
//===================================================
void displayImage(Layer::ArrOfVols &output, int b){
    cimg_library::CImg<float> imageOut(512,512,1,1,0);
    cimg_forXYC(imageOut, x, y, c){
        imageOut(x,y,c) = output[b](c,x,y);
    }
    imageOut.display();
}
