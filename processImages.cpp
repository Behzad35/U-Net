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
        string_builder << "../training_data" << "/images/image" << imageCounter << ".jpg";
        path = string_builder.str();
        cimg_library::CImg<float> imageIn(path.c_str());
        
        for(int i = 0; i<512; i++){
            for(int j = 0; j < 512; j++){
                 
                float R = *imageIn.data(i,j,0,0);
                float G = *imageIn.data(i,j,0,1);
                float B = *imageIn.data(i,j,0,2);

                float Clinear = 0.2126 * R + 0.7152 * G + 0.0722 * B;
                float Csrgb = 0;

                if (Clinear <= 0.0031308){
                    Csrgb = 12.92 * Clinear;
                }
                else{
                    Csrgb = 1.055 * pow(Clinear, 1/2.4) - 0.055;
                }
                
                output[b](i,j,0) = Csrgb;
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
        imageOut(x,y,c) = output[b](x,y,c);
    }
    imageOut.display();
}
