#include "CImg.h"
#include "Layer.h"
#include "iostream"
#include <sstream>
#include <cmath>

void ReadImages(Layer::ArrOfVols &output, int batchNr, int batchSize, int input_imgsize);
void ReadAnnot(Layer::ArrOfVols &output, int batchNr, int batchSize, int input_imgsize);
void displayImage(Layer::ArrOfVols &output, int b, int c);