#include <cmath>
#include <iostream>
#include "Layer.h"
#include "Volume.h"

extern int batchsize;
extern int input_imgsize;

void relu(Layer::ArrOfVols &input);
void conv(Layer::ArrOfVols const &input, Layer::ArrOfVols const &kernel, Layer::ArrOfVols &output);
void fullconv(Layer::ArrOfVols const &input, Layer::ArrOfVols const &kernel, Layer::ArrOfVols &output);
void avgpool(Layer::ArrOfVols const &input, Layer::ArrOfVols &output);
void avgpool_backward(Layer::ArrOfVols const &input, Layer::ArrOfVols &output);
void upconv(Layer::ArrOfVols const &input, Layer::ArrOfVols const &kernel, Layer::ArrOfVols &output);
void upconv_backward(Layer::ArrOfVols const &input, Layer::ArrOfVols const &kernel, Layer::ArrOfVols &output);
void concat(Layer::ArrOfVols const &input1, Layer::ArrOfVols const &input2, Layer::ArrOfVols &output);
void create_Aok_backward(Layer::ArrOfVols const &Aok, Layer::ArrOfVols &Aok_back);
void create_all_Aok_backward(Layer *layers, int num_of_layers);
void compute_Aoloss(Layer::ArrOfVols &Aoloss, Layer::ArrOfVols const &Aof_final, Layer::ArrOfVols const &Ao_annots);
Layer::ArrOfVols create_ArrOfVols(int num_of_arrs, int depth, int width);
void compute_Aok_gradient(Layer::ArrOfVols const &input, Layer::ArrOfVols const &error_tensor, Layer::ArrOfVols &Aok_gradient_avg);
void create_all_Aok_gradient(Layer *layers, int num_of_layers);
void compute_Aoe_final(Layer::ArrOfVols const &Aof_final, Layer::ArrOfVols &Aoe_final, Layer::ArrOfVols &Ao_annots);
