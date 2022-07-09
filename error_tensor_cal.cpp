#include <cmath>

void error_tensor_cal(Layer::ArrOfVols const &output, Layer::ArrOfVols &error_tensor){
    for (int b = 0; b < batchsize; ++b){
        for (int j = 0; j < output[0].w; ++j){
            for (int i = 0; i < output[0].w; ++i){
                float sum = 0;
                for (int c = 0; c < output[0].d; ++c){
                    sum += exp(output[b](c,i,j));
                for (int c = 0; c < output[0].d; ++c){
                    error_tensor[b](c,i,j) = exp(output[b](c,i,j)) / sum;
        }
    }
}