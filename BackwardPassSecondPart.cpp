//each input image with a error tensor will generate several gradient kernels. (the number depends on the depth of the error tensor)
//so for each batch, there will be (batchsize * error tensor depth) gradient kernels. 

Layer::ArrOfVols gradient_kernel;
void b2(Layer::ArrOfVols const &input, Layer::ArrOfVols const &error_tensor){
    for (int b = 0; b < batchsize; ++b){
        for (int n = 0; n < error_tensor[0].d; ++n){ // the number of gradient kernels for each input = error_tensor[0].d
            for (int d = 0; d < input[0].d; ++d){ // the depth of each gradient kernel = input[0].d
                for (int h = 0; h < 3; ++h){ //the size for each gradient kernel will always be 3 * 3 in 2D
                    for (int w = 0; w < 3; ++w){
                        float tmp = 0;
                        for (int j = 0; j < error_tensor[0].w; ++j){
                            for (int i = 0; i < error_tensor[0].w; ++i){
                                tmp += error_tensor[b](n,i,j) * input[b](d,i+w,j+h);
                            }
                        }
                        gradient_kernel[n + b * depth_of_error_tensor](d,w,h) = tmp;
                    }
                }
            }
        }
    }
}