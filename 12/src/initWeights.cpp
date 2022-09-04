#include "initWeights.h"

template <typename Scalar>
npy_data<Scalar> test_load(const char * path) {
  npy_data<Scalar> d;
  npy::LoadArrayFromNumpy(path, d.shape, d.fortran_order, d.data);
  return d;
}


  
void readKernel(Layer::ArrOfVols const &target, const char * path, int dim[]){
    auto d = test_load<float>(path);
    int index=0;
    //std::copy(d.data.begin(), d.data.end(), target);
    for(int i=0; i < dim[0]; i++){
        for(int j=0; j < dim[1]; j++){
            for(int n=0; n < dim[2]; n++){
                for(int m=0; m < dim[3]; m++){
                    target[i](j,n,m) = d.data[index];
                    index++;
                }
            }
        }
    }    
}

void readTwoKernel(Layer::ArrOfVols const &target1, Layer::ArrOfVols const &target2, const char * path, int dim[]){
    auto d = test_load<float>(path);
    int index=0;
    int d2 = dim[1] / 2;
    int j2 = 0;
    std::cout << path;
    std::cout<<"Dimension ArrOfVols: "<<target1[0].d<< " x  " << target1[0].w<<std::endl;

    //std::copy(d.data.begin(), d.data.end(), target);
    for(int i=0; i < dim[0]; i++){
        for(int j=0; j < dim[1]; j++){
            for(int n=0; n < dim[2]; n++){
                for(int m=0; m < dim[3]; m++){
                    if(j < d2){
                        target1[i](j,n,m) = d.data[index];
                    }else{
                        j2 = j-d2;
                        target2[i](j2,n,m) = d.data[index];
                    }
                    index++;
                }
            }
        }
    }
}


void init_weights(Layer * layers){

    //--dimensions of weights--
    int d0[] = {4,3,3,3};
    int d1[] = {4,4,3,3};
    int d2[] = {8,4,3,3};
    int d3[] = {8,8,3,3};
    int d4[] = {16,8,3,3};
    int d5[] = {16,16,3,3};
    int d6[] = {32,16,3,3};
    int d7[] = {32,32,3,3};
    int d8[] = {64,32,3,3};
    int d9[] = {64,64,3,3};
    int d10[] = {32,64,3,3};
    int d11[] = {16,32,3,3};
    int d12[] = {16,32,3,3};
    int d13[] = {8,16,3,3};
    int d14[] = {8,16,3,3};
    int d15[] = {4,8,3,3};
    int d16[] = {8,8,3,3};
    int d17[] = {8,8,3,3};
    int d18[] = {3,8,3,3};
    
    int dt0[] = {32,32,2,2};
    int dt1[] = {16,16,2,2};
    int dt2[] = {8,8,2,2};
    int dt3[] = {4,4,2,2};
    
    readKernel(layers[0].Aok1d, "./weights/conv0.weight.npy", d0);
    readKernel(layers[0].Aok2d, "./weights/conv1.weight.npy", d1);
    readTwoKernel(layers[0].Aok1u1,layers[0].Aok1u2, "./weights/conv14.weight.npy", d16);
    readKernel(layers[0].Aok2u, "./weights/conv17.weight.npy", d17);
    readKernel(layers[0].Aok_final, "./weights/conv18.weight.npy", d18);

    readKernel(layers[1].Aok1d, "./weights/conv2.weight.npy", d2);
    readKernel(layers[1].Aok2d, "./weights/conv3.weight.npy", d3);
    readTwoKernel(layers[1].Aok1u1,layers[1].Aok1u2, "./weights/conv14.weight.npy", d14);
    readKernel(layers[1].Aok2u, "./weights/conv15.weight.npy", d15);
    readKernel(layers[1].Aok_uc, "./weights/transposed_conv3.weight.npy", dt3);

    readKernel(layers[2].Aok1d, "./weights/conv4.weight.npy", d4);
    readKernel(layers[2].Aok2d, "./weights/conv5.weight.npy", d5);
    readTwoKernel(layers[2].Aok1u1,layers[2].Aok1u2, "./weights/conv14.weight.npy", d12);
    readKernel(layers[2].Aok2u, "./weights/conv13.weight.npy", d13 );
    readKernel(layers[2].Aok_uc, "./weights/transposed_conv2.weight.npy", dt2);

    readKernel(layers[3].Aok1d, "./weights/conv6.weight.npy", d6);
    readKernel(layers[3].Aok2d, "./weights/conv7.weight.npy", d7);
    readTwoKernel(layers[3].Aok1u1,layers[3].Aok1u2, "./weights/conv14.weight.npy", d10);
    readKernel(layers[3].Aok2u, "./weights/conv11.weight.npy", d11);
    readKernel(layers[3].Aok_uc, "./weights/transposed_conv1.weight.npy", dt1);

    readKernel(layers[4].Aok1d, "./weights/conv8.weight.npy", d8);
    readKernel(layers[4].Aok2u, "./weights/conv9.weight.npy", d9);
    readKernel(layers[4].Aok_uc, "./weights/transposed_conv0.weight.npy", dt0);

}

void init_biases(Layer * layers){
//TODO//
    return;
}

void init_kernels(Layer * layers){
   init_weights(layers);
   //init_biases(layers); 
}
