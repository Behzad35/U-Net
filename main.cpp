class Layer{
	Layer(n) : n(n){}

	int n;
	int num_of_features = pow(2,6+n);
	int imgsize = 512/pow(2,n);
	typedef double valtype;
	// =================== Array of features ==================
	valtype Aofind [num_of_features/2][imgsize][imgsize];
	// Aof1d = (128 elements/feature channels, every elmenent = 256x256)
	valtype Aof1d [num_of_features][imgsize][imgsize];
	valtype Aof2d [num_of_features][imgsize][imgsize]; // white
	valtype Aofinu [num_of_features][imgsize][imgsize]; // blue
	valtype Aof1u [num_of_features][imgsize][imgsize];
	valtype Aof2u [num_of_features][imgsize][imgsize];

	//=====================padded feature arrays============================
	valtype Aof1dPadded [num_of_features][imgsize+2][imgsize+2];
	valtype Aof2dPadded [num_of_features][imgsize+2][imgsize+2]; // white
	valtype AofinuPadded [num_of_features][imgsize+2][imgsize+2]; // blue
	valtype Aof1uPadded [num_of_features][imgsize+2][imgsize+2];
	valtype Aof2uPadded [num_of_features][imgsize+2][imgsize+2];
	// ================== Array of kernels ===================
	// Aok1d = (128 elements/kernels, every element = 3x3x64)
	// Aok2d = (128 elements/kernels, every element = 3x3x128)
	// Aok1u = (128 elements/kernels, every element = 3x3x256)
	// Aok2u = (128 elements/kernels, every element = 3x3x128)
	// Aok_uc (empty for firstlayr) = (64 elements/kernels, every element = 2x2x128) from current layer to upper layer
	// 		[number][depth][w][h]
	valtype Aok1d 	[num_of_features][num_of_features/2][3][3];
	valtype Aok2d 	[num_of_features][num_of_features][3][3];
	valtype Aok1u 	[num_of_features][num_of_features*2][3][3];
	valtype Aok2u 	[num_of_features][num_of_features][3][3];
	valtype Aok_uc 	[num_of_features/2][num_of_features][2][2];


	// ============================
	// 1 Errortensor per feature channel
	// (6 errortensor arrs per layer)
	// and maybe more...
	// ============================
	// (pad conv and ReLU)
	// (pad the input first, then conv, apply ReLU, write to output)
	// conv(double* ker, double* feat, bool overwrite=False){}
	// conv(Aok1d, Aofind) (depth of every kernel in Aok1d is equal to number of input channels/features) --> (overwrite Aof1d)
	// conv(Aok2u, Aof1u) ---> (overwrite Aof2u)
	void pad(double*** input, double*** output){
		for(int c = 0; c < num_of_features; c++){
			for(int i = 0; i < imgsize+2; i++){
				output[c][0][i] = 0;
				output[c][i][0] = 0;
				output[c][0][imgsize+1] = 0;
				output[c][imgsize+1][0] = 0;
			}
		}
		for(int c = 0; c < num_of_features; c++){
			for(int i = 0; i < imgsize; i++){
				for(int j = 0; j < imgsize; j++){
					output[c][i+1][j+1] = input[c][i][j];
				}
			}
		}
	}

	void ReLU(double*** input){
		for(int c = 0; c < num_of_features; c++){
			for(int i = 0; i < imgsize; i++){
				for(int j = 0; j < imgsize; j++){
					if(input[c][i][j] < 0) input[c][i][j] = 0;
				}
			}
		}
	}

	void conv(double*** input, double**** kernel, double*** output){
		int s = 1;			//stride
		int layernNr = n;
		int num_of_kernels = num_of_features/2;
		int channels = num_of_features;
		for(int i = 0; i < imgsize; i++){
			for(int j = 0; j < imgsize; j++){
				for(int h = 0; h < num_of_kernels; h++){
					for(int k = 0; k < channels; k++){
						for(int n = 0; n < 3; n++){
							for(int m = 0; m < 3; m++){
								output[k][i][j] += input[k][i*s+n+1][j*s+m+1] * kernel[h][k][n][m];
							}
						}
					}
				}
			}
		}
	}

	// =============================
	void avgpool(double* Aofind_lower){ // writes to lower layer
		for (int i=0; i<num_of_features; ++i){
			for (int x=0; x<imgsize; x+=2){
				for(int y=0; y<imgsize; y+=2){
					Aofind_lower[i][x/2][y/2] = 0.25*(Aof2d[i][x][y] + Aof2d[i][x+1][y] + Aof2d[i][x][y+1] + Aof2d[i][x+1][y+1]);      
				}
			}
		}
	}
	// ==============================
	void upconv(double* Aofinu_upper){ // writes to upper layer
		int num_of_upconv_kernels = num_of_features/2;
		int depth_of_upconv_kernels = num_of_features;
		for (int i=0; i<num_of_upconv_kernels; ++i){
			for (int j=0; j<depth_of_upconv_kernels; ++j){
				for (int x=0; x<imgsize; ++x){
					for (int y=0; y<imgsize; ++y){
						Aofinu_upper[i][2*x][2*y]		= Aok_uc[i][j][0][0] * Aof2u[j][x][y];
						Aofinu_upper[i][2*x+1][2*y]		= Aok_uc[i][j][1][0] * Aof2u[j][x][y];
						Aofinu_upper[i][2*x][2*y+1]		= Aok_uc[i][j][0][1] * Aof2u[j][x][y];
						Aofinu_upper[i][2*x+1][2*y+1]	= Aok_uc[i][j][1][1] * Aof2u[j][x][y];
					}
				}
			}
		}
	}

};


int main(){


Layer layers[4];
for (int i=0; i<=4; ++i){
	layers[i]=Layer(i);
}

readimage into layers[0].Aofind[0];

for (int i=0; i<=3; ++i){
layers[i].conv(layers[i].Aok1d, layers[i].Aofind);
layers[i].conv(layers[i].Aok2d, layers[i].Aof1d);
layers[i].avgpool(layers[i+1].Aofind);
}

layers[4].conv(layers[4].Aok1d, layers[4].Aofind);
layers[4].conv(layers[4].Aok2d, layers[4].Aof1d);

for (int i=3; i>=0; --i){
layers[i+1].upconv(layers[i].Aofinu)
layers[i].conv(layers[i].Aok1u, layers[i].Aofinu);
layers[i].conv(layers[i].Aok1u, layers[i].Aof2d, 1);
layers[i].conv(layers[i].Aok2u, layers[i].Aof1u);
}


return 0;
}
