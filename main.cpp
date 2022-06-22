class Layer{
	Layer(n) : n(n){}

	int n;
	int num_of_features = pow(2,6+n);
	int imgsize = 512/pow(2,n);
	// =================== Array of features ==================
	Aofind [num_of_features/2][imgsize][imgsize];
	// Aof1d = (128 elements/feature channels, every elmenent = 256x256)
	Aof1d [num_of_features][imgsize][imgsize];
	Aof2d [num_of_features][imgsize][imgsize]; // white
	Aofinu [num_of_features][imgsize][imgsize]; // blue
	Aof1u [num_of_features][imgsize][imgsize];
	Aof2u [num_of_features][imgsize][imgsize];
	// ================== Array of kernels ===================
	// Aok1d = (128 elements/kernels, every element = 3x3x64)
	// Aok2d = (128 elements/kernels, every element = 3x3x128)
	// Aok1u = (128 elements/kernels, every element = 3x3x256)
	// Aok2u = (128 elements/kernels, every element = 3x3x128)
	// Aok_uc (empty for firstlayr) = (64/128? elements/kernels, every element = 3x3x256)
	// Aok [w][h][depth][number]
	Aok1d 	[num_of_features][3][3][num_of_features/2];
	Aok2d 	[num_of_features][3][3][num_of_features];
	Aok1u 	[num_of_features][3][3][num_of_features*2];
	Aok2u 	[num_of_features][3][3][num_of_features];
	Aok_uc 	[num_of_features][3][3][num_of_features*2];


	// ============================
	// 1 Errortensor per feature channel
	// (6 errortensor arrs per layer)
	// and maybe more...
	// ============================
	(pad conv and ReLU)
	(pad the input first, then conv, apply ReLU, write to output)
	conv(double* ker, double* feat, bool overwrite=False){}
	conv(Aok1d, Aofind) (depth of every kernel in Aok1d is equal to number of input channels/features) --> (overwrite Aof1d)
	conv(Aok2u, Aof1u) ---> (overwrite Aof2u)
	for  
	// =============================
	avgpool(&L2.Aofind){
	overwrite L2.Aofind
	}
	// ==============================
	upconv(&L1.Aofinu){
	overwrite L1.Aofinu
	}
};




int main(){

Layer L;
for (int i=0; i<5; ++i){
	layers[i]=L(i);
}

readimage into layers[0].Aofind[0];

for (int i=0; i<4; ++i){
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