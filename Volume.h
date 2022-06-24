#ifndef Volume_H
#define Volume_H
#include <memory>

struct Volume{
	Volume()= default;
	Volume(int D, int W) : d(D), w(W), arr(new double[d*w*w]){} 

	inline double& operator ()(int i, int j, int k) {return arr[i*w*w + j*w + k];} 		// Returns Lvalue 
	inline double operator ()(int i, int j, int k) const {return arr[i*w*w + j*w + k];} 	// Returns Rvalue 

	int d; // depth
	int w; // width and height
	
private:
	std::unique_ptr<double[]> arr;	
};

#endif