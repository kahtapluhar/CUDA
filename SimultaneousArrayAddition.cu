
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>

using namespace std;

__global__ void AddInts(int *a, int *b, int count){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < count){
		a[id] += b[id];
	}
}

int main(){
	srand(time(NULL));
	int count = 1000;
	int *h_a = new int[count];
	int *h_b = new int[count];

	for (int i = 0; i < count; i++){
		h_a[i] = rand() % 1000;
		h_b[i] = rand() % 1000;
	}

	//Check Arrays (first 5)
	for (int i = 0; i < 5; i++){
		cout << "[" << h_a[i] << "]" << "	"<< "[" << h_b[i] << "]	" << endl;
	}

	int *d_a;
	int *d_b;

	//Allocate variables Memory onto GPU
	if (cudaMalloc(&d_a, sizeof(int)*count) != cudaSuccess){
		cout << "Allocation of d_a failed" << endl;
		delete[] h_a;
		delete[] h_b;
		return 0;
	}

	if (cudaMalloc(&d_b, sizeof(int)*count) != cudaSuccess){
		cout << "Allocation of d_b failed" << endl;
		delete[] h_a;
		delete[] h_b;
		cudaFree(d_a);
		return 0;
	}

	//Copy variables to allocated memory on GPU
	if (cudaMemcpy(d_a, h_a, sizeof(int)*count, cudaMemcpyHostToDevice) != cudaSuccess){
		cout << "Could not copy variables to GPU" << endl;
		delete[] h_a;
		delete[] h_b;
		cudaFree(d_a);
		cudaFree(d_b);
		return 0;
	}

	if (cudaMemcpy(d_b, h_b, sizeof(int) *count, cudaMemcpyHostToDevice) != cudaSuccess){
		cout << "Could not copy variables to GPU" << endl; 
		delete[] h_a;
		delete[] h_b;
		cudaFree(d_a);
		cudaFree(d_b);
		return 0;
	}
	
	//Launch Kernel
	AddInts << < count / 256 + 1, 256 >> > (d_a, d_b, count);
	
	//Copy results back from GPU to HOST
	if (cudaMemcpy(h_a, d_a, sizeof(int)*count, cudaMemcpyDeviceToHost) != cudaSuccess){
		delete[] h_a;
		delete[] h_b;
		cudaFree(d_a);
		cudaFree(d_b);
		cout << "Could not copy variables back" << endl;
		return 0;
	}

	//Print first 5 Additions
	for (int i = 0; i < 5; i++){
		cout << h_a[i] << endl;
	}

	cudaFree(d_a);
	cudaFree(d_b);

	//Delete variables 
	delete[] h_a;
	delete[] h_b;

	return 0;
}