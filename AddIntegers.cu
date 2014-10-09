
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void AddIntsCUDA(int* a, int* b){
	a[0] += b[0];
}

int main(){
	int a = 5;
	int b = 9;

	int *d_a;
	int *d_b;

	cudaMalloc(&d_a, sizeof(int));
	cudaMalloc(&d_b, sizeof(int));

	cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

	AddIntsCUDA <<<1, 1 >>>(d_a, d_b);

	cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);

	cout << "the answer is: " << a << endl;

	return 0;
}

//Need Error Checking i.e. DeAllocation in case of failure. 