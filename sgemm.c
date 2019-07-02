#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

int gemm(float *a,float *b,float *c,int m,int k,int n, float al, float bet, float *biases){
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context
	// on the device
	float *d_a=a; // d_a - a on the device
	float *d_b=b; // d_b - b on the device
	float *d_c=c; // d_c - c on the device

	cudaStat = cudaMalloc((void**)&d_a,m*k*sizeof(*a));
	// memory alloc for a
	cudaStat = cudaMalloc((void**)&d_b,k*n*sizeof(*b));
	// memory alloc for b
	cudaStat = cudaMalloc((void**)&d_c,m*n*sizeof(*c));
	// memory alloc for c

	stat = cublasCreate(&handle); // initialize CUBLAS context

	// copy matrices from the host to the device
	stat = cublasSetMatrix(m,k,sizeof(*a),a,m,d_a,m); //a -> d_a
	stat = cublasSetMatrix(k,n,sizeof(*b),b,k,d_b,k); //b -> d_b

	// //stat = cublasSetMatrix(m,n,sizeof(*c),c,m,d_c,m); //c -> d_c
	// matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
	// d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
	// al ,bet -scalars
	stat = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,d_a,m,d_b,k,&bet,d_c,m);

	stat = cublasGetMatrix(m,n, sizeof(*c),d_c,m,c,m); // cp d_c - >c

	cudaFree(d_a); // free device memory
	cudaFree(d_b); // free device memory
	cudaFree(d_c); // free device memory

	cublasDestroy(handle); // destroy CUBLAS context

	return cudaStat;
}