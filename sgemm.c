#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// nvcc sgemm.c -lcublas --compiler-options '-fPIC' -shared -o libsgemm.so -O3

cublasHandle_t HANDLE; // CUBLAS context

cublasStatus_t createHandle(){
	cublasCreate(&HANDLE); // initialize CUBLAS context
}

cublasStatus_t gemm(char tra,char trb,int m,int n,int k,float al,float *a,int lda,float *b, int ldb,float bet,float *c,int ldc,float *biases){
	// cudaError_t cudaStat; // cudaMalloc status
	cublasStatus_t stat; // CUBLAS functions status
	float *d_a=a; // d_a - a on the device
	float *d_b=b; // d_b - b on the device
	float *d_c=c; // d_c - c on the device
	// cudaStat = cudaMalloc((void**)&d_a,m*k*sizeof(float));
	// // memory alloc for a
	// cudaStat = cudaMalloc((void**)&d_b,n*k*sizeof(float));
	// // memory alloc for b
	// cudaStat = cudaMalloc((void**)&d_c,m*n*sizeof(float));
	// memory alloc for c
	cublasOperation_t transa,transb;
	if(tra=='T')
		transa=CUBLAS_OP_T;
	else
		transa=CUBLAS_OP_N;
	if(trb=='T')
		transb=CUBLAS_OP_T;
	else
		transb=CUBLAS_OP_N;
	// stat = cublasSetMatrix(lda,k,sizeof(float),a,lda,d_a,lda); //a -> d_a
	// cudaStat = cudaMemcpy(d_a, a, lda*k*sizeof(float), cudaMemcpyHostToDevice);
	// cudaStat = cudaMemcpy(d_b, b, ldb*k*sizeof(float), cudaMemcpyHostToDevice);
	stat = cublasSgemm(HANDLE,
					transa, transb,
					m, n, k,
					&al,
					d_a, lda,
					d_b, ldb,
					&bet,
					d_c, ldc);
	// cudaStat = cudaMemcpy(c, d_c, m*n*sizeof(float), cudaMemcpyDeviceToHost);
	// cudaFree(d_a); // free device memory
	// cudaFree(d_b); // free device memory
	// cudaFree(d_c); // free device memory
	return stat;
}

cublasStatus_t transpose(float *a,float *at,int m,int n,float al, float bet){
	/*	Python method
	sgemm.transpose(dgg.device_ctypes_pointer,
					dgt.device_ctypes_pointer,
					ctypes.c_int(n),ctypes.c_int(m),
					ctypes.c_float(al),ctypes.c_float(bet))
	*/
	cublasStatus_t stat; // CUBLAS functions status
	float *d_a=a; // d_a - a on the device
	float *d_T=at;

	stat = cublasSgeam(HANDLE,
					CUBLAS_OP_T, CUBLAS_OP_N,
					n, m,
					&al,
					d_a, m,
					&bet,
					d_T, n,		//n
					d_T, n);	//n
	// cudaStat = cudaMemcpy(c, d_c, m*n*sizeof(float), cudaMemcpyDeviceToHost);
	// cudaFree(d_a); // free device memory
	return stat;
}

cublasStatus_t destroyHandle(){
	cublasDestroy(HANDLE); // destroy CUBLAS context
}