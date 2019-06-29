// nvcc sgemm .c -lcublas
# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# define IDX2C(i,j,ld) (((j)*(ld))+(i))

int gemm(float *a,float *b,float *c,int m,int k,int n){
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context
	// print a row by row
	printf ("a:\n");
	for(int i=0;i<m;i++){
		for(int j=0;j<k;j++){
			printf (" %5.0f",a[IDX2C(i,j,m)]);
		}
		printf ("\n");
	}
	// print b row by row
	printf ("b:\n");
	for(int i=0;i<k;i++){
		for(int j=0;j<n;j++){
			printf (" %5.0f",b[IDX2C(i,j,k)]);
		}
	printf ("\n");
	}
	// print c row by row
	printf ("c:\n");
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			printf (" %5.0f",c[IDX2C(i,j,m)]);
		}
	printf ("\n");
	}
	// on the device
	float *d_a; // d_a - a on the device
	float *d_b; // d_b - b on the device
	float *d_c; // d_c - c on the device
	cudaStat = cudaMalloc((void**)& d_a ,m*k*sizeof(*a)); // device
	// memory alloc for a
	cudaStat = cudaMalloc((void**)& d_b ,k*n*sizeof(*b)); // device
	// memory alloc for b
	cudaStat = cudaMalloc((void**)& d_c ,m*n*sizeof(*c)); // device
	// memory alloc for c
	stat = cublasCreate (&handle); // initialize CUBLAS context
	// copy matrices from the host to the device
	stat = cublasSetMatrix(m,k,sizeof(*a),a,m,d_a,m); //a -> d_a
	stat = cublasSetMatrix(k,n,sizeof(*b),b,k,d_b,k); //b -> d_b
	stat = cublasSetMatrix(m,n,sizeof(*c),c,m,d_c,m); //c -> d_c
	float al=1.0f; // al =1
	float bet=0.0f; // bet =0
	// matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
	// d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
	// al ,bet -scalars
	stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,d_a,m,d_b,k,&bet,d_c,m);
	stat = cublasGetMatrix(m,n, sizeof(*c),d_c,m,c,m); // cp d_c - >c
	printf ("c after Sgemm :\n");
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			printf (" %7.0f",c[IDX2C(i,j,m)]); // print c after Sgemm
		}
	printf ("\n");
	}
	cudaFree(d_a); // free device memory
	cudaFree(d_b); // free device memory
	cudaFree(d_c); // free device memory
	cublasDestroy(handle); // destroy CUBLAS context
	return EXIT_SUCCESS ;
}
// a:
// 11 17 23 29 35
// 12 18 24 30 36
// 13 19 25 31 37
// 14 20 26 32 38
// 15 21 27 33 39
// 16 22 28 34 40
// b:
// 11 16 21 26
// 12 17 22 27
// 13 18 23 28
// 14 19 24 29
// 15 20 25 30
// c:
// 11 17 23 29
// 12 18 24 30
// 13 19 25 31
// 14 20 26 32
// 15 21 27 33
// 16 22 28 34
// c after Sgemm :
// 1566 2147 2728 3309
// 1632 2238 2844 3450
// 1698 2329 2960 3591 // c=al*a*b+bet *c
// 1764 2420 3076 3732
// 1830 2511 3192 3873
// 1896 2602 3308 4014