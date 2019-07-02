#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

typedef struct arguments
{
	float *coled, *padded;
	int64_t *ind;
	int imsz, indsz, td, cutbsz, cutb, cutcsz,m,n;
}args;

void take_thrd_F(args *ptr){
	float *img_base=ptr->padded + ptr->td*ptr->cutbsz;
	float *cole_base=ptr->coled + ptr->td*ptr->cutcsz;
	for(int bc = 0; bc < ptr->cutb; ++bc)
	{
		float *img=img_base + ptr->imsz*bc;
		float *cole=cole_base + ptr->indsz*bc;
		for (int i = 0; i < ptr->indsz; ++i)
		{
			float *addr=cole+i;
			int c_idx = (addr - ptr->coled);
			int c_row = c_idx/ptr->n;
			int c_col = c_idx%ptr->n;
			int f_idx = ptr->m*c_col + c_row;
			*(ptr->coled+f_idx)=(float)img[ptr->ind[i]];
		}
	}
}

void take_thrd_C(args *ptr){
	for(int bc = 0; bc < ptr->cutb; ++bc)
	{
		float *img=ptr->padded + ptr->imsz*bc + ptr->td*ptr->cutbsz;
		float *cole=ptr->coled + ptr->indsz*bc + ptr->td*ptr->cutcsz;
		for (int i = 0; i < ptr->indsz; ++i)
		{
			cole[i]=(float)img[ptr->ind[i]];	//F, caching is fast.
		}
	}
}

int take(float *padded,int64_t *ind,float *coled,int batches,int imsz, int indsz,int m,int n,char order, int num_threads){
	pthread_t threads[num_threads];
	args *arg=malloc(num_threads*sizeof(args));
	for (int td = 0; td < num_threads; ++td)
	{
		arg[td].padded=padded;
		arg[td].coled=coled;		// mxn
		arg[td].ind=ind;
		arg[td].indsz=indsz;
		arg[td].imsz=imsz;
		arg[td].cutb=batches/num_threads;
		arg[td].cutbsz=arg[td].cutb*imsz;
		arg[td].cutcsz=arg[td].cutb*indsz;
		arg[td].td=td;
		arg[td].m=m;
		arg[td].n=n;
	}
	int rem=batches%num_threads;
	if(rem){
		arg[num_threads-1].cutb+=rem;
	}
	void *func;
	if(order=='F'){
		func = take_thrd_F;
	}
	else{
		func = take_thrd_C;
	}
	for (int td = 0; td < num_threads; ++td)
	{
		pthread_create(&threads[td],NULL,(void*)func,&arg[td]);
	}
	for (int td = 0; td < num_threads; ++td)
	{
		pthread_join(threads[td],NULL);
	}
	free(arg);
	return 0;
}