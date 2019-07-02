#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define num_threads 4

typedef struct arguments
{
	float *coled, *padded;
	int64_t *ind;
	int imsz, indsz, td, cutbsz, cutb, cutcsz;
}args;

void take_thrd(args *ptr){
	for(int bc = 0; bc < ptr->cutb; ++bc)
	{
		float *img=ptr->padded + ptr->imsz*bc + ptr->td*ptr->cutbsz;
		float *cole=ptr->coled + ptr->indsz*bc + ptr->td*ptr->cutcsz;
		for (int i = 0; i < ptr->indsz; ++i)
		{
			cole[i]=(float)img[ptr->ind[i]];
		}
	}
}

int take(float *padded,int64_t *ind,float *coled,int batches,int imsz, int indsz){
	pthread_t threads[num_threads];
	args *arg=malloc(num_threads*sizeof(args));
	for (int td = 0; td < num_threads; ++td)
	{
		arg[td].padded=padded;
		arg[td].coled=coled;
		arg[td].ind=ind;
		arg[td].indsz=indsz;
		arg[td].imsz=imsz;
		arg[td].cutb=batches/num_threads;
		arg[td].cutbsz=arg[td].cutb*imsz;
		arg[td].cutcsz=arg[td].cutb*indsz;
		arg[td].td=td;
	}
	int rem=batches%num_threads;
	if(rem){
		arg[num_threads-1].cutb+=rem;
	}
	for (int td = 0; td < num_threads; ++td)
	{
		pthread_create(&threads[td],NULL,(void*)take_thrd,&arg[td]);
	}
	for (int td = 0; td < num_threads; ++td)
	{
		pthread_join(threads[td],NULL);
	}
	free(arg);
	return 0;
}