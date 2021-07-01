__kernel void Mat_multi(__global float *A, __global float *B, __global float *C,
const int Ndim, const int Mdim)//,const int N)
{
	//int N = 40;
	int j, k;
	int i = get_global_id(0);
	j = get_global_id(1);
	for (k = 0; k < Ndim; k++)
	{
		C[i*Ndim+j] += A[i*Ndim+k] * B[k * Ndim+j];
	}
//printf("Hello\n%d");?????????if this is added, the result can't be printed out.......
}


