//nvcc -o hello 6780_hmw01.cu
#include "stdio.h"
//#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include "device_launch_parameters.h"
#include <memory>
#include "cuda_runtime.h"
#define imin(a, b) (a<b?a:b)

using namespace std;

//float c, d_c;

float *a, *b;// host copies of a, b, c
double c = 0.0;
float *pc;
float *d_a, *d_b;// device copies of a, b, c
float *d_pc;

float vec_a[1<<24];
float vec_b[1<<24];
const int num = 1<<24;
const int threadPerBlock = 512;
const int blocksPerGrid = imin(32, (num+threadPerBlock-1)/threadPerBlock);
float size = num * sizeof(float);
//float size2 = 1 * sizeof(float);
double r = 0.0;
//float *res = &r;

void random_floats(float *x, int size)
{
  //if(size<1) die("Number of elements must be greater than zero");
  float flt = (float)(rand()%101)/101;
  srand((unsigned)time(NULL));
  for (int i = 0; i < size; i++)
  {
    //x[i] = rand()%100+flt;
    x[i] = (float)rand()/(float)rand();
  }
}

// void random_create()
// {
//   float flt = (float)(rand()%10)/10;
//   srand((unsigned)time(NULL));
//   for(int i = 0; i < 1024; i++)
//   {
//     vec_a[i] = rand()%100+flt;
//     vec_b[i] = rand()%100+flt;
//   }
// }

void *CPU_big_dot(float *A, float *B, int N)
{
  r= 0.0;
  for(int i=0; i<N; i++)
  {
    r += A[i] * B[i];
  }
  //return &r;
}

//--------------------------------kernel 1----------------------------------------
__global__ void GPU_big_dot_kernel(float *A, float *B, float *c)
{
  //every block has a copy of cache, and they are independently(can't affected by each other)
  __shared__ float cache[threadPerBlock];
  //N[blockIdx.x] = A[blockIdx.x] * B[blockIdx.x];
  //offset
  int step = threadIdx.x + blockIdx.x*blockDim.x;//blockdim = threads in one block
  float temp_dot = 0;
  int cacheIdx = threadIdx.x;
  while (step<num)
  {
    temp_dot += A[step]*B[step];
    //num of running threads
    step+=blockDim.x*gridDim.x;
  }
  cache[cacheIdx]=temp_dot;
  //sychronol threads
  __syncthreads();
  //N[threadIdx.x] += temp_dot;
  int i = blockDim.x/2;
  //sum in each block
  //interleaved pair
  while(i>0)
  {
    if(cacheIdx<i)
    {
      cache[cacheIdx]+=cache[cacheIdx+i];
    }
    __syncthreads();
    i /= 2;
  }

  if(cacheIdx == 0)
    c[blockIdx.x]=cache[0];
}
//----------------------------------------------------------------------------------

//struct lock
struct Lock {
  int *mutex;
  Lock( void ) {
    int state = 0;
    // HANDLE_ERROR(cudaMalloc((VOID**)&mutex, sizeof(int)));
    // HANDLE_ERROR(cudaMemcpy(mutex, &state, sizeof(int),cudaMemcpyHostToDevice));
    cudaMalloc( (void**)& mutex, sizeof(int) );
    cudaMemcpy( mutex, &state, sizeof(int), cudaMemcpyHostToDevice );
  }

  ~Lock( void ) {
    cudaFree( mutex );
  }

  __device__ void lock( void ) {
    while( atomicCAS( mutex, 0, 1 ) != 0 );
  }

  __device__ void unlock( void ) {
   atomicExch( mutex, 0 );
  }
};


//------------------------------kernel 2----------------------------------------
__global__ void GPU_big_dot_kernel_atom(float *A, float *B, float *c, Lock lock)
{
  //every block has a copy of cache, and they are independently(can't affected by each other)
  __shared__ float cache[threadPerBlock];
  //offset
  int step = threadIdx.x + blockIdx.x*blockDim.x;//blockdim = threads in one block
  float temp_dot = 0;
  int cacheIdx = threadIdx.x;
  while (step<num)
  {
    temp_dot += A[step]*B[step];
    //num of running threads
    step+=blockDim.x*gridDim.x;
  }
  cache[cacheIdx]=temp_dot;
  //sychronol threads
  __syncthreads();
  int i = blockDim.x/2;
  //sum in each block
  //interleaved pair
  while(i>0)
  {
    if(cacheIdx<i)
    {
      cache[cacheIdx]+=cache[cacheIdx+i];
    }
    __syncthreads();
    i /= 2;
  }

  if(cacheIdx == 0)
  {
    lock.lock();
    c[0]+=cache[0];
    //atomicAdd(c[0], cache[0]);
    lock.unlock();
  }
}
//----------------------------------------------------------------------------------

//=======================================================
int main( void )
{
    Lock lock;
    //creaete 2 events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //initialize
    r=0.0;

    a = (float *) malloc(size);
    b = (float *) malloc(size);
    pc = (float *)malloc(blocksPerGrid*sizeof(float));
    random_floats(a, num);
    random_floats(b, num);
    //random_create();
    //CPU_big_dot(vec_a, vec_b, num);

    //============================GPU====================================
    // Allocate space for device copies of a, b, c
    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_pc, blocksPerGrid*sizeof(float));//partial sum
    //
    // Alloc space for host copies of a, b, c and setup input values
    // a = (float *) malloc(size);
    // b = (float *) malloc(size);
    // random_floats(a, num);
    // random_floats(b, num);
    // for(int i = 0;i<10;i++)
    // {
    //   cout <<"vector:"<< a[i] << '\t'<<endl;
    // }

    // for(int i = 0; i<num; i++)
    // {
    //   //test += c[i];
    // }
    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    //record start event on the default stream
    cudaEventRecord(start);//, 0);
    // Launch dot() kernel on GPU with N blocks
    GPU_big_dot_kernel_atom<<<blocksPerGrid,threadPerBlock>>>(d_a, d_b, d_pc,lock);
    cudaMemcpy(pc, d_pc, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
    c = pc[0];
    //record stop event on the default stream
    cudaEventRecord(stop);//, 0);
    cout <<"GPU_K2(atomic)_result:"<< c << '\t'<<endl;
    //wait until the stop event completes
    cudaEventSynchronize(stop);
    //calculate the elapsed time between two events
    float time;
    cudaEventElapsedTime(&time, start, stop);
    //clean up the two events
    //cudaEventDestroy(start);
    //cudaEventDestroy(stop);
    cout <<"Kernel2_time(atomic):"<< time << '\t'<<endl;

    cudaEventRecord(start);
    GPU_big_dot_kernel<<<blocksPerGrid,threadPerBlock>>>(d_a, d_b, d_pc);
    // Copy result back to host
    cudaMemcpy(pc, d_pc, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
    // float test = 0.0;

    c=0.0;
    for(int i = 0;i<blocksPerGrid;i++)
    {
      c+=pc[i];
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);



    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //============================GPU====================================
    //===========================CPU====================================
    // for(int i = 0;i<10;i++)
    // {
    //   cout <<"vector2:"<< a[i] << '\t'<<endl;
    // }
    CPU_big_dot(a, b, num);
    //===========================CPU====================================
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_pc);

    //float speedup = ((float)timing2/(1000*1000)) /((float)timing3/(1000*1000));
    //cout <<"Speedup:"<< speedup << '\t'<<endl;
    cout <<"GPU_K1_result:"<< c << '\t'<<endl;
    cout <<"Kernel1_time:"<< time << '\t'<<endl;
    cout <<"CPU_result:"<< r << '\t'<<endl;
    //cout <<"result_corectness:"<< c - r << '\t'<<endl;
    return 0;
}

//=============================================================================
