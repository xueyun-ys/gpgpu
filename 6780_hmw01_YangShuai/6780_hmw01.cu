#include "stdio.h"
//#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include "device_launch_parameters.h"
#include <memory>
#include "cuda_runtime.h"

using namespace std;

//long n = 1024*1024;
//#define N 1024*1024; wrong

//float c, d_c;

float *a, *b;// host copies of a, b, c
float c = 0.0;
float *pc;
float *d_a, *d_b;// device copies of a, b, c
float *d_pc;

float vec_a[1024*1024];
float vec_b[1024*1024];
const int num = 1024*1024;
const int threadPerBlock = 512;
const int blocksPerGrid = imin(32, (num+threadPerBlock-1));
//const int blocksPerGrid = 256;
float size = num * sizeof(float);
float size2 = 1 * sizeof(float);
float r = 0.0;
float *res = &r;
long long timing1;
long long timing2;
long long timing3;

void random_floats(float *x, int size)
{
  float flt = (float)(rand()%101)/101;
  srand((unsigned)time(NULL));
  for (int i = 0; i < size; i++)
  {
    x[i] = rand()%100+flt;
  }
}

void random_create()
{
  float flt = (float)(rand()%10)/10;
  srand((unsigned)time(NULL));
  for(long i = 0; i < 1024*1024; i++)
  {
    vec_a[i] = rand()%100+flt;
    vec_b[i] = rand()%100+flt;
    // if (i<100)
     //cout << vec_a[i] << '\t';
  }
  //cout<<endl;
}

//==============Timer functions================
long long start_timer()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec*1000000 + tv.tv_usec;
}

long long stop_timer(long long start_time, char *name)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
  printf("%s: %.5f sec\n", name, ((float)(end_time-start_time))/(1000*1000));
  return end_time-start_time;
}
//==============Timer functions================

float *CPU_big_dot(float *A, float *B, int N)
{
  //timing1 = start_timer();
  r= 0.0;
  for(int i=0; i<N; i++)
  {
    r += A[i] * B[i];
  }
  //timing2 = stop_timer(timing1, "CPU_time");
  return &r;
}

// __global__ float *GPU_big_dot_kernel(float *A, float *B, int N)
// __global__ void GPU_big_dot_kernel(float *A, float *B, float N)
// {
//   int a = threadIdx.y;
//   int b = threadIdx.x;
//   N += A[a] * B[b];
// }

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
  while(i!=0)
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


int main( void )
{
    //float ttest[256];

    r=0.0;
    a = (float *) malloc(size);
    b = (float *) malloc(size);
    random_floats(a, num);
    random_floats(b, num);
    //random_create();
    //CPU_big_dot(vec_a, vec_b, num);

    //============================GPU====================================
    timing1 = start_timer();
    // Allocate space for device copies of a, b, c
    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_pc, blocksPerGrid*sizeof(float));
    //
    // Alloc space for host copies of a, b, c and setup input values
    // a = (float *) malloc(size);
    // b = (float *) malloc(size);
    pc = (float *)malloc(blocksPerGrid*sizeof(float));
    //timing3 = stop_timer(timing1, "temp");
    // random_floats(a, num);
    // random_floats(b, num);
    // for(int i = 0;i<10;i++)
    // {
    //   cout <<"vector:"<< a[i] << '\t'<<endl;
    // }
    //timing1 = start_timer();

    // for(int i = 0; i<num; i++)
    // {
    //   //test += c[i];
    // }
    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch dot() kernel on GPU with N blocks
    GPU_big_dot_kernel<<<blocksPerGrid,threadPerBlock>>>(d_a, d_b, d_pc);
    // Copy result back to host
    cudaMemcpy(pc, d_pc, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
    // float test = 0.0;

    c=0.0;
    for(int i = 0;i<blocksPerGrid;i++)
    {
      c+=pc[i];
    }
    //cout <<"GPU_result:"<< c << '\t'<<endl;
    timing3 += stop_timer(timing1, "GPU_time");
    //cout <<"Speed_test1:"<< (float)timing3/(1000*1000) << '\t'<<endl;
    //============================GPU====================================
    //===========================CPU====================================
    // for(int i = 0;i<10;i++)
    // {
    //   cout <<"vector2:"<< a[i] << '\t'<<endl;
    // }
    timing1 = start_timer();
    CPU_big_dot(a, b, num);
    timing2 = stop_timer(timing1, "CPU_time");
    //===========================CPU====================================
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_pc);

    float speedup = ((float)timing2/(1000*1000)) /((float)timing3/(1000*1000));
    //cout <<"Speed_test2:"<< (float)timing3/(1000*1000) << '\t'<<endl;
    cout <<"Speedup:"<< speedup << '\t'<<endl;
    cout <<"GPU_result:"<< c << '\t'<<endl;
    cout <<"CPU_result:"<< r << '\t'<<endl;
    cout <<"result_corectness:"<< c - r << '\t'<<endl;
    return 0;
}

//=============================================================================





















//===============================================================================================================================================================
// #include "stdio.h"
// //#include <stdio.h>
// #include <time.h>
// #include <sys/time.h>
// #include <stdlib.h>
// #include <iostream>
//
// using namespace std;
//
// //long n = 1024*1024;
// #define N 1024*1024;
// // float size = N * sizeof(float);
// // float size2 = 1 * sizeof(float);
// //
// // float *a, *b, *c; // host copies of a, b, c
// // //float c, d_c;
// // float *d_a, *d_b, *d_c; // device copies of a, b, c
//
//
// float vec_a[1024*1024];
// float vec_b[1024*1024];
// int num = 1024*1024;
// float r = 0.0;
// float *res = &r;
// long long timing1;
// long long timing2;
//
// void random_floats(float *x, int size)
// {
//   float flt = (float)(rand()%10)/10;
//   srand((unsigned)time(NULL));
//   for (int i = 0; i < size; i++)
//   {
//     x[i] = rand()%100+flt;
//   }
// }
//
// void random_create()//(vec, size)
// {
//   float flt = (float)(rand()%10)/10;
//   srand((unsigned)time(NULL));
//   for(long i = 0; i < 1024*1024; i++)
//   {
//     vec_a[i] = rand()%100+flt;
//     vec_b[i] = rand()%100+flt;
//     // if (i<100)
//      //cout << vec_a[i] << '\t';
//   }
//   //cout<<endl;
// }
//
// //Timer functions
// long long start_timer()
// {
//   struct timeval tv;
//   gettimeofday(&tv, NULL);
//   return tv.tv_sec*1000000 + tv.tv_usec;
// }
//
// long long stop_timer(long long start_time, char *name)
// {
//   struct timeval tv;
//   gettimeofday(&tv, NULL);
//   long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
//   printf("%s: %.5f sec\n", name, ((float)(end_time-start_time))/(1000*1000));
//   return end_time-start_time;
// }
//
//
// float *CPU_big_dot(float *A, float *B, int N)
// {
//   timing1 = start_timer();
//   //r= 0.0;
//   for(int i=0; i<N; i++)
//   {
//     r += A[i] * B[i];
//   }
//   timing2 = stop_timer(timing1, "CPU_time");
//   return &r;
// }
//
// // __global__ float *GPU_big_dot_kernel(float *A, float *B, int N)
// // __global__ void GPU_big_dot_kernel(float *A, float *B, float N)
// // {
// //   int a = threadIdx.y;
// //   int b = threadIdx.x;
// //   N += A[a] * B[b];
// // }
//
// __global__ void GPU_big_dot_kernel(float *A, float *B, float *N)
// {
//   N[0] += A[blockIdx.x] * B[blockIdx.x];
// }
//
//
// int main( void )
// {
//     float size = N * sizeof(float);
//     float size2 = 1 * sizeof(float);
//
//     float *a, *b, *c;
//     float *d_a, *d_b, *d_c;
//     r=0.0;
//     random_create();
//
//     CPU_big_dot(vec_a, vec_b, num);
//
//     //===========================================================
//     // Allocate space for device copies of a, b, c
//     cudaMalloc((void **) &d_a, size);
//     cudaMalloc((void **) &d_b, size);
//     cudaMalloc((void **) &d_c, size2);
//
//     // Alloc space for host copies of a, b, c and setup input values
//     a = (float *) malloc(size);
//     random_floats(a, N);
//     b = (float *) malloc(size);
//     random_floats(b, N);
//     c = (float *)malloc(size2);
//
//     // Copy inputs to device
//     cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
//     //===========================================================
//     // Launch dot() kernel on GPU with N blocks
//     GPU_big_dot_kernel<<<N,1>>>(d_a, d_b, d_c);
//     // Copy result back to host
//     cudaMemcpy(c, d_c, size2, cudaMemcpyDeviceToHost);
//
//     // Cleanup
//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_c);
//     //cout << c << '\t'<<endl;
//     //printf( "Hello, World!\n" );
//     return 0;
// }
//
// //=============================================================================



// #include "stdio.h"
// //#include <stdio.h>
// #include <time.h>
// #include <sys/time.h>
// #include <stdlib.h>
// #include <iostream>
// #include "device_launch_parameters.h"
// #include <memory>
// #include "cuda_runtime.h"
//
// using namespace std;
//
// //long n = 1024*1024;
// //#define N 1024*1024; wrong
//
// //float c, d_c;
//
// float *a, *b, c;// host copies of a, b, c
// float *pc;
// float *d_a, *d_b;// device copies of a, b, c
// float *d_pc;
//
// float vec_a[1024*1024];
// float vec_b[1024*1024];
// const int num = 1024*1024;
// const int threadPerBlock = 256;
// const int blocksPerGrid = 1024;
// float size = num * sizeof(float);
// float size2 = 1 * sizeof(float);
// float r = 0.0;
// float *res = &r;
// long long timing1;
// long long timing2;
// long long timing3;
//
// void random_floats(float *x, int size)
// {
//   float flt = (float)(rand()%101)/101;
//   srand((unsigned)time(NULL));
//   for (int i = 0; i < size; i++)
//   {
//     x[i] = rand()%100+flt;
//   }
// }
//
// void random_create()
// {
//   float flt = (float)(rand()%10)/10;
//   srand((unsigned)time(NULL));
//   for(long i = 0; i < 1024*1024; i++)
//   {
//     vec_a[i] = rand()%100+flt;
//     vec_b[i] = rand()%100+flt;
//     // if (i<100)
//      //cout << vec_a[i] << '\t';
//   }
//   //cout<<endl;
// }
//
// //==============Timer functions================
// long long start_timer()
// {
//   struct timeval tv;
//   gettimeofday(&tv, NULL);
//   return tv.tv_sec*1000000 + tv.tv_usec;
// }
//
// long long stop_timer(long long start_time, char *name)
// {
//   struct timeval tv;
//   gettimeofday(&tv, NULL);
//   long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
//   printf("%s: %.5f sec\n", name, ((float)(end_time-start_time))/(1000*1000));
//   return end_time-start_time;
// }
// //==============Timer functions================
//
// float *CPU_big_dot(float *A, float *B, int N)
// {
//   //timing1 = start_timer();
//   r= 0.0;
//   for(int i=0; i<N; i++)
//   {
//     r += A[i] * B[i];
//   }
//   //timing2 = stop_timer(timing1, "CPU_time");
//   return &r;
// }
//
// // __global__ float *GPU_big_dot_kernel(float *A, float *B, int N)
// // __global__ void GPU_big_dot_kernel(float *A, float *B, float N)
// // {
// //   int a = threadIdx.y;
// //   int b = threadIdx.x;
// //   N += A[a] * B[b];
// // }
//
// __global__ void GPU_big_dot_kernel(float *A, float *B, float *c)
// {
//   //every block has a copy of cache, and they are independently(can't affected by each other)
//   __shared__ float cache[threadPerBlock];
//   //N[blockIdx.x] = A[blockIdx.x] * B[blockIdx.x];
//   //offset
//   int step = threadIdx.x + blockIdx.x*blockDim.x;//blockdim = threads in one block
//   float temp_dot = 0;
//   int cacheIdx = threadIdx.x;
//   while (step<num)
//   {
//     temp_dot += A[step]*B[step];
//     //num of running threads
//     step+=blockDim.x*gridDim.x;
//   }
//   cache[cacheIdx]=temp_dot;
//   //sychronol threads
//   __syncthreads();
//   //N[threadIdx.x] += temp_dot;
//   int i = blockDim.x/2;
//   //sum in each block
//   while(i!=0)
//   {
//     if(cacheIdx<i)
//     {
//       cache[cacheIdx]+=cache[cacheIdx+i];
//     }
//     __syncthreads();
//     i /= 2;
//   }
//
//   if(cacheIdx == 0)
//     c[blockIdx.x]=cache[0];
// }
//
//
// int main( void )
// {
//     //float ttest[256];
//
//     r=0.0;
//     //random_create();
//     //CPU_big_dot(vec_a, vec_b, num);
//
//     //============================GPU====================================
//     timing1 = start_timer();
//     // Allocate space for device copies of a, b, c
//     cudaMalloc((void **) &d_a, size);
//     cudaMalloc((void **) &d_b, size);
//     cudaMalloc((void **) &d_pc, blocksPerGrid*sizeof(float));
//     //
//     // Alloc space for host copies of a, b, c and setup input values
//     a = (float *) malloc(size);
//     b = (float *) malloc(size);
//     pc = (float *)malloc(blocksPerGrid*sizeof(float));
//     timing3 = stop_timer(timing1, "temp");
//     random_floats(a, num);
//     random_floats(b, num);
//     // for(int i = 0;i<10;i++)
//     // {
//     //   cout <<"vector:"<< a[i] << '\t'<<endl;
//     // }
//     timing1 = start_timer();
//
//     // for(int i = 0; i<num; i++)
//     // {
//     //   //test += c[i];
//     // }
//     // Copy inputs to device
//     cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
//
//     // Launch dot() kernel on GPU with N blocks
//     GPU_big_dot_kernel<<<blocksPerGrid,threadPerBlock>>>(d_a, d_b, d_pc);
//     // Copy result back to host
//     cudaMemcpy(pc, d_pc, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
//     // float test = 0.0;
//
//     c=0.0;
//     for(int i = 0;i<blocksPerGrid;i++)
//     {
//       c+=pc[i];
//     }
//     //cout <<"GPU_result:"<< c << '\t'<<endl;
//     timing3 += stop_timer(timing1, "GPU_time");
//     //============================GPU====================================
//     //===========================CPU====================================
//     // for(int i = 0;i<10;i++)
//     // {
//     //   cout <<"vector2:"<< a[i] << '\t'<<endl;
//     // }
//     timing1 = start_timer();
//     CPU_big_dot(a, b, num);
//     timing2 = stop_timer(timing1, "CPU_time");
//     //===========================CPU====================================
//     // Cleanup
//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_pc);
//
//     double speedup = (float)timing2/(1000*1000) /(float)timing3/(1000*1000);
//     cout <<"Speedup:"<< speedup << '\t'<<endl;
//     cout <<"GPU_result:"<< c << '\t'<<endl;
//     cout <<"CPU_result:"<< r << '\t'<<endl;
//     cout <<"result_corectness:"<< c - r << '\t'<<endl;
//     return 0;
// }
//
// //=============================================================================







// #include "stdio.h"
// //#include <stdio.h>
// #include <time.h>
// #include <sys/time.h>
// #include <stdlib.h>
// #include <iostream>
// #include "device_launch_parameters.h"
// #include <memory>
// #include "cuda_runtime.h"
//
// using namespace std;
//
// //long n = 1024*1024;
// //#define N 1024*1024; wrong
//
// //float c, d_c;
//
// float *a, *b, c;// host copies of a, b, c
// float *pc;
// float *d_a, *d_b;// device copies of a, b, c
// float *d_pc;
//
// float vec_a[1024*1024];
// float vec_b[1024*1024];
// const int num = 1024*1024;
// const int threadPerBlock = 256;
// const int blocksPerGrid = 1024;
// float size = num * sizeof(float);
// float size2 = 1 * sizeof(float);
// float r = 0.0;
// float *res = &r;
// long long timing1=0;
// long long timing2=0;
// long long timing3=0;
//
// void random_floats(float *x, int size)
// {
//   float flt = (float)(rand()%101)/101;
//   srand((unsigned)time(NULL));
//   for (int i = 0; i < size; i++)
//   {
//     x[i] = rand()%100+flt;
//   }
// }
//
// void random_create()
// {
//   float flt = (float)(rand()%10)/10;
//   srand((unsigned)time(NULL));
//   for(long i = 0; i < 1024*1024; i++)
//   {
//     vec_a[i] = rand()%100+flt;
//     vec_b[i] = rand()%100+flt;
//     // if (i<100)
//      //cout << vec_a[i] << '\t';
//   }
//   //cout<<endl;
// }
//
// //==============Timer functions================
// long long start_timer()
// {
//   struct timeval tv;
//   gettimeofday(&tv, NULL);
//   return tv.tv_sec*1000000 + tv.tv_usec;
// }
//
// long long stop_timer(long long start_time, char *name)
// {
//   struct timeval tv;
//   gettimeofday(&tv, NULL);
//   long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
//   printf("%s: %.5f sec\n", name, ((float)(end_time-start_time))/(1000*1000));
//   return end_time-start_time;
// }
// //==============Timer functions================
//
// float *CPU_big_dot(float *A, float *B, int N)
// {
//   //timing1 = start_timer();
//   r= 0.0;
//   for(int i=0; i<N; i++)
//   {
//     r += A[i] * B[i];
//   }
//   //timing2 = stop_timer(timing1, "CPU_time");
//   return &r;
// }
//
// // __global__ float *GPU_big_dot_kernel(float *A, float *B, int N)
// // __global__ void GPU_big_dot_kernel(float *A, float *B, float N)
// // {
// //   int a = threadIdx.y;
// //   int b = threadIdx.x;
// //   N += A[a] * B[b];
// // }
//
// __global__ void GPU_big_dot_kernel(float *A, float *B, float *c)
// {
//   //every block has a copy of cache, and they are independently(can't affected by each other)
//   __shared__ float cache[threadPerBlock];
//   //N[blockIdx.x] = A[blockIdx.x] * B[blockIdx.x];
//   //offset
//   int step = threadIdx.x + blockIdx.x*blockDim.x;//blockdim = threads in one block
//   float temp_dot = 0.0;
//   int cacheIdx = threadIdx.x;
//   while (step<num)
//   {
//     temp_dot += A[step]*B[step];
//     //num of running threads
//     step+=blockDim.x*gridDim.x;
//   }
//   cache[cacheIdx]=temp_dot;
//   //sychronol threads
//   __syncthreads();
//   //N[threadIdx.x] += temp_dot;
//   int i = blockDim.x/2;
//   //sum in each block
//   while(i!=0)
//   {
//     if(cacheIdx<i)
//     {
//       cache[cacheIdx]+=cache[cacheIdx+i];
//     }
//     __syncthreads();
//     i /= 2;
//   }
//   // for(int i=0;i<blockDim.x;i++)
//   // {
//   //   c[blockIdx.x]+=cache[i];
//   // }
//   if(cacheIdx == 0)
//     c[blockIdx.x]=cache[0];
// }
//
//
// int main( void )
// {
//     //float ttest[256];
//
//     r=0.0;
//     //random_create();
//     //CPU_big_dot(vec_a, vec_b, num);
//
//     //============================GPU====================================
//     timing1 = start_timer();
//     // Allocate space for device copies of a, b, c
//     cudaMalloc((void **) &d_a, size);
//     cudaMalloc((void **) &d_b, size);
//     cudaMalloc((void **) &d_pc, blocksPerGrid*sizeof(float));
//     //
//     // Alloc space for host copies of a, b, c and setup input values
//     a = (float *) malloc(size);
//     b = (float *) malloc(size);
//     pc = (float *)malloc(blocksPerGrid*sizeof(float));
//     timing3 = stop_timer(timing1, "temp");
//     random_floats(a, num);
//     random_floats(b, num);
//     // for(int i = 0;i<10;i++)
//     // {
//     //   cout <<"vector:"<< a[i] << '\t'<<endl;
//     // }
//     timing1 = start_timer();
//
//     // for(int i = 0; i<num; i++)
//     // {
//     //   //test += c[i];
//     // }
//     // Copy inputs to device
//     cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
//
//     // Launch dot() kernel on GPU with N blocks
//     GPU_big_dot_kernel<<<blocksPerGrid,threadPerBlock>>>(d_a, d_b, d_pc);
//     // Copy result back to host
//     cudaMemcpy(pc, d_pc, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
//     // float test = 0.0;
//
//     c=0.0;
//     for(int i = 0;i<blocksPerGrid;i++)
//     {
//       c+=pc[i];
//     }
//     //cout <<"GPU_result:"<< c << '\t'<<endl;
//     timing3 += stop_timer(timing1, "GPU_time");
//     //============================GPU====================================
//     //===========================CPU====================================
//     // for(int i = 0;i<10;i++)
//     // {
//     //   cout <<"vector2:"<< a[i] << '\t'<<endl;
//     // }
//     timing1 = start_timer();
//     CPU_big_dot(a, b, num);
//     timing2 = stop_timer(timing1, "CPU_time");
//     //===========================CPU====================================
//     // Cleanup
//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_pc);
//
//     double speedup = ((float)timing2/(1000*1000)) /((float)timing3/(1000*1000));
//     cout <<"Speedup:"<< speedup << '\t'<<endl;
//     cout <<"GPU_result:"<< c << '\t'<<endl;
//     cout <<"CPU_result:"<< r << '\t'<<endl;
//     cout <<"result_corectness:"<< c - r << '\t'<<endl;
//     return 0;
// }
