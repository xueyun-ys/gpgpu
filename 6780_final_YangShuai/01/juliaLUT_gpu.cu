/*references:
 * NVIDIA Corporation code: Julia.
 Letures and notes from 6780 GPGPU & 6040 CGI
 libraries: common files from class
 *
 */

// #include <cuda_runtime_api.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <math.h>
// #include <values.h>
#include "stdio.h"
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include "device_launch_parameters.h"
#include <memory>
#include "cuda_runtime.h"

#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1000

struct LUT {
    float colori=0.0;

    __device__ LUT( float a ) : colori(a)  {}
    __device__ float red( void )
    {
      	if (colori/2.0 <0.1)
      	    return 0.4;
      	else if (colori/2.0 <0.2)
      	    return 0.1;
      	else if (colori/2.0 <0.3)
      	    return 0.2;
      	else if (colori/2.0 <0.4)
      	    return 0.4;
      	else if (colori/2.0 <0.5)
      	    return 0.7;
      	else if (colori/2.0 <0.6)
      	    return 0.9;
      	else if (colori/2.0 <0.7)
      	    return 0.65;
      	else if (colori/2.0 <0.8)
      	    return 0.47;
      	else if (colori/2.0 <0.9)
      	    return 0.78;
      	else if (colori/2.0 <= 1.0)
      	    return 0.34;
    }
    __device__ float green( void )
    {
      	if (colori/2.0 <0.1)
      	    return 0.6;
      	else if (colori/2.0 <0.2)
      	    return 0.1;
      	else if (colori/2.0 <0.3)
      	    return 0.46;
      	else if (colori/2.0 <0.4)
      	    return 0.38;
      	else if (colori/2.0 <0.5)
      	    return 0.37;
      	else if (colori/2.0 <0.6)
      	    return 0.6;
      	else if (colori/2.0 <0.7)
      	    return 0.65;
      	else if (colori/2.0 <0.8)
      	    return 0.91;
      	else if (colori/2.0 <0.9)
      	    return 0.34;
      	else if (colori/2.0 <= 1.0)
      	    return 0.56;

    }
    __device__ float blue( void )
    {
      	if (colori/2.0 <0.1)
      	    return 0.8;
      	else if (colori/2.0 <0.2)
      	    return 0.23;
      	else if (colori/2.0 <0.3)
      	    return 0.2;
      	else if (colori/2.0 <0.4)
      	    return 0.62;
      	else if (colori/2.0 <0.5)
      	    return 0.32;
      	else if (colori/2.0 <0.6)
      	    return 0.51;
      	else if (colori/2.0 <0.7)
      	    return 0.13;
      	else if (colori/2.0 <0.8)
      	    return 0.93;
      	else if (colori/2.0 <0.9)
      	    return 0.46;
      	else if (colori/2.0 <= 1.0)
      	    return 0.76;
    }
};

struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __device__ float magnitude2( void ) { return r * r + i * i; }
    __device__ cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
    __device__ float getx()
    {
	     return r;
    }
    __device__ float gety()
    {
	     return i;
    }
};

__device__ float julia( int x, int y ) {
    const float scale = 1;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);
    //float res[3];
    //cuComplex c(-0.8, 0.156);
    cuComplex c(-0.82, 0.156);
    //cuComplex c(-0.65, 0.226);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 2)
            return 0;
    }
    LUT l(a.magnitude2());
    //res[0]= l.red();
    //res[1]= l.green();
    //res[2]= l.blue();
    return l.red()*255.0;
}

__device__ float julia2( int x, int y ) {
    const float scale = 1;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);
    //float res[3];
    cuComplex c(-0.8, 0.149);
    //cuComplex c(-0.65, 0.226);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 2)
            return 0;
    }
    LUT l(a.magnitude2());
    //res[0]= l.red();
    //res[1]= l.green();
    //res[2]= l.blue();
    return l.green()*255.0;
}

__device__ float julia3( int x, int y ) {
    const float scale = 1;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);
    //float res[3];
    cuComplex c(-0.8, 0.146);
    //cuComplex c(-0.65, 0.226);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 2)
            return 0;
    }
    LUT l(a.magnitude2());
    //res[0]= l.red();
    //res[1]= l.green();
    //res[2]= l.blue();
    return l.blue()*255.0;
}

__global__ void kernel( unsigned char *ptr ){
  // map from blockIdx to pixel position
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = x + y * gridDim.x;
	    //float f[3]= {julia( x, y )[0], julia( x, y )[1], julia( x, y )[2]};

  ptr[offset*4 + 0] = julia( x, y );
  ptr[offset*4 + 1] = julia2( x, y );
  ptr[offset*4 + 2] = julia3( x, y );
  ptr[offset*4 + 3] = 255;
}

 // globals needed by the update routine
 struct DataBlock {
     unsigned char   *dev_bitmap;
 };

int main( void ) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  DataBlock   data;
  CPUBitmap bitmap( DIM, DIM, &data );
  unsigned char    *dev_bitmap;

  HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
  data.dev_bitmap = dev_bitmap;

  dim3    grid(DIM,DIM);


  //record start event on the default stream
  cudaEventRecord(start);//, 0);
  // Launch dot() kernel on GPU with N blocks

  kernel<<<grid,1>>>( dev_bitmap );
  HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                            bitmap.image_size(),
                            cudaMemcpyDeviceToHost ) );
  HANDLE_ERROR( cudaFree( dev_bitmap ) );

  //record stop event on the default stream
  cudaEventRecord(stop);//, 0);
  //cout <<"GPU_result:"<< '\t'<<endl;
  //wait until the stop event completes
  cudaEventSynchronize(stop);
  //calculate the elapsed time between two events
  float time;
  cudaEventElapsedTime(&time, start, stop);


  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::cout <<""<<std::endl;
  std::cout <<"Kernel_time:           "<< time << '\t'<<std::endl;





  bitmap.display_and_exit();
}
