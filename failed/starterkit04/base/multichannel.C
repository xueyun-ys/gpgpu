#include <cmath>
#include <omp.h>
#include "imgproc.h"
#include "CmdLineFind.h"
#include <vector>
#include "multichannel.h"
//#include "fftanalysis.h"

#include <GL/gl.h>   // OpenGL itself.
#include <GL/glu.h>  // GLU support library.
#include <GL/glut.h> // GLUT support library.


#include <iostream>
#include <stack>


using namespace std;
using namespace img;

// __global__ void kernel_coherent_two_channel_estimate_fft(const ImgProc& input,
//   int channel0, int channel1, float weight0, float weight1, ImgProc& output)
// {
//   // map from blockIdx to pixel position
//   int x = blockIdx.x;
//   int y = blockIdx.y;
//   int offset = x + y * gridDim.x;
//
//   FFTImgProc fftinput;
//   load_fft_new(input, fftinput);
//   fftinput.fft_forward();
//
//   FFTImgProc psd00 = fftinput;//sigma
//   FFTImgProc psd01 = fftinput;
//   FFTImgProc psd11 = fftinput;
//
//   // double ave0 = 0;
//   // double ave1 = 0;
//   // double sigma00 = 0;
//   // double sigma01 = 0;
//   // double sigma11 = 0;
//
// //unsmoothhhed psds
//   //for(int j=0;j<fftinput.ny();j++)
//   //{
//   //  #pragma omp parallel for
//     //for(int i=0;i<fftinput.nx();i++)
//   //  {
//       //std::vector<float> value;
//       std::vector<std::complex<double> > intilde;
//       //input.value(i, j, value);
//       fftinput.value(i, j, intilde);
//       std::complex<double> f11 = std::conj(intilde[channel1]) * intilde[channel1];
//       std::complex<double> f01 = std::conj(intilde[channel0]) * intilde[channel1];
//       std::complex<double> f00 = std::conj(intilde[channel0]) * intilde[channel0];
//       //ave0 += value[channel0];
//       //ave1 += value[channel1];
//       intilde[0] = f11;
//       intilde[1] = std::complex<double>(0, 0);
//       intilde[2] = std::complex<double>(0, 0);
//
//       psd11.set_value(i, j, intilde);
//       intilde[0]=f01;
//       psd01.set_value(i, j, intilde);
//       intilde[0]=f00;
//       psd00.set_value(i, j, intilde);
//     }
//   }
//   // ave0 /= input.nx()*input.ny();
//   // ave1 /= input.nx()*input.ny();
//   std::cout << "Raw PSDs done" << std::endl;
//
//   //smoothing the psds. Putting smoothed result into channel 1
//   //=================================================================================
//   int smooth_width = 5;
//   for(int j=0;j<fftinput.ny();j++)
//   {
//     #pragma omp parallel for
//     for(int i=0;i<fftinput.nx();i++)
//     {
//       std::complex<double> smoothed01(0.0, 0.0);
//       std::complex<double> smoothed11(0.0, 0.0);
//       std::complex<double> smoothed00(0.0, 0.0);
//       for(int jj=j-smooth_width;jj<=j+smooth_width;jj++)
//       {
//         int jjj = jj;
//         if(jjj<0){jjj+=fftinput.ny();}
//         if(jjj>=fftinput.ny()){jjj -= fftinput.ny();}
//
//         for(int ii=i-smooth_width;ii<=i+smooth_width;ii++)
//         {
//           int iii = ii;
//           if(iii<0){iii+=fftinput.nx();}
//           if(iii >= fftinput.nx()){iii-=fftinput.nx();}
//           std::vector<std::complex<double> >value;
//           psd01.value(iii, jjj, value);
//           smoothed01+=value[0];
//           psd11.value(iii,jjj, value);
//           smoothed11+=value[0];
//           psd00.value(iii,jjj, value);
//           smoothed00+=value[0];
//         }
//       }
//       smoothed01 /= (smooth_width*smooth_width);
//       smoothed11 /= (smooth_width*smooth_width);
//       smoothed00 /= (smooth_width*smooth_width);
//
//       std::vector<std::complex<double> > smoothvalue;
//
//       psd01.value(i, j, smoothvalue);
//       smoothvalue[1] = smoothed01;
//       psd01.set_value(i, j, smoothvalue);
//
//       psd11.value(i, j, smoothvalue);
//       smoothvalue[1] = smoothed11;
//       psd11.set_value(i, j, smoothvalue);
//
//       psd00.value(i, j, smoothvalue);
//       smoothvalue[1] = smoothed00;
//       psd00.set_value(i, j, smoothvalue);
//     }
//   }
//   std::cout << "Smoothed PSDs done" << std::endl;
//   //=================================================================================
//   //--------------------------------------------------------------------------------
//   std::complex<double> w0(weight0,0.0);
//   std::complex<double> w1(weight1,0.0);
//
//   for(int j=0;j<fftinput.ny();j++)
//   {
//     #pragma omp parallel for
//     for(int i=0;i<fftinput.nx();i++)
//     {
//       std::vector<std::complex<double> > value;//data
//       std::vector<std::complex<double> > p01;
//       std::vector<std::complex<double> > p11;
//       std::vector<std::complex<double> > p00;
//       fftinput.value(i, j, value);//data);
//       psd01.value(i, j, p01);//sigma01
//       psd11.value(i, j, p11);
//       psd00.value(i, j, p00);
//
//       std::complex<double> denominator = p11[1]*w0*w0
//       -(p01[1]+std::conj(p01[1]))*w0*w1
//       +p00[1]*w1*w1;
//
//       // std::complex<double> cs = (double)weight0*data[channel0]*p11[1]
//       //                           -(double)weight0*data[channel1]*p01[1]
//       //                           -(double)weight1*data[channel0]*p01[1]
//       //                           +(double)weight1*data[channel1]*p00[1];
//       std::complex<double> cs = value[channel0]*p11[1]*w0
//                                 -value[channel1]*p01[1]*w0
//                                 -value[channel0]*std::conj(p01[1])*w1
//                                 +value[channel1]*p00[1]*w1;
//       //data[channel0] - (std::conj(p01[1])/p11[1])*data[channel1];
//       cs /= denominator;
//       for(size_t c=0; c<(size_t)fftinput.depth();c++)
//       {
//         //data[c] = cs;
//         value[c] = cs;
//       }
//       //fftinput.set_value(i, j, data);
//       fftinput.set_value(i, j, value);
//     }
//   }
//
//   fftinput.fft_backward();
//   std::cout<<"FFT CE done"<<std::endl;
//   //float determinant = sigma00*sigma11 - sigma01*sigma01;
//
//   //float denominator = weight0*weight0*sigma11 -2.0*weight0*weight1*sigma01 + weight1*weight1*sigma11;
//   //denominator = denominator/determinant;
//   //----------------------------------------------------------------------------
//
//   //==============================================================
//   output = input;
//
//   for(int j=0;j<output.ny();j++)
//   {
//     #pragma omp parallel for
//     for(int i=0;i<output.nx();i++)
//     {
//       //std::vector<float> value;
//       std::vector<std::complex<double> > fftdata;
//       //input.value(i, j, value);
//       fftinput.value(i, j, fftdata);
//       // float coherent_estimate = weight0*(value[channel0]-ave0)*psd11
//       //                           -weight0*(value[channel1]-ave1)*psd01
//       //                           -weight1*(value[channel0]-ave0)*psd01
//       //                           +weight1*(value[channel1]-ave1)*psd00;
//       // float coherent_estimate = weight0*(value[channel0]-ave0)*sigma11
//       //                           -weight0*(value[channel1]-ave1)*sigma01
//       //                           -weight1*(value[channel0]-ave0)*sigma01
//       //                           +weight1*(value[channel1]-ave1)*sigma00;
//       std::vector<float> data(output.depth(),0.0);
//       for(size_t c=0;c<(size_t)output.depth();c++)
//       {
//         data[c] = fftdata[c].real();
//       }
//       output.set_value(i,j,data);
//     }
//   }
//   //==============================================================
// }

__global__ void kernel_coherent_two_channel_estimate(const ImgProc& input, int channel0,
   int channel1, float weight0, float weight1, ImgProc& output)
 {
   int x = blockIdx.x;
   int y = blockIdx.y;
   int offset = x + y * gridDim.x;

   double ave0 = 0;
   double ave1 = 0;
   double sigma00 = 0;
   double sigma01 = 0;
   double sigma11 = 0;

   //unsmoothhhed psds
   std::vector<float> value;
   input.value(x, y, value);
   ave0 += value[channel0];
   ave1 += value[channel1];
   __syncthreads();

   ave0 /= input.nx()*input.ny();
   ave1 /= input.nx()*input.ny();

   //--------------------------------------------------------------------------------

   std::vector<float> value;
   input.value(x, y, value);
   sigma00 += (value[channel0]-ave0)*(value[channel0]*ave0);
   sigma01 += (value[channel0]-ave0)*(value[channel1]*ave1);
   sigma11 += (value[channel1]-ave1)*(value[channel1]*ave1);
   __syncthreads();

   sigma00 /= input.nx()*input.ny();
   sigma01 /= input.nx()*input.ny();
   sigma11 /= input.nx()*input.ny();

   //==============================================================
   output = input;

   std::vector<float> value;
   input.value(x, y, value);
   float coherent_estimate = weight0*(value[channel0]-ave0)*sigma11
                             -weight0*(value[channel1]-ave1)*sigma01
                             -weight1*(value[channel0]-ave0)*sigma01
                             +weight1*(value[channel1]-ave1)*sigma00;
   for(size_t c=0;c<value.size();c++)
   {
     value[c] = coherent_estimate;
   }
   output.set_value(x,y,value);
   __syncthreads();
   //==============================================================
 }
