

#ifndef MULTICHANNEl_H
#define multichannel_H

#include <string>
#include <vector>

#include <imgproc.h>
#include <fftimgproc.h>


namespace img
{
void coherent_subtraction_fft(const ImgProc& input, int channel0, int channel1, ImgProc& output);
void coherent_two_channel_estimate_fft(const ImgProc& input, int channel0, int channel1, float
weight0, float weight1, ImgProc& output);
void coherent_two_channel_estimate(const ImgProc& input, int channel0, int channel1, float weight0,
  float weight1, ImgProc& output);

std::complex<double> det(const std::vector<std::complex<double> >& m);
std::complex<double> cofactor(int i, int j, const std::vector<std::complex<double> >& m);

}


#endif
