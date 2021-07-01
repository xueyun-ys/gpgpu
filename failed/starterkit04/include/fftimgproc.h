
#ifndef FFTIMGPROC_H
#define FFTIMGPROC_H

#include <vector>
#include <complex>
#include "fftw3.h"

#include <string>
#include "imgproc.h"
#include "fftimgproc.h"
#include <iostream>
#include "multichannel.h"
namespace img
{

class FFTImgProc
{
  public:

    //! Construct with no content
    FFTImgProc();
   ~FFTImgProc();

    //! delete existing content and leave in a blank state
    void clear();
    //! delete existing content and re-initialize to the input dimensions with value 0.0
    void clear(int nX, int nY, int nC);

    //! Retrieve the width
    int nx() const { return Nx; }
    //! Retrieve the height
    int ny() const { return Ny; }
    //! Retrieve the number of channels
    int depth() const { return Nc; }

    //! Retrieve the (multichannel) complex value at a pixel
    void value( int i, int j, std::vector< std::complex<double> >& pixel) const;
    //! Set the (multichannel) complex value at a pixel.
    void set_value( int i, int j, const std::vector<std::complex<double> >& pixel);

    double kx(int i) const;
    double ky(int j) const;

    //! Copy constructor. Clears existing content.
    FFTImgProc(const FFTImgProc& v);
    //! Copy assignment. Clears existing content.
    FFTImgProc& operator=(const FFTImgProc& v);

    //! indexing to a particular pixel and channel
    long index(int i, int j, int c) const;
    //! indexing to a particular pixel
    long index(int i, int j) const;


    fftw_complex* raw(){ return img_data; }

    void fft_forward();
    void fft_backward();


  private:

    int Nx, Ny, Nc;
    long Nsize;
    fftw_complex * img_data;
    std::vector<fftw_plan> forward;
    std::vector<fftw_plan> backward;
};

void load_fft_new( const ImgProc& input, FFTImgProc& fftoutput );
}

#endif
