
#include "fftimgproc.h"

using namespace img;

FFTImgProc::FFTImgProc() :
   Nx(0),
   Ny(0),
   Nc(0),
   img_data(nullptr)
{}

FFTImgProc::~FFTImgProc(){ clear(); }

void img::load_fft_new( const ImgProc& input, FFTImgProc& fftoutput )
{
   fftoutput.clear( input.nx(), input.ny(), input.depth() );
   for(int j=0;j<input.ny();j++)
   {
      for(int i=0;i<input.nx();i++)
      {
         std::vector<float> ci;
	 std::vector< std::complex<double> > citilde;
	 input.value(i,j,ci);
	 for(size_t c=0;c<ci.size();c++)
	 {
	    std::complex<double> v(ci[c], 0.0);
	    citilde.push_back(v);
	 }
	 fftoutput.set_value(i,j,citilde);
      }
   }
}

void FFTImgProc::clear()
{
   if( img_data != nullptr )
   {
      for(size_t i=0;i<forward.size();i++){ fftw_destroy_plan(forward[i]); }
      for(size_t i=0;i<backward.size();i++){ fftw_destroy_plan(backward[i]); }
      fftw_free(img_data);
      img_data = nullptr;
   }
   Nx = 0;
   Ny = 0;
   Nc = 0;
   Nsize = 0;
}

void FFTImgProc::clear(int nX, int nY, int nC)
{
   clear();
   Nx = nX;
   Ny = nY;
   Nc = nC;
   Nsize = Nx*Ny*Nc;
   img_data = (fftw_complex*)fftw_malloc( sizeof(fftw_complex)*Nx*Ny*Nc );
#pragma omp parallel for
   for(long i=0;i<Nsize;i++){ img_data[i][0] = 0.0;  img_data[i][1] = 0.0; }

   forward.resize(Nc);
   backward.resize(Nc);
   for(int i=0;i<Nc;i++)
   {
      forward[i] = fftw_plan_dft_2d( Ny, Nx, &img_data[index(0,0,i)], &img_data[index(0,0,i)], FFTW_FORWARD, FFTW_ESTIMATE);
      backward[i] = fftw_plan_dft_2d( Ny, Nx, &img_data[index(0,0,i)], &img_data[index(0,0,i)], FFTW_BACKWARD, FFTW_ESTIMATE);
   }
}

void FFTImgProc::value( int i, int j, std::vector<std::complex<double> >& pixel) const
{
   if( img_data == nullptr ){ return; }
   if( i<0 || i>=Nx ){ return; }
   if( j<0 || j>=Ny ){ return; }
   pixel.resize(Nc);
   for( int c=0;c<Nc;c++ )
   {
      const fftw_complex& v = img_data[index(i,j,c)];
      std::complex<double> a(v[0],v[1]);
      pixel[c] = a;
   }
   return;
}



FFTImgProc::FFTImgProc(const FFTImgProc& v) :
  Nx (v.Nx),
  Ny (v.Ny),
  Nc (v.Nc),
  Nsize (v.Nsize)
{
   img_data = (fftw_complex*)fftw_malloc( sizeof(fftw_complex)*Nx*Ny*Nc );
#pragma omp parallel for
   for(long i=0;i<Nsize;i++){ img_data[i][0] = v.img_data[i][0];  img_data[i][1] = v.img_data[i][1]; }

   forward.resize(Nc);
   backward.resize(Nc);
   for(int i=0;i<Nc;i++)
   {
      forward[i] = fftw_plan_dft_2d( Ny, Nx, &img_data[index(0,0,i)], &img_data[index(0,0,i)], FFTW_FORWARD, FFTW_ESTIMATE);
      backward[i] = fftw_plan_dft_2d( Ny, Nx, &img_data[index(0,0,i)], &img_data[index(0,0,i)], FFTW_BACKWARD, FFTW_ESTIMATE);
   }
}

FFTImgProc& FFTImgProc::operator=(const FFTImgProc& v)
{
   if( this == &v ){ return *this; }
   if( Nx != v.Nx || Ny != v.Ny || Nc != v.Nc )
   {
      clear();
      Nx = v.Nx;
      Ny = v.Ny;
      Nc = v.Nc;
      Nsize = v.Nsize;
   }

   img_data = (fftw_complex*)fftw_malloc( sizeof(fftw_complex)*Nx*Ny*Nc );
#pragma omp parallel for
   for(long i=0;i<Nsize;i++){ img_data[i][0] = v.img_data[i][0];  img_data[i][1] = v.img_data[i][1]; }

   forward.resize(Nc);
   backward.resize(Nc);
   for(int i=0;i<Nc;i++)
   {
      forward[i] = fftw_plan_dft_2d( Ny, Nx, &img_data[index(0,0,i)], &img_data[index(0,0,i)], FFTW_FORWARD, FFTW_ESTIMATE);
      backward[i] = fftw_plan_dft_2d( Ny, Nx, &img_data[index(0,0,i)], &img_data[index(0,0,i)], FFTW_BACKWARD, FFTW_ESTIMATE);
   }
   return *this;
}











void FFTImgProc::set_value( int i, int j, const std::vector<std::complex<double> >& pixel)
{
   if( img_data == nullptr ){ return; }
   if( i<0 || i>=Nx ){ return; }
   if( j<0 || j>=Ny ){ return; }
   if( pixel.size() != (size_t)Nc ){ return; }
   for( int c=0;c<Nc;c++ )
   {
      fftw_complex& v = img_data[index(i,j,c)];
      v[0] = pixel[c].real();
      v[1] = pixel[c].imag();
   }
   return;
}



void FFTImgProc::fft_forward()
{
   for(size_t i=0;i<forward.size();i++){ fftw_execute(forward[i]); }
}

void FFTImgProc::fft_backward()
{
   for(size_t i=0;i<backward.size();i++){ fftw_execute(backward[i]); }
   double norm = 1.0/(Nx*Ny);
   for(int j=0;j<Ny;j++)
   {
#pragma omp parallel for
      for(int i=0;i<Nx;i++)
      {
         for(int c=0;c<Nc;c++)
	 {
	    img_data[index(i,j,c)][0] *= norm;
	    img_data[index(i,j,c)][1] *= norm;
	 }
      }
   }
}

double FFTImgProc::kx(int i) const
{
   double v = (double)i/Nx;
   if(i>Nx/2){ v -= 1.0; }
   return 2.0*3.14159265*v;
}

double FFTImgProc::ky(int j) const
{
   double v = (double)j/Ny;
   if(j>Ny/2){ v -= 1.0; }
   return 2.0*3.14159265*v;
}


long FFTImgProc::index(int i, int j, int c) const
{
   // return (long) c + (long) Nc * index(i,j); // interleaved channels

   return index(i,j) + (long)Nx * (long)Ny * (long)c; // sequential channels
}

long FFTImgProc::index(int i, int j) const
{
   return (long) i + (long)Nx * (long)j;
}
