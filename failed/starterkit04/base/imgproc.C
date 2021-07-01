
#include <cmath>
#include "imgproc.h"
#include <string>
#include <iostream>
#include <algorithm>

//#include <IFSFunction.h>

#include <OpenImageIO/imageio.h>
OIIO_NAMESPACE_USING

using namespace img;
using namespace std;

ColorLUT::ColorLUT(double gam) :
  gamma(gam)
{
  std::vector<float> C;
  C.push_back(0.0);
  C.push_back(0.0);
  C.push_back(0.0);
  black = C;

  C[0] = 246.0/255.0;
  C[1] = 103.0/255.0;
  C[2] = 51.0/255.0;
  bands.push_back(C);

  C[0] = 212.0/255.0;
  C[1] = 201.0/255.0;
  C[2] = 158.0/255.0;
  bands.push_back(C);

  C[0] = 86.0/255.0;
  C[1] = 97.0/255.0;
  C[2] = 39.0/255.0;
  bands.push_back(C);

  C[0] = 58.0/255.0;
  C[1] = 73.0/255.0;
  C[2] = 88.0/255.0;
  bands.push_back(C);

  C[0] = 181.0/255.0;
  C[1] = 195.0/255.0;
  C[2] = 39.0/255.0;
  bands.push_back(C);

  C[0] = 16.0/255.0;
  C[1] = 157.0/255.0;
  C[2] = 192.0/255.0;
  bands.push_back(C);

  C[0] = 16.0/255.0;
  C[1] = 16.0/255.0;
  C[2] = 16.0/255.0;
  bands.push_back(C);
}

void ColorLUT::operator()(const double& value, std::vector<float>& C) const
{
    C = black;
    if(value > 1.0|| value <0.0)
    {
      //std::cout<<"error!!!!"<<std::endl;
      return;
    }
    double x = std::pow(value, gamma)*(bands.size()-1);
    size_t low_index = (size_t)x;
    size_t high_index = low_index + 1;

    double weight = x -(double)low_index;
    if(high_index >= bands.size())
    {
      high_index = bands.size()-1;
    }
    for(size_t c = 0;c<C.size();c++)
    {
      C[c] = bands[low_index][c] * (1.0-weight) + bands[high_index][c]*weight;
    }
}

ImgProc::ImgProc() :
  Nx (0),
  Ny (0),
  Nc (0),
  Nsize (0),
  single_Nsize (0),
  img_data (nullptr)
{}

ImgProc::~ImgProc()
{
   clear();
}

void ImgProc::clear()
{
   if( img_data != nullptr ){ delete[] img_data; img_data = nullptr;}
   Nx = 0;
   Ny = 0;
   Nc = 0;
   Nsize = 0;
   single_Nsize = 0;
}

void ImgProc::clear(int nX, int nY, int nC)
{
   clear();
   Nx = nX;
   Ny = nY;
   Nc = nC;
   Nsize = (long)Nx * (long)Ny * (long)Nc;
   single_Nsize = (long)Nx * (long)Ny;
   img_data = new float[Nsize];
#pragma omp parallel for
   for(long i=0;i<Nsize;i++){ img_data[i] = 0.0; }
}

bool ImgProc::load( const std::string& filename )
{
   auto in = ImageInput::create (filename);
   if (!in) {return false;}
   ImageSpec spec;
   in->open (filename, spec);
   clear();
   Nx = spec.width;
   Ny = spec.height;
   Nc = spec.nchannels;
   Nsize = (long)Nx * (long)Ny * (long)Nc;
   single_Nsize = (long)Nx * (long)Ny;
   img_data = new float[Nsize];
   in->read_image(TypeDesc::FLOAT, img_data);
   in->close ();
   return true;
}


__device__ void ImgProc::value( int i, int j, std::vector<float>& pixel) const
{
   pixel.clear();
   if( img_data == nullptr ){ return; }
   if( i<0 || i>=Nx ){ return; }
   if( j<0 || j>=Ny ){ return; }
   pixel.resize(Nc);
   for( int c=0;c<Nc;c++ )
   {
      pixel[c] = img_data[index(i,j,c)];
   }
   return;
}

__device__ void ImgProc::set_value( int i, int j, const std::vector<float>& pixel)
{
   if( img_data == nullptr ){ return; }
   if( i<0 || i>=Nx ){ return; }
   if( j<0 || j>=Ny ){ return; }
   if( Nc > (int)pixel.size() ){ return; }
#pragma omp parallel for
   for( int c=0;c<Nc;c++ )
   {
      img_data[index(i,j,c)] = pixel[c];
   }
   return;
}


ImgProc::ImgProc(const ImgProc& v) :
  Nx (v.Nx),
  Ny (v.Ny),
  Nc (v.Nc),
  Nsize (v.Nsize),
  single_Nsize (v.single_Nsize)
{
   img_data = new float[Nsize];
#pragma omp parallel for
   for( long i=0;i<Nsize;i++){ img_data[i] = v.img_data[i]; }
}

ImgProc& ImgProc::operator=(const ImgProc& v)
{
   if( this == &v ){ return *this; }
   if( Nx != v.Nx || Ny != v.Ny || Nc != v.Nc )
   {
      clear();
      Nx = v.Nx;
      Ny = v.Ny;
      Nc = v.Nc;
      Nsize = v.Nsize;
      single_Nsize = v.single_Nsize;
   }
   img_data = new float[Nsize];
#pragma omp parallel for
   for( long i=0;i<Nsize;i++){ img_data[i] = v.img_data[i]; }
   return *this;
}


void ImgProc::operator*=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] *= v; }
}

void ImgProc::operator/=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] /= v; }
}

void ImgProc::operator+=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] += v; }
}

void ImgProc::operator-=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] -= v; }
}


//******************************************************************************
void ImgProc::compliment()
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] = 1.0 - img_data[i]; }


//====undo=====
   undo_vec.push_back(10);
}

void ImgProc::brightness_up()
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ )
   {
       img_data[i] = img_data[i] * 1.1;
   }

//====undo=====
   undo_vec.push_back(1);
}

void ImgProc::brightness_down()
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ )
   {
       img_data[i] = img_data[i] / 1.1;
   }

//====undo=====
   undo_vec.push_back(2);
}

void ImgProc::bias_up()
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ )
   {
       img_data[i] = img_data[i] + 0.05;
   }

//====undo=====
   undo_vec.push_back(3);
}

void ImgProc::bias_down()
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ )
   {
       img_data[i] = img_data[i] - 0.05;
   }

//====undo=====
   undo_vec.push_back(4);
}

void ImgProc::gamma_up()
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ )
   {
       img_data[i] = pow(img_data[i], 0.9);
   }

//====undo=====
   undo_vec.push_back(5);
}

void ImgProc::gamma_down()
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ )
   {
       img_data[i] = pow(img_data[i], 1.1);
   }

//====undo=====
   undo_vec.push_back(6);
}

void ImgProc::grayscale()
{
   if( img_data == nullptr ){ return; }
   vector<float> undo_temp;
   undo_temp.resize(Nsize);

#pragma omp parallel for
   for( long i=0;i<Nsize;i = i+Nc )
   {
       for(int j=0; j<Nc; j++)
       {
         undo_temp[i+j] = img_data[i+j];
       }
      float g = img_data[i] * 0.2126 + img_data[i+1] * 0.7152 + img_data[i+2] * 0.0722;
       img_data[i] =  g;
       img_data[i+1] =  g;
       img_data[i+2] =  g;
   }

//====undo=====
   undo_vec.push_back(7);
   temp_data.push_back(undo_temp);
}
//pixel[c] = img_data[index(i,j,c)];

void ImgProc::quantize()
{
   if( img_data == nullptr ){ return; }
   vector<float> undo_temp;
   undo_temp.resize(Nsize);

#pragma omp parallel for
   for( long i=0;i<Nsize;i++ )
   {
      undo_temp[i] = img_data[i];
      img_data[i] = floor(img_data[i] * 10) / 10.0;
   }

//====undo=====
   undo_vec.push_back(8);
   temp_data.push_back(undo_temp);
}

void ImgProc::rms_contrast()
{
   if( img_data == nullptr ){ return; }
   vector<float> undo_temp;
   undo_temp.resize(Nsize);
   //try to improve my code to be compatible
   //float mean_r, mean_g, mean_b;
   float mean = 0.0;
   //float dev_r, dev_g, dev_b;
   float dev = 0.0;
   // mean_r = 0.0;
   // mean_g = 0.0;
   // mean_b = 0.0;
   // dev_r = 0.0;
   // dev_g = 0.0;
   // dev_b = 0.0;

   for(int j=0; j<Nc; j++)
   {
     mean = 0.0;
     dev = 0.0;
     //for undo,stroe the data into memory
     for(long i=j; i<Nsize; i=i+Nc)
     {
       undo_temp[i] = img_data[i];
     }


     for( long i=j;i<Nsize;i = i+Nc )
     {
        mean += img_data[i];
     }
     mean /= (float)single_Nsize;

     for( long i=j;i<Nsize;i = i+Nc )
     {
        dev += (img_data[i] - mean) * (img_data[i] - mean);
     }
     dev /= (float)single_Nsize;
     dev = sqrt(dev);

     for( long i=j;i<Nsize;i = i+Nc )
     {
         img_data[i] = (img_data[i] - mean) / dev;
     }

   }


//====undo=====
   undo_vec.push_back(9);
   temp_data.push_back(undo_temp);
}

//====================histogram start======================
//histogram_bins_num = 10;
float delta_i;
void ImgProc::compute_msg(int n)
{
  float mean = 0.0;
  float dev = 0.0;
  float maxi = -100000.0;
  float mini = 100000.0;


  int j=n;
  mean = 0.0;
  dev = 0.0;

  for( long i=j;i<Nsize;i = i+Nc )
  {
    if(img_data[i] > maxi)
    {
      maxi = img_data[i];
      //std::cout <<"maxi"<<maxi<<std::endl;

    }
    if(img_data[i] < mini)
    {
      mini = img_data[i];
    }
     mean += img_data[i];
  }
  mean /= (float)single_Nsize;

  for( long i=j;i<Nsize;i = i+Nc )
  {
     dev += (img_data[i] - mean) * (img_data[i] - mean);
  }
  dev /= (float)single_Nsize;
  dev = sqrt(dev);

  min_value = mini;
  max_value = maxi;
  //\n
  std::cout << "min value for channel  "<<j<<":     "<<mini<<std::endl;
  std::cout << "max value for channel  "<<j<<":     "<<maxi<<std::endl;
  std::cout << "deviation for channel  "<<j<<":     "<<dev<<std::endl;
  std::cout << "average for channel    "<<j<<":     "<<mean<<std::endl;

  //std::cout <<"min"<<min_value<<std::endl;
  //std::cout <<"max"<<max_value<<std::endl;
  //std::cout <<"dev"<<dev<<std::endl;
  delta_i = (max_value - min_value)/histogram_bins_num;
}

void ImgProc::compute_PDF()
{
  for(int i=0; i<10; i++)
  {
    pdf[i] = hist[i] / (float)single_Nsize;
  }
}

void ImgProc::compute_CDF()
{
  cdf[0] = pdf[0];
  for(int i=1; i<10; i++)
  {
    cdf[i] = cdf[i-1] + pdf[i];
  }
}

//histogram equalization
void ImgProc::equal_histogram()
{
  if( img_data == nullptr ){ return; }
  vector<float> undo_temp;
  undo_temp.resize(Nsize);

  for(int j=0; j<Nc; j++)
  {
    //for undo,stroe the data into memory
    for(long i=j; i<Nsize; i=i+Nc)
    {
      undo_temp[i] = img_data[i];
    }

    ImgProc::compute_msg(j);
    for(int l=0;l <10; l++)
    {
      hist[l]=0;
    }
    for( long i=j;i<Nsize;i = i+Nc )
    {
      bin_index = (img_data[i] - min_value)/delta_i;
      bin_index = floor(bin_index);
      if(bin_index<10)
        hist[bin_index]++;
    }

    ImgProc::compute_PDF();
    ImgProc::compute_CDF();

    //equalization
    float q = 0.0;
    int qq = 0;
    float w = 0.0;
    for( long i=j;i<Nsize;i = i+Nc )
    {
      q = (img_data[i] - min_value)/delta_i;
      qq = floor(q);
      w = q - qq;
      if(qq<histogram_bins_num-1)
        img_data[i] = cdf[qq]*(1-w) + cdf[qq+1]*w;
      else
        img_data[i] = cdf[qq];
    }
  }

  //====undo=====
     undo_vec.push_back(11);
     temp_data.push_back(undo_temp);
}

//====================histogram end======================


//====================log quench start======================
void ImgProc::log_quench()
{
  if( img_data == nullptr ){ return; }
  vector<float> undo_temp;
  undo_temp.resize(Nsize);

  //double L[single_Nsize];//correct
  vector<double> Li;

  for( long i=0;i<Nsize;i = i+Nc )
  {
    //for undo,stroe the data into memory
    for(int j=0; j<Nc; j++)
    {
      undo_temp[i+j] = img_data[i+j];
    }
    Li.push_back(img_data[i] * 0.2126 + img_data[i+1] * 0.7152 + img_data[i+2] * 0.0722);

  }
  // for(int i=0;i<Li.size();i++)
  // {
  //   cout<<Li.size()<<endl;
  // }
  // if(Li.size()==single_Nsize)
  //   std::cout<<"aaaaaaaaaaaaaaaaaaa"<<std::endl;
  double min_l = 100000;
  double max_l = -100000;
  for(int i=0;i<Li.size();i++)
  {
    if(Li[i]<min_l)
    {
      min_l = Li[i];
    }
    if(Li[i]>max_l)
    {
      max_l = Li[i];
    }
  }
  double tau = 0.2*(max_l-min_l);
  for(int i=0;i<Li.size();i++)
  {
    Li[i]=(log(Li[i]+tau)-log(min_l+tau))/(log(max_l+tau)-log(min_l+tau));
  }
  for( long i=0;i<Nsize;i = i+Nc )
  {
    for(int j=0; j<Nc; j++)
    {
      img_data[i+j]*=Li[i/Nc];
    }
  }

//====undo=====
  undo_vec.push_back(12);
  temp_data.push_back(undo_temp);
}
//====================log quench end======================

//====================IFS start===========================
void img::FractalFlameIFS(const size_t nb_iterations,const std::vector<IFSFunction*>& func_list, const ColorLUT& lut, ImgProc& out)
{
  float Pi = 3.1415926;
  //Point P;
  std::vector<float> P;
  P.push_back(0.0);
  P.push_back(0.0);
  P[0] = 2.0*drand48() - 1.0;
  P[1] = 2.0*drand48() - 1.0;
  float w = drand48();
  //std::cout<<"Px"<<P.x<<",Py"<<P.y<<std::endl;
  for(size_t iter = 0;iter<nb_iterations;iter++)
  {
    size_t ic = (size_t)(drand48()*5);//func_list.size());
    //std::cout<<"Px11:       "<<P[0]<<",Py1:      "<<P[1]<<endl;
    //P = (*func_list[0])(P);
    //std::cout<<"ic:       "<<ic<<std::endl;
    if(ic==0)
    {
      std::vector<float> pp;
      pp.push_back(0.0);
      pp.push_back(0.0);
      double r = sqrt(P[0]*P[0] + P[1]*P[1]);
      pp[0] = pow(P[1], 5) / max(0.01,pow(r, 5));
      pp[1] = -1.0*pow(P[0], 5) / max(0.01,pow(r, 5));
      std::cout<<"Px2:       "<<pp[0]<<",Py2:      "<<pp[1]<<std::endl;
      P[0] = pp[0];
      P[1] = pp[1];
      std::cout<<"Px22:       "<<P[0]<<",Py2:      "<<P[1]<<endl;
    }
    if(ic==1)
    {
      std::vector<float> pp;
      pp.push_back(0.0);
      pp.push_back(0.0);
      float r = sqrt(P[0]*P[0] + P[1]*P[1]);
      float thi;
      if(P[1]<0.01)
        thi = atan2( P[0] , 0.01);
      else
        thi = atan2( P[0] ,  P[1] );
      pp[0]= thi/Pi;
      pp[1]=r-1.0;
      P[0] = pp[0];
      P[1] = pp[1];
    }
    if(ic==2)
    {
      std::vector<float> pp;
      pp.push_back(0.0);
      pp.push_back(0.0);
      float r = sqrt(P[0]*P[0] + P[1]*P[1]);
      float thi;
      if(P[1]<0.01)
        thi = atan2( P[0] , 0.01);
      else
        thi = atan2( P[0] ,  P[1] );
      pp[0]= r * sin(thi + r);
      pp[1]= r * cos(thi - r);
      // std::vector<float> pp;
      // pp.push_back(0.0);
      // pp.push_back(0.0);
      // float r = sqrt(P[0]*P[0] + P[1]*P[1]);
      // float c = 0.7;
      // float temp = (int)(r+c*c)%(int)(2*c*c) - c*c + r*(1.0-c*c);
      // float thi = atan2( P[0] ,  P[1] );
      // pp[0]= temp*cos(thi);
      // pp[1]= temp*sin(thi);
      P[0] = pp[0];
      P[1] = pp[1];
    }
    if(ic==3)
    {
      std::vector<float> pp;
      pp.push_back(0.0);
      pp.push_back(0.0);
      float r = sqrt(P[0]*P[0] + P[1]*P[1]);
      float thi;
      if(P[1]<0.01)
        thi = atan2( P[0] , 0.01);
      else
        thi = atan2( P[0] ,  P[1] );
      pp[0]= thi/Pi;
      pp[1]=r-1.0;
      P[0] = pp[0];
      P[1] = pp[1];
    }
    if(ic==4)
    {
      std::vector<float> pp;
      pp.push_back(0.0);
      pp.push_back(0.0);
      float r;
      if(r<0.01)
        r = 0.01;
      else
        r = sqrt(P[0]*P[0] + P[1]*P[1]);
      float thi;
      if(P[1]<0.01)
        thi = atan2( P[0] , 0.01);
      else
        thi = atan2( P[0] ,  P[1] );
      pp[0]= (cos(thi)+sin(r))/r;
      pp[1]= (sin(thi)-cos(r))/r;
      P[0] = pp[0];
      P[1] = pp[1];
    }


    //define a weight function and use here
    //w = (w+func_list[ic]->color()[0])*0.5;//use the function color red value for the weight
    w = (w+0.6)*0.5;
    if(iter > 20)
    {
      if(P[0] >= -1.0 && P[0] <= 1.0 &&P[1] >=-1.0 && P[1] <= 1.0)
      {
        float x = P[0] + 1.0;
        float y = P[1] + 1.0;
        // std::cout<<"P.x     "<<x<<std::endl;
        // std::cout<<"P.y     "<<y<<std::endl;
        // std::cout<<"         "<<std::endl;
        x *= 0.5*out.nx();
        y *= 0.5*out.ny();
        int ii = x;
        if(ii < out.nx())
        {
          int jj = y;
          if(jj < out.ny())
          {
            std::vector<float> color;
            lut(w, color);
            // std::cout<<"color0"<<color[0]<<std::endl;
            // std::cout<<"color1"<<color[1]<<std::endl;
            // std::cout<<"color2"<<color[2]<<std::endl;

            std::vector<float> cc;
            out.value(ii, jj, cc);
            for(size_t iic=0;iic<cc.size()-1;iic++)
            {
              cc[iic] = cc[iic]*cc[cc.size()-1];
              cc[iic] = (cc[iic]+color[iic])/(cc[cc.size()-1]+1.0);
            }
            cc[cc.size()-1] += 1;
            // std::cout<<"color0  "<<cc[0]<<std::endl;
            // std::cout<<"color1  "<<cc[1]<<std::endl;
            // std::cout<<"color2  "<<cc[2]<<std::endl;
            // std::cout<<"ij  "<<ii<<"    "<<jj<<std::endl;
            // std::cout<<"                  "<<std::endl;
            out.set_value(ii, jj, cc);
            //std::cout<<ii<<"j:"<<cc[cc.size()-1]<<std::endl;
          }
        }
      }
    }
  }
}
// bool ifs_started = false;
// void ImgProc::doIFS()
// {
//   if(!ifs_started)
//   {
//     image.clear(image.nx(), image.ny(), 4);
//     ifs_started = true;
//   }
//
//   IFSFunction f0;
//   //Sinusoidal f0;
//   //Popcorm f1(0.1, 0.1);
//
//   std::vector<img::IFSFunction*> s;
//   s.push_back(&f0);
//   //s.push_back(&f1);
//
//   size_t nb_iter = 100000;
//   ColorLUT lut;
//   FractalFlameIFS(nb_iter, s, lut, image);
//   glutPostRedisplay();
// }
//====================IFS end===========================



//==============part for the undo system===========
void ImgProc::undo_step()
{
   if (undo_vec.size() == 0)
   {
     cout << "No more steps to undo\n";
     return;
   }
   int udo_type = 0;
   //udo_type = undo_vec.back();
   udo_type = undo_vec[undo_vec.size()-1];
   //std::cout << "type:"    <<  udo_type <<"\n";
   if (udo_type == 0)
   {
     cout << "Something is not working I guess\n";
     return;
   }

   else if (udo_type == 1)
   {
     if( img_data == nullptr ){ return; }
   #pragma omp parallel for
     for( long i=0;i<Nsize;i++ )
     {
         img_data[i] = img_data[i] / 1.1;
     }
     undo_vec.pop_back();
     std::cout << "undo\n";
     //std::cout << "type"    <<  udo_type <<"\n";
   }

   else if (udo_type == 2)
   {
     if( img_data == nullptr ){ return; }
  #pragma omp parallel for
     for( long i=0;i<Nsize;i++ )
     {
         img_data[i] = img_data[i] * 1.1;
     }
     undo_vec.pop_back();
     std::cout << "undo\n";
   }

   else if (udo_type == 3)
   {
     if( img_data == nullptr ){ return; }
   #pragma omp parallel for
     for( long i=0;i<Nsize;i++ )
     {
         img_data[i] = img_data[i] - 0.05;
     }
     undo_vec.pop_back();
     std::cout << "undo\n";
   }

   else if (udo_type == 4)
   {
     if( img_data == nullptr ){ return; }
   #pragma omp parallel for
     for( long i=0;i<Nsize;i++ )
     {
         img_data[i] = img_data[i] + 0.05;
     }
     undo_vec.pop_back();
     std::cout << "undo\n";
   }

   else if (udo_type == 5)
   {
     if( img_data == nullptr ){ return; }
   #pragma omp parallel for
     for( long i=0;i<Nsize;i++ )
     {
         img_data[i] = pow(img_data[i], (1.0/0.9));
     }
     undo_vec.pop_back();
     std::cout << "undo\n";
   }

   else if (udo_type == 6)
   {
     if( img_data == nullptr ){ return; }
   #pragma omp parallel for
     for( long i=0;i<Nsize;i++ )
     {
         img_data[i] = pow(img_data[i], (1.0/1.1));
     }
     undo_vec.pop_back();
     std::cout << "undo\n";
   }

   else if (udo_type == 7)
   {
     if( img_data == nullptr ){ return; }
     int t = temp_data.size()-1;
     if(t<0){ return; }
   #pragma omp parallel for
     for( long i=0;i<Nsize;i++ )
     {
         img_data[i] = temp_data[t][i];
     }
     temp_data.pop_back();
     undo_vec.pop_back();
     std::cout << "undo\n";
   }

   else if (udo_type == 8)
   {
     if( img_data == nullptr ){ return; }
     int t = temp_data.size()-1;
     if(t<0){ return; }
   #pragma omp parallel for
     for( long i=0;i<Nsize;i++ )
     {
         img_data[i] = temp_data[t][i];
     }
     temp_data.pop_back();
     undo_vec.pop_back();
     std::cout << "undo\n";
   }

   else if (udo_type == 9)
   {
     if( img_data == nullptr ){ return; }
     int t = temp_data.size()-1;
     if(t<0){ return; }
   #pragma omp parallel for
     for( long i=0;i<Nsize;i++ )
     {
         img_data[i] = temp_data[t][i];
     }
     temp_data.pop_back();
     undo_vec.pop_back();
     std::cout << "undo\n";
   }

   else if (udo_type == 10)
   {
     if( img_data == nullptr ){ return; }
   #pragma omp parallel for
     for( long i=0;i<Nsize;i++ )
     {
         img_data[i] = 1.0 - img_data[i];
     }
     undo_vec.pop_back();
     std::cout << "undo\n";
   }

   else if (udo_type == 11)
   {
     if( img_data == nullptr ){ return; }
     int t = temp_data.size()-1;
     if(t<0){ return; }
   #pragma omp parallel for
     for( long i=0;i<Nsize;i++ )
     {
         img_data[i] = temp_data[t][i];
     }
     temp_data.pop_back();
     undo_vec.pop_back();
     std::cout << "undo\n";

   }
}


//===================overview of the struct(included in C system)==================
// struct tm
// {
//   int tm_sec; //0~59, 60 if leap second23:59:60
//   int tm_min; //0-59
//   int tm_hour; //0-23
//   int tm_mday; //1-31
//   int tm_mon; //0-11
//   int tm_year; //Year - 1900
//   int tm_wday; // 0-6, Sunday = 0
//   int tm_yday; //0-365, 1 Jan = 0
//   int tm_isdst; //>0 summer time on; <0 unusable;
//   char  *tm_zone; //zone name
// }

//========================output system==========================
void ImgProc::output_exr()
{
  time_t tt = time(NULL);
  struct tm * t = localtime(&tt);
  int ty = t->tm_year + 1900;
  int tm = t->tm_mon + 1;
  string tmp = to_string(ty) + to_string(tm)+ to_string(t->tm_mday) + to_string(t->tm_hour)
   + to_string(t->tm_min)+ "_" + to_string(t->tm_sec);
   if(undo_vec.size() == 0)
   {tmp+="original.exr";}
   else
   {
     int udo_type = undo_vec[undo_vec.size()-1];
     if (udo_type == 1||udo_type == 2)
     {
       tmp += "brightness.exr";
     }

     else if (udo_type == 3||udo_type == 4)
     {
       tmp+="bias.exr";
     }

     else if (udo_type == 5||udo_type == 6)
     {
       tmp+="gamma.exr";
     }

     else if (udo_type == 7)
     {
       tmp+="grayscale.exr";
     }

     else if (udo_type == 8)
     {
       tmp+="quantize.exr";
     }

     else if (udo_type == 9)
     {
       tmp+="rms_contrast.exr";
     }

     else if (udo_type == 10)
     {
       tmp+="compliment.exr";
     }
  }

  //name the file
  const char *filename = tmp.c_str();
  //ttmp = ty + tm + "foo.exr";
  //const char *filename = tostring(ty) + tostring(tm) + "foo.exr";
  std::cout << "file_name:"  <<  tmp <<"\n";
  const int xres = Nx, yres = Ny;
  const int channels = Nc;
  //unsigned char pixels[xres*yres*channels];

  //std::unique_ptr<ImageOutput> out = ImageOutput::create (filename);
  auto out = ImageOutput::create (filename);
  if(! out)
      return;
  //ImageSpec spec (xres, yres, channels, TypeDesc::UINT8);
  ImageSpec spec (xres, yres, channels, TypeDesc::FLOAT);
  out->open (filename, spec);
  //out->write_image (TypeDesc::UINT8, pixels);
  out->write_image (TypeDesc::FLOAT, img_data);
  out->close ();
}

void ImgProc::output_jpeg()
{
  time_t tt = time(NULL);
  struct tm * t = localtime(&tt);
  int ty = t->tm_year + 1900;
  int tm = t->tm_mon + 1;
  string tmp = to_string(ty) + to_string(tm)+ to_string(t->tm_mday) + to_string(t->tm_hour)
   + to_string(t->tm_min)+ "_" + to_string(t->tm_sec);
  if(undo_vec.size() == 0)
  {tmp+="original.jpg";}
  else
  {
    int udo_type = undo_vec[undo_vec.size()-1];
    if (udo_type == 1||udo_type == 2)
    {
      tmp += "brightness.jpg";
    }

    else if (udo_type == 3||udo_type == 4)
    {
      tmp+="bias.jpg";
    }

    else if (udo_type == 5||udo_type == 6)
    {
      tmp+="gamma.jpg";
    }

    else if (udo_type == 7)
    {
      tmp+="grayscale.jpg";
    }

    else if (udo_type == 8)
    {
      tmp+="quantize.jpg";
    }

    else if (udo_type == 9)
    {
      tmp+="rms_contrast.jpg";
    }

    else if (udo_type == 10)
    {
      tmp+="compliment.jpg";
    }
  }


  //name the file
  const char *filename = tmp.c_str();
  std::cout << "file_name:"  <<  tmp <<"\n";
  const int xres = Nx, yres = Ny;
  const int channels = Nc;
  auto out = ImageOutput::create (filename);
  if(! out)
      return;
  ImageSpec spec (xres, yres, channels, TypeDesc::FLOAT);
  out->open (filename, spec);
  out->write_image (TypeDesc::FLOAT, img_data);
  out->close ();

}

//========================output system==========================

long ImgProc::index(int i, int j, int c) const
{
   return (long) c + (long) Nc * index(i,j); // interleaved channels

   // return index(i,j) + (long)Nx * (long)Ny * (long)c; // sequential channels
}

// long ImgProc::index2(int i, int j, int c) const
// {
//    return index(i,j) + (long)Nx * (long)Ny * (long)c; // sequential channels
// }

long ImgProc::index(int i, int j) const
{
   return (long) i + (long)Nx * (long)j;
}



void img::swap(ImgProc& u, ImgProc& v)
{
   float* temp = v.img_data;
   int Nx = v.Nx;
   int Ny = v.Ny;
   int Nc = v.Nc;
   long Nsize = v.Nsize;

   v.Nx = u.Nx;
   v.Ny = u.Ny;
   v.Nc = u.Nc;
   v.Nsize = u.Nsize;
   v.img_data = u.img_data;

   u.Nx = Nx;
   u.Ny = Ny;
   u.Nc = Nc;
   u.Nsize = Nsize;
   u.img_data = temp;
}
