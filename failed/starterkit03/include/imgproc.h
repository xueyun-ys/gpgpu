

#ifndef IMGPROC_H
#define IMGPROC_H

#include <string>
#include <vector>

#include <IFSFunction.h>
//#include <linux/cdev.h>

namespace img
{
//typedef struct...
// struct Point
// {
//   float x;
//   float y;
// };

    // class Point
    // {
    // public:
    //
    //   Point(){}
    //   ~Point(){}
    //
    //   float x;
    //   flaot y;
    // }

    class ColorLUT
    {
      public:

        ColorLUT(double gamma = 1.0);
        ~ColorLUT(){}

        //Generate color value from the input value
        void operator()(const double &value, std::vector<float>& C) const;

      private:

        double gamma;
        std::vector<float> black;
        std::vector<std::vector<float> >bands;
    };

    class ImgProc
    {

      public:

        //! Construct with no content
        ImgProc();
       ~ImgProc();

        //! delete existing content and leave in a blank state
        void clear();
        //! delete existing content and re-initialize to the input dimensions with value 0.0
        void clear(int nX, int nY, int nC);

        //! Load an image from a file.  Deletes exising content.
        bool load( const std::string& filename );

        //! Retrieve the width
        int nx() const { return Nx; }
        //! Retrieve the height
        int ny() const { return Ny; }
        //! Retrieve the number of channels
        int depth() const { return Nc; }

        //! Retrieve the (multichannel) value at a pixel.  Copies the value into parameter 'pixel'.
        void value( int i, int j, std::vector<float>& pixel) const;
        //! Set the (multichannel) value at a pixel.
        void set_value( int i, int j, const std::vector<float>& pixel);

        //! Copy constructor. Clears existing content.
        ImgProc(const ImgProc& v);
        //! Copy assignment. Clears existing content.
        ImgProc& operator=(const ImgProc& v);

        friend void swap(ImgProc& u, ImgProc& v);

        //! multiplies all pixels and channels by a value
        void operator*=(float v);
        //! divides all pixels and channels by a value
        void operator/=(float v);
        //! adds a value to all pixels and channels
        void operator+=(float v);
        //! subtracts a value from all pixels and channels
        void operator-=(float v);

        //! converts image to its compliment in-place
        void compliment();
        //==============================================================================
        //My Manipulations
        void brightness_up();
        void brightness_down();
        void bias_up();
        void bias_down();
        void gamma_up();
        void gamma_down();
        void grayscale();
        void quantize();
        void rms_contrast();
        void undo_step();
        void output_exr();
        void output_jpeg();
        void compute_msg(int n);
        void equal_histogram();
        void compute_PDF();
        void compute_CDF();
        void log_quench();
        void doIFS();
        //==============================================================================

         //! indexing to a particular pixel and channel
        long index(int i, int j, int c) const;
        //! indexing to a particular pixel
        long index(int i, int j) const;

        //! returns raw pointer to data (dangerous)
        float* raw(){ return img_data; }

      private:

        int Nx, Ny, Nc;
        long Nsize;
        long single_Nsize;
        float * img_data;
        std::vector<int> undo_vec;
        std::vector<std::vector<float>> temp_data;
        int histogram_bins_num = 10;
        int hist[10];//={0};
        float pdf[10];//={0};
        float cdf[10];//={0};
        float delta_i;
        std::vector<float> min_v;
        float min_value;
        std::vector<float> max_v;
        float max_value;
        int bin_index;
    };


    //! swaps content of two images
    void swap(ImgProc& u, ImgProc& v);
    //==============================================================================
    void FractalFlameIFS(const size_t nb_iterations,const std::vector<img::IFSFunction*>& func_list, const ColorLUT& lut, ImgProc& out);
    //==============================================================================

}


#endif
