

#ifndef IFSFUNCTION_H
#define IFSFunction_H

#include <string>
#include <vector>
#include <imgproc.h>
#include <iostream>

//#ifndef PI
//#define PI
//float Pii = 3.1415926;
//#endif
// struct Point
// {
//   float x;
//   float y;
// };

namespace img
{
  // struct Point
  // {
  //   float x;
  //   float y;
  // };
    class IFSFunction
    {

      public:

        //! Construct with no content
        IFSFunction(){};
       ~IFSFunction(){};

       //Point operator()(Point P)
       std::vector<float> operator()(std::vector<float> P)
       {
         std::cout<<"Px1:       "<<P[0]<<",Py1:      "<<P[1]<<std::endl;
         //Point pp;
         std::vector<float> pp;
         pp.push_back(0.0);
         pp.push_back(0.0);
         //float r = sqrt(P.x*P.x + P.y*P.y);
         //float r = sqrt(P[0]*P[0] + P[1]*P[1]);
         pp[0] = P[0]+0.1;//std::pow(P[1], 5) / std::pow(r, 5);
         //pp[1] = -1.0*std::pow(P[0], 5) / std::pow(r, 5);
         //p.x = exp(P.x-1.0)*cos(3.1415926*P.y);
         //p.y = exp(P.x-1.0)*sin(3.1415926*P.y);
         std::cout<<"Px2:       "<<pp[0]<<",Py2:      "<<pp[1]<<std::endl;
         return pp;
       }
       std::vector<float> color()
       {
         std::vector<float> v={0.5, 0.5, 0.5};
         return v;
       }
       //Generate a new point from the input point position
       //virtual Point operator()(Point P){std::cout << "calling IFSFunction virtual operator function" << '\n';}
       //virtual std::vector<float> color(){std::cout << "calling IFSFunction virtual operator function2" << '\n';}
       private:

          int Nx, Ny;
     };

    // class Exponential : public IFSFunction
    // {
    //   public:
    //
    //     //! Construct with no content
    //     Exponential();
    //    ~Exponential();
    //
    //    //Generate a new point from the input point position
    //    Point operator()(Point P);
    //    std::vector<float> color()
    //    {
    //      std::vector<float> v={0.5, 0.5, 0.5};
    //      return v;
    //    }
    //
    // //  private:
    // };
}


#endif
