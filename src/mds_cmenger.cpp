/*=========================================================================

  Module    : MDStress
  File      : mds_cmenger.cpp
  Authors   : A. Torres-Sanchez and J. M. Vanegas
  Modified  :
  Purpose   : Compute the local stress from MD trajectories
  Date      : 25/03/2015
  Version   :
  Changes   :

     http://mdstress.org

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  

     Please, report any bug to either of us:
     torres.sanchez.a@gmail.com
     juan.m.vanegas@gmail.com
=========================================================================*/

#include "mds_cmenger.h"
//----------------------------------------------------------------------------------------
// FIVE BODY POTENTIALS -> CALEY-MENGER DERIVATIVES FOR cCFD
// Calculate the derivative of the Caley-Menger determinant for the 5-particles case with respect to d12
mds::real_int mds::CaleyMenger5Der(mds::real_int d12,mds::real_int d13,mds::real_int d14,mds::real_int d15,mds::real_int d23,mds::real_int d24,mds::real_int d25,mds::real_int d34,mds::real_int d35,mds::real_int d45)
{
    return realval_int(-4.0)* d12 *( d45*d45*(d35*d35*(-(realval_int(2.0)*d12*d12) + d23*d23 + d24*d24 - realval_int(2.0)*d34*d34) + d34*d34*(-(realval_int(2.0)*d12*d12) + d23*d23 + d25*d25) + d13*d13*(-(realval_int(2.0)*d23*d23) + d24*d24 + d25*d25 + d34*d34 + d35*d35)) + d45*d45*d45*d45*(-(-d12*d12 + d13*d13 + d23*d23)) - (d34 - d35)*(d34 + d35)*(d34*d34*(d25*d25 - d12*d12) + d35*d35*(d12 - d24)*(d12 + d24) + d13*d13*(d24 - d25)*(d24 + d25)) + d14*d14*(d23*d23*(-d34*d34 + d35*d35 + d45*d45) + d35*d35*(-(realval_int(2.0)*d24*d24) + d34*d34 - d35*d35 + d45*d45) + d25*d25*(d34*d34 + d35*d35 - d45*d45)) + d15*d15*(d23*d23*(d34*d34 - d35*d35 + d45*d45) + d24*d24*(d34*d34 + d35*d35 - d45*d45) + d34*d34*(-(realval_int(2.0)*d25*d25) - d34*d34 + d35*d35 + d45*d45)));
}

//Calculate the normal to the shape space for the 5-particles case
void mds::ShapeSpace5Normal(mds::real_int d12,mds::real_int d13,mds::real_int d14,mds::real_int d15,mds::real_int d23,mds::real_int d24,mds::real_int d25,mds::real_int d34,mds::real_int d35,mds::real_int d45, mds::real_int *normal)
{
    mds::real_int normarray;
    int nDist = 10;
    int i;

    mds::real_int d12_, d13_, d14_, d15_, d23_, d24_, d25_, d34_, d35_, d45_;

    //No change
    d12_ = d12; d13_=d13; d14_=d14; d15_=d15; d23_=d23; d24_=d24; d25_=d25; d34_=d34; d35_=d35; d45_=d45;
    normal[0] = CaleyMenger5Der(d12_,d13_,d14_,d15_,d23_,d24_,d25_,d34_,d35_,d45_);

    d12_ = d13; d13_=d12; d14_=d14; d15_=d15; d23_=d23; d24_=d34; d25_=d35; d34_=d24; d35_=d25; d45_=d45;
    normal[1] = CaleyMenger5Der(d12_,d13_,d14_,d15_,d23_,d24_,d25_,d34_,d35_,d45_);

    d12_ = d14; d13_=d12; d14_=d13; d15_=d15; d23_=d24; d24_=d34; d25_=d45; d34_=d23; d35_=d25; d45_=d35;
    normal[2] = CaleyMenger5Der(d12_,d13_,d14_,d15_,d23_,d24_,d25_,d34_,d35_,d45_);

    d12_ = d15; d13_=d12; d14_=d13; d15_=d14; d23_=d25; d24_=d35; d25_=d45; d34_=d23; d35_=d24; d45_=d34;
    normal[3] = CaleyMenger5Der(d12_,d13_,d14_,d15_,d23_,d24_,d25_,d34_,d35_,d45_);

    d12_ = d23; d13_=d12; d14_=d24; d15_=d25; d23_=d13; d24_=d34; d25_=d35; d34_=d14; d35_=d15; d45_=d45;
    normal[4] = CaleyMenger5Der(d12_,d13_,d14_,d15_,d23_,d24_,d25_,d34_,d35_,d45_);

    d12_ = d24; d13_=d12; d14_=d23; d15_=d25; d23_=d14; d24_=d34; d25_=d45; d34_=d13; d35_=d15; d45_=d35;
    normal[5] = CaleyMenger5Der(d12_,d13_,d14_,d15_,d23_,d24_,d25_,d34_,d35_,d45_);

    d12_ = d25; d13_=d12; d14_=d23; d15_=d24; d23_=d15; d24_=d35; d25_=d45; d34_=d13; d35_=d14; d45_=d34;
    normal[6] = CaleyMenger5Der(d12_,d13_,d14_,d15_,d23_,d24_,d25_,d34_,d35_,d45_);

    d12_ = d34; d13_=d13; d14_=d23; d15_=d35; d23_=d14; d24_=d24; d25_=d45; d34_=d12; d35_=d15; d45_=d25;
    normal[7] = CaleyMenger5Der(d12_,d13_,d14_,d15_,d23_,d24_,d25_,d34_,d35_,d45_);

    d12_ = d35; d13_=d13; d14_=d23; d15_=d34; d23_=d15; d24_=d25; d25_=d45; d34_=d12; d35_=d14; d45_=d24;
    normal[8] = CaleyMenger5Der(d12_,d13_,d14_,d15_,d23_,d24_,d25_,d34_,d35_,d45_);

    d12_ = d45; d13_=d14; d14_=d24; d15_=d34; d23_=d15; d24_=d25; d25_=d35; d34_=d12; d35_=d13; d45_=d23;
    normal[9] = CaleyMenger5Der(d12_,d13_,d14_,d15_,d23_,d24_,d25_,d34_,d35_,d45_);
    
    normarray = realval_int(0.0);
    for ( i = 0; i < nDist; i++ )
    {
        normarray += normal[i]*normal[i];
    }
    normarray = sqrt(normarray);

    if (normarray > mds_eps)
    {
        for ( i = 0; i < nDist; i++ )
        {
            normal[i] /= normarray;
        }
    }
}
