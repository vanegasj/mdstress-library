/*=========================================================================

  Module    : MDStress
  Authors   : A. Torres-Sanchez and J. M. Vanegas
  Modified  : B. Himberg and A. L. Lewis
  Purpose   : Compute the local stress from MD trajectories
  Date      : Aug-18-2025
  Version   :
  Changes   :

     http://mdstress.org

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

     Report any bugs to:
     juan.m.vanegas@gmail.com
=========================================================================*/

#ifndef mds_cmenger_h 
#define mds_cmenger_h

#include "mds_common.h"
#include "mds_defines.h"

namespace mds
{
    //----------------------------------------------------------------------------------------
    // FIVE BODY POTENTIALS -> CALEY-MENGER DERIVATIVES FOR cCFD
    // Calculate the derivative of the Caley-Menger determinant for the 5-particles case with respect to d12
    static inline real_mds CaleyMenger5Der(
            real_mds d12, real_mds d13, real_mds d14, real_mds d15,
                          real_mds d23, real_mds d24, real_mds d25,
                                        real_mds d34, real_mds d35,
                                                      real_mds d45) {
        return realval_mds(-4.0)* d12 *( d45*d45*(d35*d35*(-(realval_mds(2.0)*d12*d12) + d23*d23 + d24*d24 - realval_mds(2.0)*d34*d34) + d34*d34*(-(realval_mds(2.0)*d12*d12) + d23*d23 + d25*d25) + d13*d13*(-(realval_mds(2.0)*d23*d23) + d24*d24 + d25*d25 + d34*d34 + d35*d35)) + d45*d45*d45*d45*(-(-d12*d12 + d13*d13 + d23*d23)) - (d34 - d35)*(d34 + d35)*(d34*d34*(d25*d25 - d12*d12) + d35*d35*(d12 - d24)*(d12 + d24) + d13*d13*(d24 - d25)*(d24 + d25)) + d14*d14*(d23*d23*(-d34*d34 + d35*d35 + d45*d45) + d35*d35*(-(realval_mds(2.0)*d24*d24) + d34*d34 - d35*d35 + d45*d45) + d25*d25*(d34*d34 + d35*d35 - d45*d45)) + d15*d15*(d23*d23*(d34*d34 - d35*d35 + d45*d45) + d24*d24*(d34*d34 + d35*d35 - d45*d45) + d34*d34*(-(realval_mds(2.0)*d25*d25) - d34*d34 + d35*d35 + d45*d45)));
    }

    //Calculate the normal to the shape space for the 5-particles case
    static inline void ShapeSpace5Normal(
            real_mds d12, real_mds d13, real_mds d14, real_mds d15,
                          real_mds d23, real_mds d24, real_mds d25,
                                        real_mds d34, real_mds d35,
                                                      real_mds d45, real_mds normal[10]) {
        normal[0] = CaleyMenger5Der(d12, d13, d14, d15, d23, d24, d25, d34, d35, d45);
        normal[1] = CaleyMenger5Der(d13, d12, d14, d15, d23, d34, d35, d24, d25, d45);
        normal[2] = CaleyMenger5Der(d14, d12, d13, d15, d24, d34, d45, d23, d25, d35);
        normal[3] = CaleyMenger5Der(d15, d12, d13, d14, d25, d35, d45, d23, d24, d34);
        normal[4] = CaleyMenger5Der(d23, d12, d24, d25, d13, d34, d35, d14, d15, d45);
        normal[5] = CaleyMenger5Der(d24, d12, d23, d25, d14, d34, d45, d13, d15, d35);
        normal[6] = CaleyMenger5Der(d25, d12, d23, d24, d15, d35, d45, d13, d14, d34);
        normal[7] = CaleyMenger5Der(d34, d13, d23, d35, d14, d24, d45, d12, d15, d25);
        normal[8] = CaleyMenger5Der(d35, d13, d23, d34, d15, d25, d45, d12, d14, d24);
        normal[9] = CaleyMenger5Der(d45, d14, d24, d34, d15, d25, d35, d12, d13, d23);
        
        mds::real_mds normarray = realval_mds(0.0);
        for (int i = 0; i < 10; i++ ) {
            normarray += normal[i]*normal[i];
        }
        normarray = sqrt(normarray);

        if (normarray > mds_eps) {
            for (int i = 0; i < 10; i++ ) {
                normal[i] /= normarray;
            }
        }
    }
}

#endif // mds_cmenger_h
