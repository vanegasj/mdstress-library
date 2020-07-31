/*=========================================================================

  Module    : MDStress
  File      : mds_basicops.h
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

#ifndef mds_basicops_h
#define mds_basicops_h

#include "mds_common.h"
#include "mds_defines.h"

namespace mds
{
    static inline void sumarray ( darray a, darray b, darray c)
    {
        c[0] = a[0]+b[0];
        c[1] = a[1]+b[1];
        c[2] = a[2]+b[2];
    }

    static inline void diffarray ( darray a, darray b, darray c)
    {
        c[0] = a[0]-b[0];
        c[1] = a[1]-b[1];
        c[2] = a[2]-b[2];
    }

    static inline void diffarray ( darray a, darray b, darray c, dmatrix box, barray periodic)
    {
        c[0] = a[0]-b[0];
        c[1] = a[1]-b[1];
        c[2] = a[2]-b[2];

        if (periodic[3] == true)
        {
            for (int i = 0; i < mds_ndim; i++)
            {
                if (periodic[i] == true)
                {
                    while (c[i] > 0.5*box[i][i])
                        c[i] -= box[i][i];
                    while (c[i] <= -0.5*box[i][i])
                        c[i] += box[i][i];
                }
            }
        }
    }

    static inline void scalearray ( darray b, double a, darray c)
    {
        c[0] = a * b[0];
        c[1] = a * b[1];
        c[2] = a * b[2];
    }

    static inline double normarray ( darray a )
    {
        return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    }

    static inline void copyarray ( darray a, darray b)
    {
        b[0] = a[0];
        b[1] = a[1];
        b[2] = a[2];
    }

    static inline void summatrix ( dmatrix a, dmatrix b, dmatrix c)
    {
        c[0][0] = a[0][0] + b[0][0];
        c[0][1] = a[0][1] + b[0][1];
        c[0][2] = a[0][2] + b[0][2];
        c[1][0] = a[1][0] + b[1][0];
        c[1][1] = a[1][1] + b[1][1];
        c[1][2] = a[1][2] + b[1][2];
        c[2][0] = a[2][0] + b[2][0];
        c[2][1] = a[2][1] + b[2][1];
        c[2][2] = a[2][2] + b[2][2];
    }

    static inline void copymatrix ( dmatrix a, dmatrix b)
    {
        b[0][0] = a[0][0];
        b[0][1] = a[0][1];
        b[0][2] = a[0][2];
        b[1][0] = a[1][0];
        b[1][1] = a[1][1];
        b[1][2] = a[1][2];
        b[2][0] = a[2][0];
        b[2][1] = a[2][1];
        b[2][2] = a[2][2];
    }

    static inline void scalematrix ( dmatrix b, double a, dmatrix c)
    {
        c[0][0] = a * b[0][0];
        c[0][1] = a * b[0][1];
        c[0][2] = a * b[0][2];
        c[1][0] = a * b[1][0];
        c[1][1] = a * b[1][1];
        c[1][2] = a * b[1][2];
        c[2][0] = a * b[2][0];
        c[2][1] = a * b[2][1];
        c[2][2] = a * b[2][2];
    }
    
    #define scalesummatrix(a,b,c)\
        c[0][0] += a * b[0][0];\
        c[0][1] += a * b[0][1];\
        c[0][2] += a * b[0][2];\
        c[1][0] += a * b[1][0];\
        c[1][1] += a * b[1][1];\
        c[1][2] += a * b[1][2];\
        c[2][0] += a * b[2][0];\
        c[2][1] += a * b[2][1];\
        c[2][2] += a * b[2][2]
    
    #define scalesumarray(a,b,c)\
        c[0] += a * b[0];\
        c[1] += a * b[1];\
        c[2] += a * b[2];
    
    static inline void inversematrix ( dmatrix A, dmatrix iA)
    {
        double det = 0.0;
        double iDet = 0.0;
        
        //It only works for 3D
        det  += (A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]));
        det  += (A[0][1]*(A[1][2]*A[2][0] - A[1][0]*A[2][2]));
        det  += (A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]));
        
        iDet = 1.0/det;
        
        iA[0][0] = ((A[1][1] * A[2][2]) - (A[1][2]*A[2][1])) * iDet;
        iA[0][1] = ((A[1][2] * A[2][0]) - (A[1][0]*A[2][2])) * iDet;
        iA[0][2] = ((A[1][0] * A[2][1]) - (A[1][1]*A[2][0])) * iDet;
        iA[1][0] = ((A[2][1] * A[0][2]) - (A[2][2]*A[0][1])) * iDet;
        iA[1][1] = ((A[2][2] * A[0][0]) - (A[2][0]*A[0][2])) * iDet;
        iA[1][2] = ((A[2][0] * A[0][1]) - (A[2][1]*A[0][0])) * iDet;
        iA[2][0] = ((A[0][1] * A[1][2]) - (A[0][2]*A[1][1])) * iDet;
        iA[2][1] = ((A[0][2] * A[1][0]) - (A[0][0]*A[1][2])) * iDet;
        iA[2][2] = ((A[0][0] * A[1][1]) - (A[0][1]*A[1][0])) * iDet;
    }

    static inline void zeromatrix ( dmatrix A )
    {
        A[0][0] = 0.0;
        A[0][1] = 0.0;
        A[0][2] = 0.0;
        A[1][0] = 0.0;
        A[1][1] = 0.0;
        A[1][2] = 0.0;
        A[2][0] = 0.0;
        A[2][1] = 0.0;
        A[2][2] = 0.0;
    }
    
    static inline void zeroarray ( darray A )
    {
        A[0] = 0.0;
        A[1] = 0.0;
        A[2] = 0.0;
    }

    static inline bool iszeromatrix ( dmatrix A )
    {   
        for ( int i = 0; i < mds_ndim; i++ )
            for ( int j = 0; j < mds_ndim; j++ )
                if (A[i][j] > mds_eps) return false;
                
        return true;
    }

    // Modulo operation
    #define modulo(a,b) ((a+b) % b)
}

#endif // mds_basicops_h
