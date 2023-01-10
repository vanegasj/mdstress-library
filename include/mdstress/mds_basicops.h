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
    template<class float_type>
    static inline void sumarray3 ( const float_type a[3], const float_type b[3], float_type c[3])
    {
        c[0] = a[0]+b[0];
        c[1] = a[1]+b[1];
        c[2] = a[2]+b[2];
    }

    template<class float_type>
    static inline void sumarray6 ( const float_type a[6], const float_type b[6], float_type c[6])
    {
        c[0] = a[0]+b[0];
        c[1] = a[1]+b[1];
        c[2] = a[2]+b[2];
        c[3] = a[3]+b[3];
        c[4] = a[4]+b[4];
        c[5] = a[5]+b[5];
    }

    template<class float_type_ab, class float_type_c>
    static inline void diffarray3 ( const float_type_ab a[3], const float_type_ab b[3], float_type_c c[3])
    {
        c[0] = a[0]-b[0];
        c[1] = a[1]-b[1];
        c[2] = a[2]-b[2];
    }

    template<class array_type_ab, class array_type_c, class matrix_type>
    static inline void diffarray3 ( const array_type_ab & a, const array_type_ab & b, array_type_c c, const matrix_type & box, const barray & periodic)
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
                    while (c[i] > (real_mds)(0.5)*box[i][i])
                        c[i] -= box[i][i];
                    while (c[i] <= (real_mds)(-0.5)*box[i][i])
                        c[i] += box[i][i];
                }
            }
        }
    }

    template<class float_type>
    static inline void scalearray3 ( const float_type b[3], float_type a, float_type c[3])
    {
        c[0] = a * b[0];
        c[1] = a * b[1];
        c[2] = a * b[2];
    }

    template<class float_type>
    static inline float_type normarray3 ( const float_type a[3] )
    {
        return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    }

    template<class float_type_a, class float_type_b>
    static inline void copyarray3 ( const float_type_a a[3], float_type_b b[3])
    {
        b[0] = (float_type_b)(a[0]);
        b[1] = (float_type_b)(a[1]);
        b[2] = (float_type_b)(a[2]);
    }

    template<class float_type>
    static inline void summatrix3 ( const float_type a[3][3], const float_type b[3][3], float_type c[3][3])
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

    template<class float_type>
    static inline void summatrix6(const float_type a[6][6], const float_type b[6][6], float_type c[6][6])
    {
        for (int i = 0; i < 6; ++i)
        {
            c[i][0] = a[i][0] + b[i][0];
            c[i][1] = a[i][1] + b[i][1];
            c[i][2] = a[i][2] + b[i][2];
            c[i][3] = a[i][3] + b[i][3];
            c[i][4] = a[i][4] + b[i][4];
            c[i][5] = a[i][5] + b[i][5];
        }
    }

    // Stiffness matrix in Voigt notation
    // 0 = xx; 1 = yy; 2 = zz; 3 = yz or zy; 4 = xz or zx; 5 = xy or yx
    // All indices                         Voigt indices           Stress indices
    // ( xxxx xxyy xxzz xxyz xxxz xxxy ) = ( 00 01 02 03 04 05 ) = [ 0000 0011 0022 0012 0002 0001 ]
    // ( yyxx yyyy yyzz yyyz yyxz yyxy ) = ( 10 11 12 13 14 15 ) = [ 1100 1111 1122 1112 1102 1101 ]
    // ( zzxx zzyy zzzz zzyz zzxz zzxy ) = ( 20 21 22 23 24 25 ) = [ 2200 2211 2222 2212 2202 2201 ]
    // ( yzxx yzyy yzzz yzyz yzxz yzxy ) = ( 30 31 32 33 34 35 ) = [ 1200 1211 1222 1212 1202 1201 ]
    // ( xzxx xzyy xzzz xzyz xzxz xzxy ) = ( 40 41 42 43 44 45 ) = [ 0200 0211 0222 0212 0202 0201 ]
    // ( xyxx xyyy xyzz xyyz xyxz xyxy ) = ( 50 51 52 53 54 55 ) = [ 0100 0111 0122 0112 0102 0101 ]
    //
    template<class float_type>
    static inline void summatrix6matrixsq(const float_type a[6][6], const float_type b[3][3] , float_type c[6][6])
    {
        for (int i = 0; i < 3; ++i)
        {
            c[i][0] = a[i][0] + b[i][i]*b[0][0];
            c[i][1] = a[i][1] + b[i][i]*b[1][1];
            c[i][2] = a[i][2] + b[i][i]*b[2][2];
            c[i][3] = a[i][3] + b[i][i]*b[1][2];
            c[i][4] = a[i][4] + b[i][i]*b[0][2];
            c[i][5] = a[i][5] + b[i][i]*b[0][1];
        }

        c[3][0] = a[3][0] + b[1][2]*b[0][0];
        c[3][1] = a[3][1] + b[1][2]*b[1][1];
        c[3][2] = a[3][2] + b[1][2]*b[2][2];
        c[3][3] = a[3][3] + b[1][2]*b[1][2];
        c[3][4] = a[3][4] + b[1][2]*b[0][2];
        c[3][5] = a[3][5] + b[1][2]*b[0][1];

        c[4][0] = a[4][0] + b[0][2]*b[0][0];
        c[4][1] = a[4][1] + b[0][2]*b[1][1];
        c[4][2] = a[4][2] + b[0][2]*b[2][2];
        c[4][3] = a[4][3] + b[0][2]*b[1][2];
        c[4][4] = a[4][4] + b[0][2]*b[0][2];
        c[4][5] = a[4][5] + b[0][2]*b[0][1];

        c[5][0] = a[5][0] + b[0][1]*b[0][0];
        c[5][1] = a[5][1] + b[0][1]*b[1][1];
        c[5][2] = a[5][2] + b[0][1]*b[2][2];
        c[5][3] = a[5][3] + b[0][1]*b[1][2];
        c[5][4] = a[5][4] + b[0][1]*b[0][2];
        c[5][5] = a[5][5] + b[0][1]*b[0][1];
    }

    template<class float_type>
    static inline void matrixouterprod6(const float_type a[3][3], const float_type b[3][3], float_type c[6][6])
    {
        c[0][0] = a[0][0]*b[0][0]; c[0][1] = a[0][0]*b[1][1]; c[0][2] = a[0][0]*b[2][2]; c[0][3] = a[0][0]*b[1][2]; c[0][4] = a[0][0]*b[0][2]; c[0][5] = a[0][0]*b[0][1];
        c[1][0] = a[1][1]*b[0][0]; c[1][1] = a[1][1]*b[1][1]; c[1][2] = a[1][1]*b[2][2]; c[1][3] = a[1][1]*b[1][2]; c[1][4] = a[1][1]*b[0][2]; c[1][5] = a[1][1]*b[0][1];
        c[2][0] = a[2][2]*b[0][0]; c[2][1] = a[2][2]*b[1][1]; c[2][2] = a[2][2]*b[2][2]; c[2][3] = a[2][2]*b[1][2]; c[2][4] = a[2][2]*b[0][2]; c[2][5] = a[2][2]*b[0][1];
        c[3][0] = a[1][2]*b[0][0]; c[3][1] = a[1][2]*b[1][1]; c[3][2] = a[1][2]*b[2][2]; c[3][3] = a[1][2]*b[1][2]; c[3][4] = a[1][2]*b[0][2]; c[3][5] = a[1][2]*b[0][1];
        c[4][0] = a[0][2]*b[0][0]; c[4][1] = a[0][2]*b[1][1]; c[4][2] = a[0][2]*b[2][2]; c[4][3] = a[0][2]*b[1][2]; c[4][4] = a[0][2]*b[0][2]; c[4][5] = a[0][2]*b[0][1];
        c[5][0] = a[0][1]*b[0][0]; c[5][1] = a[0][1]*b[1][1]; c[5][2] = a[0][1]*b[2][2]; c[5][3] = a[0][1]*b[1][2]; c[5][4] = a[0][1]*b[0][2]; c[5][5] = a[0][1]*b[0][1];
    }

/*
    static inline void submatrix6sq ( dmatrix6 a, dmatrix b, dmatrix6 c)
    {
        c[0][0] = a[0][0] - b[0][0]*b[0][0]; c[0][1] = a[0][1] - b[0][0]*b[1][1]; c[0][2] = a[0][2] - b[0][0]*b[2][2]; c[0][3] = a[0][3] - b[0][0]*b[1][2]; c[0][4] = a[0][4] - b[0][0]*b[0][2]; c[0][5] = a[0][5] - b[0][0]*b[0][1];
        c[1][0] = a[1][0] - b[1][1]*b[0][0]; c[1][1] = a[1][1] - b[1][1]*b[1][1]; c[1][2] = a[1][2] - b[1][1]*b[2][2]; c[1][3] = a[1][3] - b[1][1]*b[1][2]; c[1][4] = a[1][4] - b[1][1]*b[0][2]; c[1][5] = a[1][5] - b[1][1]*b[0][1];
        c[2][0] = a[2][0] - b[2][2]*b[0][0]; c[2][1] = a[2][1] - b[2][2]*b[1][1]; c[2][2] = a[2][2] - b[2][2]*b[2][2]; c[2][3] = a[2][3] - b[2][2]*b[1][2]; c[2][4] = a[2][4] - b[2][2]*b[0][2]; c[2][5] = a[2][5] - b[2][2]*b[0][1];
        c[3][0] = a[3][0] - b[1][2]*b[0][0]; c[3][1] = a[3][1] - b[1][2]*b[1][1]; c[3][2] = a[3][2] - b[1][2]*b[2][2]; c[3][3] = a[3][3] - b[1][2]*b[1][2]; c[3][4] = a[3][4] - b[1][2]*b[0][2]; c[3][5] = a[3][5] - b[1][2]*b[0][1];
        c[4][0] = a[4][0] - b[0][2]*b[0][0]; c[4][1] = a[4][1] - b[0][2]*b[1][1]; c[4][2] = a[4][2] - b[0][2]*b[2][2]; c[4][3] = a[4][3] - b[0][2]*b[1][2]; c[4][4] = a[4][4] - b[0][2]*b[0][2]; c[4][5] = a[4][5] - b[0][2]*b[0][1];
        c[5][0] = a[5][0] - b[0][1]*b[0][0]; c[5][1] = a[5][1] - b[0][1]*b[1][1]; c[5][2] = a[5][2] - b[0][1]*b[2][2]; c[5][3] = a[5][3] - b[0][1]*b[1][2]; c[5][4] = a[5][4] - b[0][1]*b[0][2]; c[5][5] = a[5][5] - b[0][1]*b[0][1];
    }

    static inline void submatrix6sq ( dmatrix6 a, dmatrix b, dmatrix6 c, double s)
    {
        c[0][0] = s*(a[0][0] - b[0][0]*b[0][0]); c[0][1] = s*(a[0][1] - b[0][0]*b[1][1]); c[0][2] = s*(a[0][2] - b[0][0]*b[2][2]); c[0][3] = s*(a[0][3] - b[0][0]*b[1][2]); c[0][4] = s*(a[0][4] - b[0][0]*b[0][2]); c[0][5] = s*(a[0][5] - b[0][0]*b[0][1]);
        c[1][0] = s*(a[1][0] - b[1][1]*b[0][0]); c[1][1] = s*(a[1][1] - b[1][1]*b[1][1]); c[1][2] = s*(a[1][2] - b[1][1]*b[2][2]); c[1][3] = s*(a[1][3] - b[1][1]*b[1][2]); c[1][4] = s*(a[1][4] - b[1][1]*b[0][2]); c[1][5] = s*(a[1][5] - b[1][1]*b[0][1]);
        c[2][0] = s*(a[2][0] - b[2][2]*b[0][0]); c[2][1] = s*(a[2][1] - b[2][2]*b[1][1]); c[2][2] = s*(a[2][2] - b[2][2]*b[2][2]); c[2][3] = s*(a[2][3] - b[2][2]*b[1][2]); c[2][4] = s*(a[2][4] - b[2][2]*b[0][2]); c[2][5] = s*(a[2][5] - b[2][2]*b[0][1]);
        c[3][0] = s*(a[3][0] - b[1][2]*b[0][0]); c[3][1] = s*(a[3][1] - b[1][2]*b[1][1]); c[3][2] = s*(a[3][2] - b[1][2]*b[2][2]); c[3][3] = s*(a[3][3] - b[1][2]*b[1][2]); c[3][4] = s*(a[3][4] - b[1][2]*b[0][2]); c[3][5] = s*(a[3][5] - b[1][2]*b[0][1]);
        c[4][0] = s*(a[4][0] - b[0][2]*b[0][0]); c[4][1] = s*(a[4][1] - b[0][2]*b[1][1]); c[4][2] = s*(a[4][2] - b[0][2]*b[2][2]); c[4][3] = s*(a[4][3] - b[0][2]*b[1][2]); c[4][4] = s*(a[4][4] - b[0][2]*b[0][2]); c[4][5] = s*(a[4][5] - b[0][2]*b[0][1]);
        c[5][0] = s*(a[5][0] - b[0][1]*b[0][0]); c[5][1] = s*(a[5][1] - b[0][1]*b[1][1]); c[5][2] = s*(a[5][2] - b[0][1]*b[2][2]); c[5][3] = s*(a[5][3] - b[0][1]*b[1][2]); c[5][4] = s*(a[5][4] - b[0][1]*b[0][2]); c[5][5] = s*(a[5][5] - b[0][1]*b[0][1]);
    }
*/

    template<class float_type_a, class float_type_b>
    static inline void copymatrix3 ( const float_type_a a[3][3], float_type_b b[3][3])
    {
        b[0][0] = (float_type_b)(a[0][0]);
        b[0][1] = (float_type_b)(a[0][1]);
        b[0][2] = (float_type_b)(a[0][2]);
        b[1][0] = (float_type_b)(a[1][0]);
        b[1][1] = (float_type_b)(a[1][1]);
        b[1][2] = (float_type_b)(a[1][2]);
        b[2][0] = (float_type_b)(a[2][0]);
        b[2][1] = (float_type_b)(a[2][1]);
        b[2][2] = (float_type_b)(a[2][2]);
    }

    template<class float_type_a, class float_type_b>
    static inline void copymatrix6 ( const float_type_a a[6][6], float_type_b b[6][6])
    {
        for (int i = 0; i < 6; ++i)
        {
            b[i][0] = (float_type_b)(a[i][0]);
            b[i][1] = (float_type_b)(a[i][1]);
            b[i][2] = (float_type_b)(a[i][2]);
            b[i][3] = (float_type_b)(a[i][3]);
            b[i][4] = (float_type_b)(a[i][4]);
            b[i][5] = (float_type_b)(a[i][5]);
        }
    }

    template<class float_type>
    static inline void scalematrix3 ( const float_type b[3][3], float_type a, float_type c[3][3])
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

    template<class float_type>
    static inline void scalematrix6 ( const float_type b[6][6], float_type a, float_type c[6][6])
    {
        for (int i = 0; i < 6; i++)
        {
            c[i][0] = a * b[i][0];
            c[i][1] = a * b[i][1];
            c[i][2] = a * b[i][2];
            c[i][3] = a * b[i][3];
            c[i][4] = a * b[i][4];
            c[i][5] = a * b[i][5];
        }
    }

    template<class float_type>
    static inline void scalesummatrix6(float_type a, const float_type b[6][6], float_type c[6][6])
    {
        for (int i = 0; i < 6; ++i)
        {
            c[i][0] += a * b[i][0];
            c[i][1] += a * b[i][1];
            c[i][2] += a * b[i][2];
            c[i][3] += a * b[i][3];
            c[i][4] += a * b[i][4];
            c[i][5] += a * b[i][5];
        }
    }

    template<class float_type>
    static inline void scalesummatrix6matrixsq(float_type a, const float_type b[6][6], float_type c[6][6])
    {
        for (int i = 0; i < 3; ++i)
        {
            c[i][0] += a * b[i][i]*b[0][0];
            c[i][1] += a * b[i][i]*b[1][1];
            c[i][2] += a * b[i][i]*b[2][2];
            c[i][3] += a * b[i][i]*b[1][2];
            c[i][4] += a * b[i][i]*b[0][2];
            c[i][5] += a * b[i][i]*b[0][1];
        }

        c[3][0] += a * b[1][2]*b[0][0];
        c[3][1] += a * b[1][2]*b[1][1];
        c[3][2] += a * b[1][2]*b[2][2];
        c[3][3] += a * b[1][2]*b[1][2];
        c[3][4] += a * b[1][2]*b[0][2];
        c[3][5] += a * b[1][2]*b[0][1];

        c[4][0] += a * b[0][2]*b[0][0];
        c[4][1] += a * b[0][2]*b[1][1];
        c[4][2] += a * b[0][2]*b[2][2];
        c[4][3] += a * b[0][2]*b[1][2];
        c[4][4] += a * b[0][2]*b[0][2];
        c[4][5] += a * b[0][2]*b[0][1];

        c[5][0] += a * b[0][1]*b[0][0];
        c[5][1] += a * b[0][1]*b[1][1];
        c[5][2] += a * b[0][1]*b[2][2];
        c[5][3] += a * b[0][1]*b[1][2];
        c[5][4] += a * b[0][1]*b[0][2];
        c[5][5] += a * b[0][1]*b[0][1];
    }

    template<class float_type>
    static inline void scalesummatrix3(float_type a, const float_type b[3][3], float_type c[3][3])
    {
        c[0][0] += a * b[0][0];
        c[0][1] += a * b[0][1];
        c[0][2] += a * b[0][2];
        c[1][0] += a * b[1][0];
        c[1][1] += a * b[1][1];
        c[1][2] += a * b[1][2];
        c[2][0] += a * b[2][0];
        c[2][1] += a * b[2][1];
        c[2][2] += a * b[2][2];
    }

    template<class float_type>
    static inline void inversematrix3( const float_type A[3][3], float_type iA[3][3])
    {
        float_type det = (float_type)0.0;
        float_type iDet = (float_type)0.0;
        
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

    template<class float_type>
    static inline void zeromatrix3( float_type A[3][3] )
    {
        A[0][0] = (float_type)(0.0);
        A[0][1] = (float_type)(0.0);
        A[0][2] = (float_type)(0.0);
        A[1][0] = (float_type)(0.0);
        A[1][1] = (float_type)(0.0);
        A[1][2] = (float_type)(0.0);
        A[2][0] = (float_type)(0.0);
        A[2][1] = (float_type)(0.0);
        A[2][2] = (float_type)(0.0);
    }

    template<class float_type>
    static inline void zeromatrix6( float_type A[6][6] )
    {
        for (int i = 0; i < 6; ++i)
        {
            A[i][0] = (float_type)0.0;
            A[i][1] = (float_type)0.0;
            A[i][2] = (float_type)0.0;
            A[i][3] = (float_type)0.0;
            A[i][4] = (float_type)0.0;
            A[i][5] = (float_type)0.0;
        }
    }

    template<class float_type>
    static inline void zeroarray3 ( float_type A[3] )
    {
        A[0] = (float_type)0.0;
        A[1] = (float_type)0.0;
        A[2] = (float_type)0.0;
    }

    template<class float_type>
    static inline bool iszeromatrix3 ( const float_type A[3][3] )
    {   
        for ( int i = 0; i < mds_ndim; i++ )
            for ( int j = 0; j < mds_ndim; j++ )
                if (A[i][j] > (float_type)mds_eps) return false;
                
        return true;
    }

    // Modulo operation
    #define modulo(a,b) ((a+b) % b)
}

#endif // mds_basicops_h
