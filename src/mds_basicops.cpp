/*=========================================================================

  Module    : MDStress
  File      : mds_basicops.cpp
  Authors   : A. Torres-Sanchez and J. M. Vanegas
  Modified  :
  Purpose   : Compute the local stress from MD trajectories
  Date      : 25/03/2015
  Version   :
  Changes   :

     http://www.lacan.upc.edu/LocalStressFromMD

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  

     Please, report any bug to either of us:
     torres.sanchez.a@gmail.com
     juan.m.vanegas@gmail.com
=========================================================================*/

#include "mds_basicops.h"
#include <math.h>

void mds::sumarray ( darray a, darray b, darray c)
{
    for (int i = 0; i < mds_ndim; i++)
        c[i] = a[i]+b[i];
}

void mds::diffarray ( darray a, darray b, darray c)
{
    for (int i = 0; i < mds_ndim; i++)
        c[i] = a[i]-b[i];
}

void mds::scalearray ( darray b, double a, darray c)
{
    for (int i = 0; i < mds_ndim; i++)
        c[i] = a * b[i];
}

double mds::normarray ( darray a )
{
    return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

void mds::summatrix ( dmatrix a, dmatrix b, dmatrix c)
{
    for (int i = 0; i < mds_ndim; i++)
    {
        for (int j = 0; j < mds_ndim; j++)
            c[i][j] = a[i][j] + b[i][j];
    }
}

void mds::scalematrix ( dmatrix b, double a, dmatrix c)
{
    for (int i = 0; i < mds_ndim; i++)
    {
        for (int j = 0; j < mds_ndim; j++)
            c[i][j] = a * b[i][j];
    }
}

void mds::inversematrix ( dmatrix A, dmatrix iA)
{
    double det, iDet;
    
    //It only works for 3D
    for ( int i = 0; i < mds_ndim; i++ )
        det  += (A[0][i]*(A[1][(i+1)%3]*A[2][(i+2)%3] - A[1][(i+2)%3]*A[2][(i+1)%3]));
    
    iDet = 1.0/det;
    
    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++ )
            iA[i][j] = ((A[(i+1)%3][(j+1)%3] * A[(i+2)%3][(j+2)%3]) - (A[(i+1)%3][(j+2)%3]*A[(i+2)%3][(j+1)%3])) * iDet;
}

void mds::zeromatrix ( dmatrix A )
{
    for ( int i = 0; i < mds_ndim; i++ )
        for ( int j = 0; j < mds_ndim; j++ )
            A[i][j] = 0.0;
}

bool mds::iszeromatrix ( dmatrix A )
{   
    for ( int i = 0; i < mds_ndim; i++ )
        for ( int j = 0; j < mds_ndim; j++ )
            if (A[i][j] > mds_eps) return false;
            
    return true;
}

// Modulo operation
int mds::modulo (int a, int b)
{
    int ret;
    ret = a % b;
    if(ret < 0)
    ret+=b;
    return ret;
}