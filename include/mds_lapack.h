/*=========================================================================

  Module    : MDStress
  File      : mds_lapack.h
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
#ifndef __mdslapack_h
#define __mdslapack_h

#include <math.h>
#include <iostream>

#ifndef F77_FUNC
#define F77_FUNC(name,NAME) name ## _
#endif

namespace mds
{ 
    extern "C" void F77_FUNC(dgelsd,DGELSD)( int *M, int *N, int *NRHS, double *A, int *LDA, double *B, int *LDB, double *S, double *RCOND, int *RANK, double *WORK, int *LWORK, int *IWORK, int *INFO );
    extern "C" void F77_FUNC(dgeqp3,DGEQP3)(int *M, int *N,double *A, int *LDA, int *JPVT, double *TAU, double *WORK, int *LWORK, int *INFO);   
    extern "C" void F77_FUNC(dormqr,DORMQR)(char* SIDE, char* TRANS, int* M, int* N, int* K, double* A, int* LDA, double* TAU, double* C, int* LDC, double* WORK, int* LWORK, int* INFO );
}
    
/** \class Lapack
* \brief This class is used to solve linear systems by means of the dgelsd function and compute the QR decomposition
* of a given matrix. Sizes of auxiliary vectors are created at class creation to save computing time. Do not use this
* class if you are not sure that these sizes fit your requirements.
*/
class mds::Lapack
{  
    
public:

    /** Calls dgelsd to solve an under/over-determined system of equations */
    int SolveMinNorm   ( int m, int n, double* A, double *b );
    /** Calls dgeqp3 to compute the QR decomposition of a matrix through column pivoting */
    int QRDecomposition( int m, int n, double* A, double *tau );
    /** Calls domqr to multiply Q or QT times a matrix/vector using the compressed structure by LAPACK*/
    int MultiplyQb(char side, char trans, int m, int n, int k, double *A, int rA, double *tau, double* C);
    /** Computes the projection onto the image of the transpose of matrix A (eliminates components in the NULL space)*/
    int QQTb( int rowsA, int colsA, int rowsB, int rank, double *A, double *b );
    
    /** Constructor */
    Lapack( int nRowMax, int nColMax );

    /** Destructor */
    ~Lapack( );
        
private:

    int     liwork; //< Size of iwork
    int     lwork;  //< Size of work
    int    *iwork;  //< int auxiliary array
    double *work;   //< double auxiliary array
    int    *iarray; //< int auxiliary array (permutations in the QR decomposition)
    double *darray; //< double auxiliary array (householder reflections)
    double  eps1;   //< mds_eps
};

#endif // __mdslapack_h

