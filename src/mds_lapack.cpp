/*=========================================================================

  Module    : MDStress
  File      : mds_lapack.cpp
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

#include "mds_defines.h"
#include "mds_lapack.h"


//Constructor
mds::Lapack::Lapack( int nRowMax, int nColMax )
{
    int nRhsMax = 1;
    int liwork;
    int nlvl;
    double smlsiz = 25.0;

    if ( nRowMax <= 0 || nColMax <= 0 )
        std::cout << "ERROR::Lapack Input invalid (rowmax, colmax or rhsmax <= 0)\n";
    nlvl = mds_max( 0, static_cast<int>( log2( mds_min( nRowMax,nColMax )/(smlsiz+1) ) ) + 1 );
    
    if ( nRowMax > nColMax )
        this->lwork = 12*nColMax + 2*nColMax*smlsiz + 8*nColMax*nlvl + nColMax*nRhsMax + (smlsiz+1)*(smlsiz+1);
    else
        this->lwork = 12*nRowMax + 2*nRowMax*smlsiz + 8*nRowMax*nlvl + nRowMax*nRhsMax + (smlsiz+1)*(smlsiz+1);
    
    this->liwork = mds_max(1, 3 * mds_min(nRowMax,nColMax) * nlvl + 11 * mds_min(nRowMax,nColMax));
    
    this->work   = new double [this->lwork];  //Shared among dgelsd, dgeqp3 and dormqr
    this->iwork  = new int    [this->liwork]; //Used in dgelsd
    this->darray = new double [nRowMax];      //Used for dgelsd and dgeqp3
    this->iarray = new int    [nRowMax];      //This is used for dgeqp3 (we use it for the transpose!!!)
    
    this->eps1 = mds_eps;
    
}

//Destructor
mds::Lapack::~Lapack( )
{
    if (this->work   != NULL ) delete [] this->work;
    if (this->iwork  != NULL ) delete [] this->iwork;
    if (this->darray != NULL ) delete [] this->darray;
    if (this->iarray != NULL ) delete [] this->iarray;
    
    this->work   = NULL;
    this->iwork  = NULL;
    this->darray = NULL;
    this->iarray = NULL;
}

//Calls dgelsd to solve an under/over-determined system of equations
int mds::Lapack::SolveMinNorm(int m, int n, double* A,double *b ) 
{
    int     leading;
    int     rank;
    int     nRHS = 1;
    int     info ;

    leading = mds_max(m,n);
    
    info  =  0;
    
    F77_FUNC(dgelsd,DGELSD)(&m, &n, &nRHS, A, &m, b, &leading, this->darray, &this->eps1, &rank, work, &this->lwork, this->iwork, &info );
    
    return info;
}

// Calls dgeqp3 to compute the QR decomposition of a matrix through column pivoting
int mds::Lapack::QRDecomposition(int m, int n, double* A, double *tau) 
{
    int info;

    for (int i = 0; i < n; i++ ) 
        this->iarray[i] = 0;
        
    info = 0;
    F77_FUNC(dgeqp3,DGEQP3)(&m, &n, A, &m, this->iarray, tau, this->work, &this->lwork, &info);
            
    return info;
}

// Calls domqr to multiply Q or QT times a matrix/vector using the compressed structure by LAPACK
int mds::Lapack::MultiplyQb(char side, char trans, int m, int n, int k, double *A, int rA, double *tau, double* C)
{
    int info;
    
    info = 0;
    
    F77_FUNC(dormqr,DORMQR)(&side, &trans, &m, &n, &k, A, &rA, tau, C, &m, this->work, &this->lwork, &info);

    return info;
}
    
// Computes the projection onto the image of the transpose of matrix A (eliminates components in the NULL space)
int mds::Lapack::QQTb( int rowsA, int colsA, int rowsB, int rank, double *A, double *b ) 
{
    
    this->QRDecomposition(rowsA, colsA, A, this->darray);
    
    this->MultiplyQb('L', 'T', rowsB, 1, rank, A, rowsA, this->darray, b);   
    
    for (int i = rank; i < rowsB; i++ )
        b[i] = 0.0;
    
    this->MultiplyQb('L', 'N', rowsB, 1, rank, A, rowsA, this->darray, b);
    
    return 0;
}