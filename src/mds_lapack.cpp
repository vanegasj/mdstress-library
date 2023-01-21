/*=========================================================================

  Module    : MDStress
  File      : mds_lapack.cpp
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

#include "mds_lapack.h"


//Constructor
mds::Lapack::Lapack( int nRowMax, int nColMax )
{
    int nRhsMax = 1;
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
    
    this->work   = new double [this->lwork]();  //Shared among dgelsd, dgeqp3 and dormqr
    this->iwork  = new int    [this->liwork](); //Used in dgelsd
    this->darray = new double [nRowMax]();      //Used for dgelsd and dgeqp3
    this->iarray = new int    [nRowMax]();      //This is used for dgeqp3 (we use it for the transpose!!!)
    
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
