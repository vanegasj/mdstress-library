/*=========================================================================

  Module    : MDStress
  File      : mds_basicops.h
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

#ifndef __basicops_h
#define __basicops_h

#include "mds_defines.h"

namespace mds
{
    //AUXILIARY FUNCTIONS
    void sumarray ( darray a, darray b, darray c);

    void diffarray ( darray a, darray b, darray c);

    void scalearray ( darray b, double a, darray c);

    double normarray ( darray a );
    
    void summatrix ( dmatrix a, dmatrix b, dmatrix c);

    void scalematrix ( dmatrix b, double a, dmatrix c);
    
    void inversematrix ( dmatrix A, dmatrix iA);

    void zeromatrix ( dmatrix A );
    
    bool iszeromatrix ( dmatrix A );
    
    int modulo (int a, int b);
}

#endif // __basicops_h