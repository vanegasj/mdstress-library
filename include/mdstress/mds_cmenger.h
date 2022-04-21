/*=========================================================================

  Module    : MDStress
  File      : mds_cmenger.h
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

#ifndef mds_cmenger_h 
#define mds_cmenger_h

#include "mds_common.h"
#include "mds_defines.h"

namespace mds
{
    real_int CaleyMenger5Der(real_int d12,real_int d13,real_int d14,real_int d15,real_int d23,real_int d24,real_int d25,real_int d34,real_int d35,real_int d45);
    void   ShapeSpace5Normal(real_int d12,real_int d13,real_int d14,real_int d15,real_int d23,real_int d24,real_int d25,real_int d34,real_int d35,real_int d45, real_int *normal);
}

#endif // mds_cmenger_h
