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
    real_mds CaleyMenger5Der(real_mds d12,real_mds d13,real_mds d14,real_mds d15,real_mds d23,real_mds d24,real_mds d25,real_mds d34,real_mds d35,real_mds d45);
    void   ShapeSpace5Normal(real_mds d12,real_mds d13,real_mds d14,real_mds d15,real_mds d23,real_mds d24,real_mds d25,real_mds d34,real_mds d35,real_mds d45, real_mds *normal);
}

#endif // mds_cmenger_h
