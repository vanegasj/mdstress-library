/*=========================================================================

  Module    : MDStress
  File      : mds_cmenger.h
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

#ifndef __caleymenger_h
#define __caleymenger_h

#include "mds_defines.h"
#include <math.h>

namespace mds
{
    double CaleyMenger5Der(double d12,double d13,double d14,double d15,double d23,double d24,double d25,double d34,double d35,double d45);
    void   ShapeSpace5Normal(double d12,double d13,double d14,double d15,double d23,double d24,double d25,double d34,double d35,double d45, double *normal);
}

#endif