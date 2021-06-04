/*=========================================================================

  Module    : MDStress
  File      : mds_defines.cpp
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

#ifndef mds_defines_h
#define mds_defines_h

namespace mds
{   
    #define mds_max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
    #define mds_min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })
     
    
    #define mds_ndim    3
    #define mds_maxpart 200
    #define mds_nlvl    10 //This is very big!
    
    #define mds_units   16.6054

    #define mds_spat    0
    #define mds_atom    1


    #define mds_ccfd    0
    #define mds_ncfd    1
    #define mds_gld     2

    // stress contributions
    #define mds_none    0x0000
    #define mds_sl      0x0001
    #define mds_vdw     0x0002
    #define mds_cou     0x0004
    #define mds_ang     0x0008
    #define mds_bnd     0x0010
    #define mds_dip     0x0020
    #define mds_dii     0x0040
    #define mds_drb     0x0080
    #define mds_lin     0x0100
    #define mds_set     0x0200
    #define mds_sha     0x0400
    #define mds_kin     0x0800
    #define mds_nr      0x1000
    #define mds_cmp     0x2000
    #define mds_dio     0x4000
    #define mds_all     0x7FFF
    
    // gridc type
    #define mds_gridc_off  0x0000
    #define mds_gridc_near 0x0001
    #define mds_gridc_far  0x0002
    #define mds_gridc_full 0x0003

    #define mds_nrow3    9
    #define mds_ncol3    3

    #define mds_nrow4    12
    #define mds_ncol4    6

    #define mds_nrow5    15
    #define mds_ncol5    10
    
    #define mds_eps      1.0e-10

    #define mds_fileext "mds"

    #define mds_griddim_xxx 0
    #define mds_griddim_yyy 1
    #define mds_griddim_zzz 2
    #define mds_griddim_xyz 3

    typedef bool    barray[4];
    typedef int     iarray[3];
    typedef double  darray[3];
    typedef double  dmatrix[3][3];
    typedef double  dmatrix6[6][6];

    class  StressGrid;
    class  StressGridPython;
    class  Lapack;
}
#endif // mds_defines_h
