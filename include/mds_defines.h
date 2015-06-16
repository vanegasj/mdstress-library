/*=========================================================================

  Module    : MDStress
  File      : mds_defines.cpp
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

    #define mds_sl      0
    #define mds_all     1
    #define mds_vdw     2
    #define mds_cou     3
    #define mds_ang     4
    #define mds_bnd     5
    #define mds_dip     6
    #define mds_dii     7
    #define mds_drb     8
    #define mds_lin     9
    #define mds_set     10
    #define mds_sha     12
    #define mds_kin     13
    #define mds_nr      14
    #define mds_cmp     15

    #define mds_nrow3    9
    #define mds_ncol3    3

    #define mds_nrow4    12
    #define mds_ncol4    6

    #define mds_nrow5    15
    #define mds_ncol5    10
    
    #define mds_eps      1.0e-10

    #define mds_fileext "mds"
    
    
    typedef int     iarray[3];
    typedef double  darray[3];
    typedef double  dmatrix[3][3];
    typedef darray* darraylist;
    
    class  StressGrid;
    class  StressGridPython;
    class  Lapack;
}