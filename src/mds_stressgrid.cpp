/*=========================================================================

  Module    : MDStress
  File      : mds_stressgrid.cpp
  Authors   : A. Torres-Sanchez and J. M. Vanegas
  Modified  : B. Himberg and A. Lewis
  Purpose   : Compute the local stress from MD trajectories
  Date      : Oct-15-2021
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
//KBfac is equal to KB*10E22 to convert V/(KB*T) from J/m^3 to bar^-1
#define KBfac 1.38064852E-1

#include "mds_stressgrid.h"
#include "mds_barrier.h"
#include "mds_error.h"

// disable warning for eigen
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <Eigen/Dense>
#pragma GCC diagnostic pop

//#define CUSTRESS_ENABLE
//#ifdef CUSTRESS_ENABLE
//#include "mds_custress.h"
//#endif//CUSTRESS_ENABLE

using namespace mds;

/**
 * STATIC INLINE FUNCTIONS
 */
static inline void distribute_observables_3d(
        const state_t & state,
        const array3_mds & xi,
        const array3_mds & xj,
        const array3_mds & diff,
        const matrix3_mds * stress,
        const matrix6_mds * elast,
        matrix3_mds * stress_grid,
        matrix6_mds * elast_grid)
{
    //------------------------------------------------------------------------------------
    // calculate the grid coordinates (no pbc) for the extreme points
    iarray i1 = {
        int(state.gridCells[0] * xi[0] * state.invbox[0][0] - (xi[0] < 0.0) ),
        int(state.gridCells[1] * xi[1] * state.invbox[1][1] - (xi[1] < 0.0) ),
        int(state.gridCells[2] * xi[2] * state.invbox[2][2] - (xi[2] < 0.0) )
    };

    const iarray i2 = {
        int(state.gridCells[0] * xj[0] * state.invbox[0][0] - (xj[0] < 0.0) ),
        int(state.gridCells[1] * xj[1] * state.invbox[1][1] - (xj[1] < 0.0) ),
        int(state.gridCells[2] * xj[2] * state.invbox[2][2] - (xj[2] < 0.0) ),
    };
    const array3_mds t_c1 = {
        xi[0] / (xi[0]-xj[0]),
        xi[1] / (xi[1]-xj[1]),
        xi[2] / (xi[2]-xj[2])
    };
    const array3_mds t_c2 = {
        state.gridsp[0] / (xi[0]-xj[0]),
        state.gridsp[1] / (xi[1]-xj[1]),
        state.gridsp[2] / (xi[2]-xj[2])
    };

    const iarray c = {
        int((i2[0]>i1[0])-(i1[0]>i2[0]) ),
        int((i2[1]>i1[1])-(i1[1]>i2[1]) ),
        int((i2[2]>i1[2])-(i1[2]>i2[2]) )
    };

    iarray in = {
        int(i1[0]+(c[0]+1)/2 ),
        int(i1[1]+(c[1]+1)/2 ),
        int(i1[2]+(c[2]+1)/2 )
    };

    array3_mds d_cgrid = {
        xi[0]-(i1[0]+0.5)*state.gridsp[0],
        xi[1]-(i1[1]+0.5)*state.gridsp[1],
        xi[2]-(i1[2]+0.5)*state.gridsp[2]
    };

    // calculate parametric time in each dimension, and related constants
    array3_mds t = {
        (c[0] == 0) ? realval_mds(1.1) : t_c1[0]-in[0]*t_c2[0],
        (c[1] == 0) ? realval_mds(1.1) : t_c1[1]-in[1]*t_c2[1],
        (c[2] == 0) ? realval_mds(1.1) : t_c1[2]-in[2]*t_c2[2]
    };

    //------------------------------------------------------------------------------------
    // Distribute the stress

    // now the position/spatial constants
    const real_mds C = realval_mds(0.125)*state.invgridsp*state.invgridsp;
    const real_mds axy = diff[0]*diff[1];
    const real_mds axz = diff[0]*diff[2];
    const real_mds ayz = diff[1]*diff[2];
    const real_mds axyz = diff[0]*ayz; 
    
    // track previous time of crossing
    real_mds oldt = 0.0; 
    
    const int iterations = c[0]*(i2[0]-i1[0]) + c[1]*(i2[1]-i1[1]) + c[2]*(i2[2]-i1[2]);
    for (int count = 0; count <= iterations; ++count) {
        // figure out index
        const int cmp0x = ((t[0]<t[1]+mds_eps) + (t[0]<t[2]+mds_eps))/2;
        const int cmp1x = ((t[1]<t[0]+mds_eps) + (t[1]<t[2]+mds_eps))/2;
        const int cmp2x = ((t[2]<t[0]+mds_eps) + (t[2]<t[1]+mds_eps))/2;
        const int iX = (1-cmp0x)*(cmp1x+2*(1-cmp1x)*cmp2x);
        const real_mds newt = (iterations == count) ? 1.0 : t[iX];

        // work out the parametric time constants
        const real_mds t12 = oldt*oldt;
        const real_mds t22 = newt*newt;
        const real_mds dt1 = newt - oldt;
        const real_mds dt2 = t22 - t12;
        const real_mds dt3 = realval_mds(4.0)*(t22*newt - t12*oldt)/realval_mds(3.0);
        const real_mds dt4 = t22*t22 - t12*t12;

        // additional constants
        const real_mds bxy = d_cgrid[0]*d_cgrid[1];
        const real_mds bxz = d_cgrid[0]*d_cgrid[2];
        const real_mds byz = d_cgrid[1]*d_cgrid[2];
        const real_mds bxyz = d_cgrid[0]*byz;
    
        const int iip1 = ((i1[0] + 1 + state.gridCells[0]) % state.gridCells[0])*state.gridCells[1]*state.gridCells[2];
        const int jjp1 = ((i1[1] + 1 + state.gridCells[1]) % state.gridCells[1])*state.gridCells[2];
        const int kkp1 = ((i1[2] + 1 + state.gridCells[2]) % state.gridCells[2]);
        const int iim1 = ((i1[0] + state.gridCells[0]) % state.gridCells[0])*state.gridCells[1]*state.gridCells[2];
        const int jjm1 = ((i1[1] + state.gridCells[1]) % state.gridCells[1])*state.gridCells[2];
        const int kkm1 = ((i1[2] + state.gridCells[2]) % state.gridCells[2]);

        // the composite constants in terms of i, j, k
        const real_mds D[8] = {
            realval_mds(8.0)*bxyz*dt1 + realval_mds(4.0)*(diff[0]*byz+diff[1]*bxz+diff[2]*bxy)*dt2
                + realval_mds(2.0)*(d_cgrid[0]*ayz+d_cgrid[1]*axz+d_cgrid[2]*axy)*dt3 + realval_mds(2.0)*axyz*dt4,
            state.gridsp[0]*(realval_mds(4.0)*byz*dt1 + realval_mds(2.0)*(diff[1]*d_cgrid[2]+diff[2]*d_cgrid[1])*dt2 + ayz*dt3),
            state.gridsp[1]*(realval_mds(4.0)*bxz*dt1 + realval_mds(2.0)*(diff[0]*d_cgrid[2]+diff[2]*d_cgrid[0])*dt2 + axz*dt3),
            state.gridsp[2]*(realval_mds(4.0)*bxy*dt1 + realval_mds(2.0)*(diff[0]*d_cgrid[1]+diff[1]*d_cgrid[0])*dt2 + axy*dt3),
            state.gridsp[3]*(realval_mds(2.0)*d_cgrid[2]*dt1+diff[2]*dt2),
            state.gridsp[4]*(realval_mds(2.0)*d_cgrid[1]*dt1+diff[1]*dt2),
            state.gridsp[5]*(realval_mds(2.0)*d_cgrid[0]*dt1+diff[0]*dt2),
            state.gridsp[6]*dt1,
        };

        const real_mds sf1 = C*( D[0] + D[1] + D[2] + D[3] + D[4] + D[5] + D[6] + D[7]);
        const real_mds sf2 = C*(-D[0] - D[1] - D[2] + D[3] - D[4] + D[5] + D[6] + D[7]);
        const real_mds sf3 = C*(-D[0] - D[1] + D[2] - D[3] + D[4] - D[5] + D[6] + D[7]);
        const real_mds sf4 = C*( D[0] + D[1] - D[2] - D[3] - D[4] - D[5] + D[6] + D[7]);
        const real_mds sf5 = C*(-D[0] + D[1] - D[2] - D[3] + D[4] + D[5] - D[6] + D[7]);
        const real_mds sf6 = C*( D[0] - D[1] + D[2] - D[3] - D[4] + D[5] - D[6] + D[7]);
        const real_mds sf7 = C*( D[0] - D[1] - D[2] + D[3] + D[4] - D[5] - D[6] + D[7]);
        const real_mds sf8 = C*(-D[0] + D[1] + D[2] + D[3] - D[4] - D[5] - D[6] + D[7]);

        const int ind1 = iip1 + jjp1 + kkp1;
        const int ind2 = iip1 + jjp1 + kkm1;
        const int ind3 = iip1 + jjm1 + kkp1;
        const int ind4 = iip1 + jjm1 + kkm1;
        const int ind5 = iim1 + jjp1 + kkp1;
        const int ind6 = iim1 + jjp1 + kkm1;
        const int ind7 = iim1 + jjm1 + kkp1;
        const int ind8 = iim1 + jjm1 + kkm1;

        // perform the sums into the grid
        if (nullptr != stress_grid) {
            scalesummatrix3(sf1, *stress, stress_grid[ind1]);
            scalesummatrix3(sf2, *stress, stress_grid[ind2]);
            scalesummatrix3(sf3, *stress, stress_grid[ind3]);
            scalesummatrix3(sf4, *stress, stress_grid[ind4]);
            scalesummatrix3(sf5, *stress, stress_grid[ind5]);
            scalesummatrix3(sf6, *stress, stress_grid[ind6]);
            scalesummatrix3(sf7, *stress, stress_grid[ind7]);
            scalesummatrix3(sf8, *stress, stress_grid[ind8]);
        }
        if (nullptr != elast_grid) {
            scalesummatrix6(sf1, *elast, elast_grid[ind1]);
            scalesummatrix6(sf2, *elast, elast_grid[ind2]);
            scalesummatrix6(sf3, *elast, elast_grid[ind3]);
            scalesummatrix6(sf4, *elast, elast_grid[ind4]);
            scalesummatrix6(sf5, *elast, elast_grid[ind5]);
            scalesummatrix6(sf6, *elast, elast_grid[ind6]);
            scalesummatrix6(sf7, *elast, elast_grid[ind7]);
            scalesummatrix6(sf8, *elast, elast_grid[ind8]);
        }

        d_cgrid[iX] -= c[iX] * state.gridsp[iX];
        oldt = t[iX];
        
        i1[iX] += c[iX];
        in[iX] += c[iX];

        // Next cross point:
        t[iX] = t_c1[iX]-in[iX]*t_c2[iX];
    }
}

static inline void distribute_observables_1d(
        const state_t & state,
        const array3_mds & xi,
        const array3_mds & xj,
        const array3_mds & diff,
        const matrix3_mds * stress,
        const matrix6_mds * elast,
        matrix3_mds * stress_grid,
        matrix6_mds * elast_grid)
{
    // calculate the grid coordinates (no pbc) for the extreme points
    int i1 = state.gridCells[state.gridDims] * xi[state.gridDims] * state.invbox[state.gridDims][state.gridDims] - (xi[state.gridDims] < 0.0);
    const int i2 = state.gridCells[state.gridDims] * xj[state.gridDims] * state.invbox[state.gridDims][state.gridDims] - (xj[state.gridDims] < 0.0);
    const int c = (i2>i1)-(i1>i2);
    const real_mds t_c1 = xi[state.gridDims] / (xi[state.gridDims]-xj[state.gridDims]);
    const real_mds t_c2 = state.gridsp[state.gridDims] / (xi[state.gridDims]-xj[state.gridDims]);
    const real_mds C = realval_mds(0.5)*state.invgridsp*state.invgridsp;
    real_mds d_cgrid = xi[state.gridDims]-(i1+realval_mds(0.5))*state.gridsp[state.gridDims];
    int in = i1+(c+1)/2;

    // track previous time of crossing and check that sum is complete (?)
    real_mds oldt = 0.0; 

    // fix the number of iterations
    const int iterations = c*(i2-i1);
    for (int count = 0; count <= iterations; ++count) {
        // there is always iterations+1, where the last iteration deals
        // with any residual
        real_mds newt = (iterations == count) ? realval_mds(1.0) : t_c1-in*t_c2;

        // work out the parametric time constants
        const real_mds t12 = oldt*oldt;
        const real_mds t22 = newt*newt;
        const real_mds dt1 = newt - oldt;
        const real_mds dt2 = t22 - t12;

        const int p1 = ((i1 + 1 + state.gridCells[state.gridDims]) % state.gridCells[state.gridDims]);
        const int m1 = ((i1 + state.gridCells[state.gridDims]) % state.gridCells[state.gridDims]);

        // the composite constants in terms of i, j, k
        const real_mds D1 = state.gridsp[5-state.gridDims]*(realval_mds(2.0)*d_cgrid*dt1+diff[state.gridDims]*dt2);
        const real_mds D2 = state.gridsp[6]*dt1;

        const real_mds sf1 = C*( D1 + D2);
        const real_mds sf2 = C*(-D1 + D2);

        if (nullptr != stress_grid) {
            scalesummatrix3(sf1, *stress, stress_grid[p1]);
            scalesummatrix3(sf2, *stress, stress_grid[m1]);
        }
        if (nullptr != elast_grid) {
            scalesummatrix6(sf1, *elast, elast_grid[p1]);
            scalesummatrix6(sf2, *elast, elast_grid[m1]);
        }
        
        d_cgrid -= c * state.gridsp[state.gridDims];
        oldt = newt;
        
        i1 += c;
        in += c;
    }
}

static inline void distribute_n2(
        const state_t & state,
        const array3_mds & xi,
        const array3_mds & xj,
        const array3_mds * F,
        matrix3_mds * stress_grid,
        const array3_mds * xk,
        const array3_mds * xl,
        const real_ext * phi,
        const real_ext * kappa,
        matrix6_mds * elast_grid)
{
    // Calculate the stress tensor
    array3_mds xij;
    diffarray3( xj, xi, xij, state.box, state.periodic);

#ifdef CUSTRESS_ENABLE
    if (settings.cuda && custress_distribute_pair_interaction(xi,xj,F,batch_id))
        return;
#endif

    // stress is relatively inexpensive so just do it here to reduce cases later
    matrix3_mds stress;
    if (nullptr != F) {
        stress[0][0] = F[0][0]*xij[0];
        stress[0][1] = F[0][0]*xij[1];
        stress[0][2] = F[0][0]*xij[2];
        stress[1][0] = F[0][1]*xij[0];
        stress[1][1] = F[0][1]*xij[1];
        stress[1][2] = F[0][1]*xij[2];
        stress[2][0] = F[0][2]*xij[0];
        stress[2][1] = F[0][2]*xij[1];
        stress[2][2] = F[0][2]*xij[2];
    } else {
        stress_grid = nullptr;
    }
    
    matrix6_mds elast = {0};
    if (nullptr != phi && nullptr != phi &&
            (realval_ext(0.0) != phi[0] || realval_ext(0.0) != kappa[1])) {
        // Construct the stiffness matrix in Voigt notation
        // 0 = xx; 1 = yy; 2 = zz; 3 = yz or zy; 4 = xz or zx; 5 = xy or yx
        // All indices                         Voigt indices           Stress indices
        // ( xxxx xxyy xxzz xxyz xxxz xxxy ) = ( 00 01 02 03 04 05 ) = [ 0000 0011 0022 0012 0002 0001 ]
        // (      yyyy yyzz yyyz yyxz yyxy ) = (    11 12 13 14 15 ) = [      1111 1122 1112 1102 1101 ]
        // (           zzzz zzyz zzxz zzxy ) = (       22 23 24 25 ) = [           2222 2212 2202 2201 ]
        // (                yzyz yzxz yzxy ) = (          33 34 35 ) = [                1212 1202 1201 ]
        // (                     xzxz xzxy ) = (             44 45 ) = [                     0202 0201 ]
        // (                          xyxy ) = (                55 ) = [                          0101 ]

        array3_mds xkl;
        diffarray3( *xl, *xk, xkl, state.box, state.periodic);

        const real_mds xij_norm = normarray3(xij);
        const real_mds xkl_norm = normarray3(xkl);

        const real_mds rinv = realval_mds(1.0)/xij_norm;
        const real_mds rinv2 = real_mds(kappa[0])*rinv/xkl_norm;
        const real_mds rinv3 = real_mds(phi[0])*rinv*rinv*rinv;

        const real_mds xij00 = xij[0]*xij[0];
        const real_mds xij11 = xij[1]*xij[1];
        const real_mds xij22 = xij[2]*xij[2];
        const real_mds xij01 = xij[0]*xij[1];
        const real_mds xij02 = xij[0]*xij[2];
        const real_mds xij12 = xij[1]*xij[2];
        
        const real_mds xij00r = xij00*rinv3;
        const real_mds xij11r = xij11*rinv3;
        const real_mds xij22r = xij22*rinv3;
        const real_mds xij01r = xij01*rinv3;
        const real_mds xij02r = xij02*rinv3;
        const real_mds xij12r = xij12*rinv3;

        const real_mds xkl00r = xkl[0]*xkl[0]*rinv2;
        const real_mds xkl11r = xkl[1]*xkl[1]*rinv2;
        const real_mds xkl22r = xkl[2]*xkl[2]*rinv2;
        const real_mds xkl01r = xkl[0]*xkl[1]*rinv2;
        const real_mds xkl02r = xkl[0]*xkl[2]*rinv2;
        const real_mds xkl12r = xkl[1]*xkl[2]*rinv2;

        elast[0][0] = xij00*xkl00r - xij00*xij00r;
        elast[0][1] = xij00*xkl11r - xij00*xij11r;
        elast[0][2] = xij00*xkl22r - xij00*xij22r;
        elast[0][3] = xij00*xkl12r - xij00*xij12r;
        elast[0][4] = xij00*xkl02r - xij00*xij02r;
        elast[0][5] = xij00*xkl01r - xij00*xij01r;
        elast[1][1] = xij11*xkl11r - xij11*xij11r;
        elast[1][2] = xij11*xkl22r - xij11*xij22r;
        elast[1][3] = xij11*xkl12r - xij11*xij12r;
        elast[1][4] = xij11*xkl02r - xij11*xij02r;
        elast[1][5] = xij11*xkl01r - xij11*xij01r;
        elast[2][2] = xij22*xkl22r - xij22*xij22r;
        elast[2][3] = xij22*xkl12r - xij22*xij12r;
        elast[2][4] = xij22*xkl02r - xij22*xij02r;
        elast[2][5] = xij22*xkl01r - xij22*xij01r;
        elast[3][3] = xij12*xkl12r - xij12*xij12r;
        elast[3][4] = xij12*xkl02r - xij12*xij02r;
        elast[3][5] = xij12*xkl01r - xij12*xij01r;
        elast[4][4] = xij02*xkl02r - xij02*xij02r;
        elast[4][5] = xij02*xkl01r - xij02*xij01r;
        elast[5][5] = xij01*xkl01r - xij01*xij01r;
    } else {
        elast_grid = nullptr;
    }
    
    if (nullptr != stress_grid || nullptr != elast_grid) {
        if (state.gridDims == mds_griddim_xyz) {
            distribute_observables_3d(state, xi, xj, xij, &stress, &elast, stress_grid, elast_grid);
        } else {
            distribute_observables_1d(state, xi, xj, xij, &stress, &elast, stress_grid, elast_grid);
        }
    }
}

// Decompose 3-body potentials
static void decompose_n3(
        const state_t & state,
        const array3_ext *R,
        const array3_ext *F,
        matrix3_mds * stress_grid,
        const real_ext * phi,
        const real_ext * kappa,
        matrix6_mds * elast_grid)
{
    const array3_mds Ra = {(real_mds)R[0][0], (real_mds)R[0][1], (real_mds)R[0][2]};
    const array3_mds Rb = {(real_mds)R[1][0], (real_mds)R[1][1], (real_mds)R[1][2]};
    const array3_mds Rc = {(real_mds)R[2][0], (real_mds)R[2][1], (real_mds)R[2][2]};
    const array3_mds Fa = {(real_mds)F[0][0], (real_mds)F[0][1], (real_mds)F[0][2]};
    const array3_mds Fb = {(real_mds)F[1][0], (real_mds)F[1][1], (real_mds)F[1][2]};
    const array3_mds Fc = {(real_mds)F[2][0], (real_mds)F[2][1], (real_mds)F[2][2]};

    // Vectors between particles
    array3_mds AB, AC, BC;
    diffarray3(Rb, Ra, AB, state.box, state.periodic);
    diffarray3(Rc, Ra, AC, state.box, state.periodic);
    diffarray3(Rc, Rb, BC, state.box, state.periodic);

    /// We want to solve M*x = b
    const Eigen::Matrix<double,9,3> M_eig{
        { (double)AB[0], (double)AC[0], 0.0          },
        { (double)AB[1], (double)AC[1], 0.0          },
        { (double)AB[2], (double)AC[2], 0.0          },
        {-(double)AB[0], 0.0          , (double)BC[0]},
        {-(double)AB[1], 0.0          , (double)BC[1]},
        {-(double)AB[2], 0.0          , (double)BC[2]},
        { 0.0          ,-(double)AC[0],-(double)BC[0]},
        { 0.0          ,-(double)AC[1],-(double)BC[1]},
        { 0.0          ,-(double)AC[2],-(double)BC[2]}
    };

    const Eigen::Matrix<double,9,1> b_eig{
        {(double)Fa[0]},
        {(double)Fa[1]},
        {(double)Fa[2]},
        {(double)Fb[0]},
        {(double)Fb[1]},
        {(double)Fb[2]},
        {(double)Fc[0]},
        {(double)Fc[1]},
        {(double)Fc[2]}
    };

    Eigen::JacobiSVD<Eigen::Matrix<double,9,3>> svd;
    svd.compute(M_eig, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::Matrix<double,3,1> x = svd.solve(b_eig);

    const real_mds lab = (real_mds)x(0,0);
    const real_mds lac = (real_mds)x(1,0);
    const real_mds lbc = (real_mds)x(2,0);

    const array3_mds Fij1 = {lab * AB[0], lab * AB[1], lab * AB[2]};
    const array3_mds Fij2 = {lac * AC[0], lac * AC[1], lac * AC[2]};
    const array3_mds Fij3 = {lbc * BC[0], lbc * BC[1], lbc * BC[2]};

    if (nullptr == phi && nullptr == kappa && nullptr != elast_grid) {
        // settle specialization
        const real_mds normAB = normarray3(AB);
        const real_mds normAC = normarray3(AC);
        const real_mds normBC = normarray3(BC);

        const real_ext phi_kappa1[2] = {real_ext(lab*normAB), realval_ext(0.0)};
        const real_ext phi_kappa2[2] = {real_ext(lac*normAC), realval_ext(0.0)};
        const real_ext phi_kappa3[2] = {real_ext(lbc*normBC), realval_ext(0.0)};
        distribute_n2(state, Ra, Rb, &Fij1, stress_grid, &Ra, &Rb, phi_kappa1, phi_kappa1+1, elast_grid);
        distribute_n2(state, Ra, Rc, &Fij2, stress_grid, &Ra, &Rc, phi_kappa2, phi_kappa2+1, elast_grid);
        distribute_n2(state, Rb, Rc, &Fij3, stress_grid, &Rb, &Rc, phi_kappa3, phi_kappa3+1, elast_grid);
    } else if (nullptr != phi && nullptr != kappa && nullptr != elast_grid) {
        // calculate the diagonal terms
        distribute_n2(state, Ra, Rb, &Fij1, stress_grid, &Ra, &Rb, phi+0, kappa+0, elast_grid);
        distribute_n2(state, Ra, Rc, &Fij2, stress_grid, &Ra, &Rc, phi+1, kappa+4, elast_grid);
        distribute_n2(state, Rb, Rc, &Fij3, stress_grid, &Rb, &Rc, phi+2, kappa+8, elast_grid);

        // and the off diagonal
        const real_ext phi [] = {realval_mds(0.0)};
        distribute_n2(state, Ra, Rb, nullptr, nullptr, &Ra, &Rc, phi, kappa+1, elast_grid);
        distribute_n2(state, Ra, Rb, nullptr, nullptr, &Rb, &Rc, phi, kappa+2, elast_grid);
        distribute_n2(state, Ra, Rc, nullptr, nullptr, &Ra, &Rb, phi, kappa+3, elast_grid);
        distribute_n2(state, Ra, Rc, nullptr, nullptr, &Rb, &Rc, phi, kappa+5, elast_grid);
        distribute_n2(state, Rb, Rc, nullptr, nullptr, &Ra, &Rb, phi, kappa+6, elast_grid);
        distribute_n2(state, Rb, Rc, nullptr, nullptr, &Ra, &Rc, phi, kappa+7, elast_grid);
    } else {
        // general stress only case
        distribute_n2(state, Ra, Rb, &Fij1, stress_grid, nullptr, nullptr, nullptr, nullptr, nullptr);
        distribute_n2(state, Ra, Rc, &Fij2, stress_grid, nullptr, nullptr, nullptr, nullptr, nullptr);
        distribute_n2(state, Rb, Rc, &Fij3, stress_grid, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
}

static void decompose_n4(
        const state_t & state,
        const array3_ext *R,
        const array3_ext *F,
        matrix3_mds * grid)
{
    const array3_mds Ra = {(real_mds)R[0][0], (real_mds)R[0][1], (real_mds)R[0][2]};
    const array3_mds Rb = {(real_mds)R[1][0], (real_mds)R[1][1], (real_mds)R[1][2]};
    const array3_mds Fa = {(real_mds)F[0][0], (real_mds)F[0][1], (real_mds)F[0][2]};
    const array3_mds Rc = {(real_mds)R[2][0], (real_mds)R[2][1], (real_mds)R[2][2]};
    const array3_mds Fb = {(real_mds)F[1][0], (real_mds)F[1][1], (real_mds)F[1][2]};
    const array3_mds Fc = {(real_mds)F[2][0], (real_mds)F[2][1], (real_mds)F[2][2]};
    const array3_mds Rd = {(real_mds)R[3][0], (real_mds)R[3][1], (real_mds)R[3][2]};
    const array3_mds Fd = {(real_mds)F[3][0], (real_mds)F[3][1], (real_mds)F[3][2]};

    array3_mds AB, AC, AD, BC, BD, CD;
    diffarray3(Rb, Ra, AB, state.box, state.periodic);
    diffarray3(Rc, Ra, AC, state.box, state.periodic);
    diffarray3(Rd, Ra, AD, state.box, state.periodic);
    diffarray3(Rc, Rb, BC, state.box, state.periodic);
    diffarray3(Rd, Rb, BD, state.box, state.periodic);
    diffarray3(Rd, Rc, CD, state.box, state.periodic);

    const Eigen::Matrix<double, 12, 6> M_eig{
        { (double)AB[0],  (double)AC[0],  (double)AD[0],  0.0          ,  0.0          ,  0.0          }, 
        { (double)AB[1],  (double)AC[1],  (double)AD[1],  0.0          ,  0.0          ,  0.0          }, 
        { (double)AB[2],  (double)AC[2],  (double)AD[2],  0.0          ,  0.0          ,  0.0          }, 
        {-(double)AB[0],  0.0          ,  0.0          ,  (double)BC[0],  (double)BD[0],  0.0          }, 
        {-(double)AB[1],  0.0          ,  0.0          ,  (double)BC[1],  (double)BD[1],  0.0          }, 
        {-(double)AB[2],  0.0          ,  0.0          ,  (double)BC[2],  (double)BD[2],  0.0          }, 
        { 0.0          , -(double)AC[0],  0.0          , -(double)BC[0],  0.0          ,  (double)CD[0]}, 
        { 0.0          , -(double)AC[1],  0.0          , -(double)BC[1],  0.0          ,  (double)CD[1]}, 
        { 0.0          , -(double)AC[2],  0.0          , -(double)BC[2],  0.0          ,  (double)CD[2]}, 
        { 0.0          ,  0.0          , -(double)AD[0],  0.0          , -(double)BD[0], -(double)CD[0]}, 
        { 0.0          ,  0.0          , -(double)AD[1],  0.0          , -(double)BD[1], -(double)CD[1]}, 
        { 0.0          ,  0.0          , -(double)AD[2],  0.0          , -(double)BD[2], -(double)CD[2]},
    };

    const Eigen::Matrix<double, 12, 1> b_eig{
        {(double)Fa[0]},
        {(double)Fa[1]},
        {(double)Fa[2]},
        {(double)Fb[0]},
        {(double)Fb[1]},
        {(double)Fb[2]},
        {(double)Fc[0]},
        {(double)Fc[1]},
        {(double)Fc[2]},
        {(double)Fd[0]},
        {(double)Fd[1]},
        {(double)Fd[2]},
    };
    
    Eigen::JacobiSVD<Eigen::Matrix<double,12,6>> svd;
    svd.compute(M_eig, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::Matrix<double,6,1> x = svd.solve(b_eig);

    // Sum the 6 contributions to the stress
    const real_mds lab = (real_mds)x(0,0);
    const real_mds lac = (real_mds)x(1,0);
    const real_mds lad = (real_mds)x(2,0);
    const real_mds lbc = (real_mds)x(3,0);
    const real_mds lbd = (real_mds)x(4,0);
    const real_mds lcd = (real_mds)x(5,0);

    const array3_mds Fij1 = {lab * AB[0], lab * AB[1], lab * AB[2]};
    distribute_n2(state, Ra, Rb, &Fij1, grid, nullptr, nullptr, nullptr, nullptr, nullptr);

    const array3_mds Fij2 = {lac * AC[0], lac * AC[1], lac * AC[2]};
    distribute_n2(state, Ra, Rc, &Fij2, grid, nullptr, nullptr, nullptr, nullptr, nullptr);

    const array3_mds Fij3 = {lad * AD[0], lad * AD[1], lad * AD[2]};
    distribute_n2(state, Ra, Rd, &Fij3, grid, nullptr, nullptr, nullptr, nullptr, nullptr);

    const array3_mds Fij4 = {lbc * BC[0], lbc * BC[1], lbc * BC[2]};
    distribute_n2(state, Rb, Rc, &Fij4, grid, nullptr, nullptr, nullptr, nullptr, nullptr);

    const array3_mds Fij5 = {lbd * BD[0], lbd * BD[1], lbd * BD[2]};
    distribute_n2(state, Rb, Rd, &Fij5, grid, nullptr, nullptr, nullptr, nullptr, nullptr);

    const array3_mds Fij6 = {lcd * CD[0], lcd * CD[1], lcd * CD[2]};
    distribute_n2(state, Rc, Rd, &Fij6, grid, nullptr, nullptr, nullptr, nullptr, nullptr);
}

// Decompose 5-body potentials (CMAP)
static void decompose_n5(
        const state_t & state,
        const array3_ext *R,
        const array3_ext *F,
        matrix3_mds * grid,
        int fdecomp)
{
    const array3_mds Ra = {(real_mds)R[0][0], (real_mds)R[0][1], (real_mds)R[0][2]};
    const array3_mds Rb = {(real_mds)R[1][0], (real_mds)R[1][1], (real_mds)R[1][2]};
    const array3_mds Rc = {(real_mds)R[2][0], (real_mds)R[2][1], (real_mds)R[2][2]};
    const array3_mds Rd = {(real_mds)R[3][0], (real_mds)R[3][1], (real_mds)R[3][2]};
    const array3_mds Re = {(real_mds)R[4][0], (real_mds)R[4][1], (real_mds)R[4][2]};
    const array3_mds Fa = {(real_mds)F[0][0], (real_mds)F[0][1], (real_mds)F[0][2]};
    const array3_mds Fb = {(real_mds)F[1][0], (real_mds)F[1][1], (real_mds)F[1][2]};
    const array3_mds Fc = {(real_mds)F[2][0], (real_mds)F[2][1], (real_mds)F[2][2]};
    const array3_mds Fd = {(real_mds)F[3][0], (real_mds)F[3][1], (real_mds)F[3][2]};
    const array3_mds Fe = {(real_mds)F[4][0], (real_mds)F[4][1], (real_mds)F[4][2]};

    //************************************************************************************
    // Matrix of the system (15 equations x 10 unknowns)
    // Vector, we want to solve M*x = b
    // Scalar product of the Normal and the initial CFD
    array3_mds AB, AC, AD, AE, BC, BD, BE, CD, CE, DE;
    diffarray3(Rb, Ra, AB, state.box, state.periodic);
    diffarray3(Rc, Ra, AC, state.box, state.periodic);
    diffarray3(Rd, Ra, AD, state.box, state.periodic);
    diffarray3(Re, Ra, AE, state.box, state.periodic);
    diffarray3(Rc, Rb, BC, state.box, state.periodic);
    diffarray3(Rd, Rb, BD, state.box, state.periodic);
    diffarray3(Re, Rb, BE, state.box, state.periodic);
    diffarray3(Rd, Rc, CD, state.box, state.periodic);
    diffarray3(Re, Rc, CE, state.box, state.periodic);
    diffarray3(Re, Rd, DE, state.box, state.periodic);

    const real_mds normAB=normarray3(AB);
    const real_mds normAC=normarray3(AC);
    const real_mds normAD=normarray3(AD);
    const real_mds normAE=normarray3(AE);
    const real_mds normBC=normarray3(BC);
    const real_mds normBD=normarray3(BD);
    const real_mds normBE=normarray3(BE);
    const real_mds normCD=normarray3(CD);
    const real_mds normCE=normarray3(CE);
    const real_mds normDE=normarray3(DE);

    for(int i = 0; i < 3; i++) {
        if(normAB > mds_eps) AB[i]/=normAB;
        if(normAC > mds_eps) AC[i]/=normAC;
        if(normAD > mds_eps) AD[i]/=normAD;
        if(normAE > mds_eps) AE[i]/=normAE;
        if(normBC > mds_eps) BC[i]/=normBC;
        if(normBD > mds_eps) BD[i]/=normBD;
        if(normBE > mds_eps) BE[i]/=normBE;
        if(normCD > mds_eps) CD[i]/=normCD;
        if(normCE > mds_eps) CE[i]/=normCE;
        if(normDE > mds_eps) DE[i]/=normDE;
    }

    const Eigen::Matrix<double,15,10> M_eig{
        { (double)AB[0],  (double)AC[0],  (double)AD[0],  (double)AE[0],  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          },
        { (double)AB[1],  (double)AC[1],  (double)AD[1],  (double)AE[1],  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          },
        { (double)AB[2],  (double)AC[2],  (double)AD[2],  (double)AE[2],  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          },
        {-(double)AB[0],  0.0          ,  0.0          ,  0.0          ,  (double)BC[0],  (double)BD[0],  (double)BE[0],  0.0          ,  0.0          ,  0.0          },
        {-(double)AB[1],  0.0          ,  0.0          ,  0.0          ,  (double)BC[1],  (double)BD[1],  (double)BE[1],  0.0          ,  0.0          ,  0.0          },
        {-(double)AB[2],  0.0          ,  0.0          ,  0.0          ,  (double)BC[2],  (double)BD[2],  (double)BE[2],  0.0          ,  0.0          ,  0.0          },
        { 0.0          , -(double)AC[0],  0.0          ,  0.0          , -(double)BC[0],  0.0          ,  0.0          ,  (double)CD[0],  (double)CE[0],  0.0          },
        { 0.0          , -(double)AC[1],  0.0          ,  0.0          , -(double)BC[1],  0.0          ,  0.0          ,  (double)CD[1],  (double)CE[1],  0.0          },
        { 0.0          , -(double)AC[2],  0.0          ,  0.0          , -(double)BC[2],  0.0          ,  0.0          ,  (double)CD[2],  (double)CE[2],  0.0          },
        { 0.0          ,  0.0          , -(double)AD[0],  0.0          ,  0.0          , -(double)BD[0],  0.0          , -(double)CD[0],  0.0          ,  (double)DE[0]},
        { 0.0          ,  0.0          , -(double)AD[1],  0.0          ,  0.0          , -(double)BD[1],  0.0          , -(double)CD[1],  0.0          ,  (double)DE[1]},
        { 0.0          ,  0.0          , -(double)AD[2],  0.0          ,  0.0          , -(double)BD[2],  0.0          , -(double)CD[2],  0.0          ,  (double)DE[2]},
        { 0.0          ,  0.0          ,  0.0          , -(double)AE[0],  0.0          ,  0.0          , -(double)BE[0],  0.0          , -(double)CE[0], -(double)DE[0]},
        { 0.0          ,  0.0          ,  0.0          , -(double)AE[1],  0.0          ,  0.0          , -(double)BE[1],  0.0          , -(double)CE[1], -(double)DE[1]},
        { 0.0          ,  0.0          ,  0.0          , -(double)AE[2],  0.0          ,  0.0          , -(double)BE[2],  0.0          , -(double)CE[2], -(double)DE[2]}
    };

    const Eigen::Matrix<double,15,1> b_eig{
        {(double)Fa[0]},
        {(double)Fa[1]},
        {(double)Fa[2]},
        {(double)Fb[0]},
        {(double)Fb[1]},
        {(double)Fb[2]},
        {(double)Fc[0]},
        {(double)Fc[1]},
        {(double)Fc[2]},
        {(double)Fd[0]},
        {(double)Fd[1]},
        {(double)Fd[2]},
        {(double)Fe[0]},
        {(double)Fe[1]},
        {(double)Fe[2]},
    };
    
    Eigen::JacobiSVD<Eigen::Matrix<double,15,10>> svd;
    svd.compute(M_eig, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix<double,10,1> x = svd.solve(b_eig);

    // If cCFD project the least squares CFD to the shape space
    if(fdecomp == mds_ccfd) {
        // Calculate the normal to the Shape Space
        real_mds CaleyMengerNormal[15] = {0};
        ShapeSpace5Normal(normAB,normAC,normAD,normAE,normBC,normBD,normBE,normCD,normCE,normDE,CaleyMengerNormal);
        
        // Covariant derivative
        real_mds prod = realval_mds(0.0);
        for (int i = 0; i < 15; i++ ) {
            prod +=(real_mds)x(i,0)*CaleyMengerNormal[i];
        }

        for (int i = 0; i < 15; i++ ) {
            x(i,0) = (real_mds)x(i,0) - prod * CaleyMengerNormal[i];
        }
    }

    const real_mds lab = (real_mds)x(0,0);
    const real_mds lac = (real_mds)x(1,0);
    const real_mds lad = (real_mds)x(2,0);
    const real_mds lae = (real_mds)x(3,0);
    const real_mds lbc = (real_mds)x(4,0);
    const real_mds lbd = (real_mds)x(5,0);
    const real_mds lbe = (real_mds)x(6,0);
    const real_mds lcd = (real_mds)x(7,0);
    const real_mds lce = (real_mds)x(8,0);
    const real_mds lde = (real_mds)x(9,0);

    const array3_mds Fij1 = {lab * AB[0], lab * AB[1], lab * AB[2]};
    distribute_n2(state, Ra, Rb, &Fij1, grid, nullptr, nullptr, nullptr, nullptr, nullptr);

    const array3_mds Fij2 = {lac * AC[0], lac * AC[1], lac * AC[2]};
    distribute_n2(state, Ra, Rc, &Fij2, grid, nullptr, nullptr, nullptr, nullptr, nullptr);

    const array3_mds Fij3 = {lad * AD[0], lad * AD[1], lad * AD[2]};
    distribute_n2(state, Ra, Rd, &Fij3, grid, nullptr, nullptr, nullptr, nullptr, nullptr);

    const array3_mds Fij4 = {lae * AE[0], lae * AE[1], lae * AE[2]};
    distribute_n2(state, Ra, Re, &Fij4, grid, nullptr, nullptr, nullptr, nullptr, nullptr);

    const array3_mds Fij5 = {lbc * BC[0], lbc * BC[1], lbc * BC[2]};
    distribute_n2(state, Rb, Rc, &Fij5, grid, nullptr, nullptr, nullptr, nullptr, nullptr);

    const array3_mds Fij6 = {lbd * BD[0], lbd * BD[1], lbd * BD[2]};
    distribute_n2(state, Rb, Rd, &Fij6, grid, nullptr, nullptr, nullptr, nullptr, nullptr);

    const array3_mds Fij7 = {lbe * BE[0], lbe * BE[1], lbe * BE[2]};
    distribute_n2(state, Rb, Re, &Fij7, grid, nullptr, nullptr, nullptr, nullptr, nullptr);

    const array3_mds Fij8 = {lcd * CD[0], lcd * CD[1], lcd * CD[2]};
    distribute_n2(state, Rc, Rd, &Fij8, grid, nullptr, nullptr, nullptr, nullptr, nullptr);

    const array3_mds Fij9 = {lce * CE[0], lce * CE[1], lce * CE[2]};
    distribute_n2(state, Rc, Re, &Fij9, grid, nullptr, nullptr, nullptr, nullptr, nullptr);

    const array3_mds Fij10= {lde * DE[0], lde * DE[1], lde * DE[2]};
    distribute_n2(state, Rd, Re, &Fij10, grid, nullptr, nullptr, nullptr, nullptr, nullptr);
}

static void decompose_nbody(
        const state_t & state,
        int nAtoms,
        const array3_ext *R,
        const array3_ext *F,
        matrix3_mds * stress_grid)
{
    // calculate the size of the matrix
    const int m_rows = mds_ndim*nAtoms;
    const int n_cols = (nAtoms*(nAtoms-1))/2;

    // fill matrix M
    Eigen::MatrixXd M_eig = Eigen::MatrixXd::Zero(m_rows, n_cols);

    int col = 0;
    for (int ai = 0; ai < nAtoms; ++ai) {
        const double Ra[3] = {R[ai][0], R[ai][1], R[ai][2]};
        for (int bi = ai+1; bi < nAtoms; ++bi) {
            double Rb[3] = {R[bi][0], R[bi][1], R[bi][2]};

            double AB[3] = {0};
            diffarray3(Rb, Ra, AB, state.box, state.periodic);
            
            int rowPos = ai*3;
            M_eig(rowPos+0, col) = AB[0];
            M_eig(rowPos+1, col) = AB[1];
            M_eig(rowPos+2, col) = AB[2];
            
            int rowNeg = bi*3;
            M_eig(rowNeg+0, col) = -AB[0];
            M_eig(rowNeg+1, col) = -AB[1];
            M_eig(rowNeg+2, col) = -AB[2];

            col += 1;
        }
    }

    // fill in vector b
    Eigen::MatrixXd b_eig = Eigen::MatrixXd::Zero(m_rows, 1);

    for (int i = 0; i < nAtoms; ++i) {
        b_eig(3*i+0, 0) = F[i][0];
        b_eig(3*i+1, 0) = F[i][1];
        b_eig(3*i+2, 0) = F[i][2];
    }
    
    // solve for coefficients
    Eigen::JacobiSVD<Eigen::MatrixXd> svd;
    svd.compute(M_eig, Eigen::FullPivHouseholderQRPreconditioner | Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd x = Eigen::MatrixXd::Zero(n_cols,1);
    x = svd.solve(b_eig);

    // TODO: need to eliminate components in NULL space:
    // AtA x = At b
    // (QR)t(QR) x = (QR)t b
    // RtR x = RtQt b
    // R x = Qt b
    // QR x = QQt b
    // b^{hat} = QQt b   (b^{hat} here replaces b)

    // distribute the stress
    col = 0;
    for (int ai = 0; ai < nAtoms; ++ai) {
        array3_mds Ra = {R[ai][0], R[ai][1], R[ai][2]};
        for (int bi = ai+1; bi < nAtoms; ++bi) {
            array3_mds Rb = {R[bi][0], R[bi][1], R[bi][2]};
            
            array3_mds AB = {0};
            diffarray3(Rb, Ra, AB, state.box, state.periodic);

            const real_mds lab = x(col, 0);
            const array3_mds Fab = {lab*AB[0], lab*AB[1], lab*AB[2]};
            distribute_n2(state, Ra, Rb, &Fab, stress_grid, nullptr, nullptr, nullptr, nullptr, nullptr);

            col += 1;
        }
    }
}

/**
 * PUBLIC CLASS FUNCTIONS
 */
StressGrid::StressGrid() :
    settings({0}),
    state({0}),
    alloc({0}),
    m_filename(),
    m_max_threads(0),
    m_thread_map(),
    m_mutex_state()
{
    this->state.gridDims = mds_griddim_xyz;
}

StressGrid::~StressGrid()
{
    this->Clear();
}

void StressGrid::Init()
{
    std::lock_guard<std::mutex> lock(this->m_mutex_state);
    
    // Check some common settings before attempting initialization
    this->state.ierr = MDS_OK;

    // grid size must always be greater than or equal to 0
    this->state.ierr |= checkError(
            this->state.gridCells[0] < 0,
            "SetNumberOfGridCellsX(int) - int must not be less than 0");
    this->state.ierr |= checkError(
            this->state.gridCells[1] < 0,
            "SetNumberOfGridCellsY(int) - int must not be less than 0");
    this->state.ierr |= checkError(
            this->state.gridCells[2] < 0,
            "SetNumberOfGridCellsZ(int) - int must not be less than 0");

    // if grid size is 0 in all dimensions, we will use spacing to generate grid size
    // and so we check it here
    if (0 == this->state.gridCells[0] && 0 == this->state.gridCells[1] && 0 == this->state.gridCells[2]) {
        this->state.ierr |= checkError(
                this->settings.gridSpacing < mds_eps,
                "SetSpacing(float) - float must be greater than mds_eps");
        this->state.ierr |= checkError(
                iszeromatrix3(this->state.box),
                "SetBox(mat3,int) - all mat3 elements must be greater than mds_eps");
    }
    
    // a force decomposition algorithm needs to be selected
    this->state.ierr |= checkError(
            mds_ccfd != this->settings.fdecomp &&
            mds_ncfd != this->settings.fdecomp,
            "SetForceDecomposition(type) - type must be mds_{ccfd or ncfd}");

    // parameters unrelated to stress algorithm and md
    this->state.ierr |= checkError(
            0 == this->m_filename.size(),
            "SetFileName(string) - len(string) must be greater than 0");

    if (MDS_OK == this->state.ierr) {
        // Clear all allocations, keeping settings
        this->Clear();
    
        // Set grid sizes before allocation
        if(this->state.gridCells[0] == 0)
            this->state.gridCells[0] = static_cast<int>(this->state.box[0][0]/this->settings.gridSpacing);
        if(this->state.gridCells[1] == 0)
            this->state.gridCells[1] = static_cast<int>(this->state.box[1][1]/this->settings.gridSpacing);
        if(this->state.gridCells[2] == 0)
            this->state.gridCells[2] = static_cast<int>(this->state.box[2][2]/this->settings.gridSpacing);
        
        this->state.nCells = this->state.gridCells[0]*this->state.gridCells[1]*this->state.gridCells[2];

        if (this->state.nCells == this->state.gridCells[0]) {
            this->state.gridDims = mds_griddim_xxx;
        } else if (this->state.nCells == this->state.gridCells[1]) {
            this->state.gridDims = mds_griddim_yyy;
        } else if (this->state.nCells == this->state.gridCells[2]) {
            this->state.gridDims = mds_griddim_zzz;
        } else {
            this->state.gridDims = mds_griddim_xyz;
        }

        // Allocate all space necessary for grids
        this->alloc.sum_grid             = new matrix3_mds [this->state.nCells]();
        this->alloc.avg_grid             = new matrix3_mds [this->state.nCells]();
        this->alloc.avg_gridtot          = new matrix3_mds [1]();
        this->alloc.sum_grid_elcovar     = new matrix6_mds [this->state.nCells]();
        this->alloc.sum_grid_elkin       = new matrix6_mds [this->state.nCells]();
        this->alloc.sum_grid_elborn      = new matrix6_mds [this->state.nCells]();
        this->alloc.sum_grid_volcovar    = new matrix3_mds [this->state.nCells]();
        this->alloc.sum_gridtot_volcovar = new matrix3_mds [1]();
        this->alloc.current_grid_elkin   = new matrix6_mds [this->state.nCells*this->m_max_threads]();
        this->alloc.current_grid_elborn  = new matrix6_mds [this->state.nCells*this->m_max_threads]();
        this->alloc.current_grid         = new matrix3_mds [this->state.nCells*this->m_max_threads]();
        this->alloc.current_gridtot      = new matrix3_mds [1]();

        // Zero all allocated buffers and counters
        this->state.nframes = 0;
        this->state.avg_boxvol = realval_mds(0.0);
        this->state.var_boxvol = realval_mds(0.0);

#ifdef CUSTRESS_ENABLE
        if (this->settings.cuda) {
            custress_init(
                    this->m_max_threads,
                    this->state.nCells,
                    this->state.gridCells[0],
                    this->state.gridCells[1],
                    this->state.gridCells[2]);
        }
#endif//CUSTRESS_ENABLE
        printf("STRESSLIB: Spacing requested: %g    Using nx=%d ny=%d nz=%d, grid size %ld \n",
                this->settings.gridSpacing,
                this->state.gridCells[0],
                this->state.gridCells[1],
                this->state.gridCells[2],
                this->state.nCells);
        printf("STRESSLIB: The temperature value used for the elasticity calculations is T = %g K.\n",
                this->settings.temperature);

        // Let us know what precision is being used
        std::cout << "STRESSLIB: Internal Floating Points Precision - " << sizeof(real_mds) << " Bytes" << std::endl;
        std::cout << "STRESSLIB: External Floating Points Precision - " << sizeof(real_ext) << " Bytes" << std::endl;
        std::cout << "STRESSLIB: Output Floating Points Precision - " << sizeof(real_out) << " Bytes" << std::endl;

        // we have successfully initialized
        this->settings.initialized = true;
    }
}

void StressGrid::UpdateBoxSpacings ( matrix3_ext box )
{
    if (MDS_OK == this->state.ierr) {
        if (this->m_thread_map[std::this_thread::get_id()] == 0) {
            copymatrix3( box, this->state.box);
            inversematrix3( this->state.box, this->state.invbox );

            this->state.gridsp[0] = this->state.box[0][0]/static_cast<real_mds>(this->state.gridCells[0]);
            this->state.gridsp[1] = this->state.box[1][1]/static_cast<real_mds>(this->state.gridCells[1]);
            this->state.gridsp[2] = this->state.box[2][2]/static_cast<real_mds>(this->state.gridCells[2]);
            this->state.gridsp[3] = this->state.gridsp[0]*this->state.gridsp[1];
            this->state.gridsp[4] = this->state.gridsp[0]*this->state.gridsp[2];
            this->state.gridsp[5] = this->state.gridsp[1]*this->state.gridsp[2];
            this->state.gridsp[6] = this->state.gridsp[0]*this->state.gridsp[1]*this->state.gridsp[2];
            this->state.invgridsp =  realval_mds(1.0)/(this->state.gridsp[0]*this->state.gridsp[1]*this->state.gridsp[2]);

#ifdef CUSTRESS_ENABLE
            if (this->settings.cuda)
                custress_update_box_spacings(this->state.box, this->state.invbox, this->state.gridsp);
#endif//CUSTRESS_ENABLE
        }
    }
}

void StressGrid::DispersionCorrection (real_ext shift)
{
    if (false == this->settings.nodispcor) {
        // only the 0th batch id adds the shift, so that it only gets added once
        int batch_id = this->m_thread_map[std::this_thread::get_id()];
        if (0 == batch_id) {
            // add shift to grid: note this is the 0th grid
            for(int i = 0; i < this->state.nCells; i++) {
                this->alloc.current_grid[i][0][0] += realval_mds(shift);
                this->alloc.current_grid[i][1][1] += realval_mds(shift);
                this->alloc.current_grid[i][2][2] += realval_mds(shift);
            }
        }
    }
}

void StressGrid::DistributeInteraction(
        int nAtoms,
        const array3_ext *R,
        const array3_ext *F,
        const real_ext *phi,
        const real_ext *kappa)
{
    int batch_id = this->m_thread_map[std::this_thread::get_id()];
    
    if (MDS_OK == this->state.ierr)
    {
        //------------------------------------------------------------------------------------
        matrix3_mds * stress_grid = this->alloc.current_grid+batch_id*this->state.nCells;
        matrix6_mds * elast_grid = this->alloc.current_grid_elborn+batch_id*this->state.nCells;

        // Call functions tailored for specific number of atoms
        if (2 == nAtoms) {
            const array3_mds Ra = {(real_mds)R[0][0], (real_mds)R[0][1], (real_mds)R[0][2]};
            const array3_mds Rb = {(real_mds)R[1][0], (real_mds)R[1][1], (real_mds)R[1][2]};
            const array3_mds Fa = {(real_mds)F[0][0], (real_mds)F[0][1], (real_mds)F[0][2]};
            distribute_n2(this->state, Ra, Rb, &Fa, stress_grid, &Ra, &Rb, phi, kappa, elast_grid);
        } else if (3 == nAtoms) {
            decompose_n3(this->state, R, F, stress_grid, phi, kappa, elast_grid);
        } else if (-3 == nAtoms) {
            decompose_n3(this->state, R, F, stress_grid, nullptr, nullptr, elast_grid);
        } else if (4 == nAtoms) {
            decompose_n4(this->state, R, F, stress_grid);
        } else if (5 == nAtoms) {
            decompose_n5(this->state, R, F, stress_grid, this->settings.fdecomp);
        } else {
            decompose_nbody(this->state, nAtoms, R, F, stress_grid);
        }
    }
}

void StressGrid::DistributeKinetic(
        real_ext mass,
        array3_ext x,
        array3_ext va,
        array3_ext vb = nullptr)
{
    matrix3_mds stress = {0};
    matrix6_mds elast = {0};

    if (MDS_OK == this->state.ierr)
    {
        // get the batch id
        int batch_id = this->m_thread_map[std::this_thread::get_id()];
        
        // select grid based on batch index
        matrix6_mds * elast_grid = this->alloc.current_grid_elkin+batch_id*this->state.nCells;
        matrix3_mds * stress_grid = this->alloc.current_grid+batch_id*this->state.nCells;
            
        if (vb == nullptr)
        {
            stress[0][0] = (real_mds)(-mass)*(real_mds)(va[0])*(real_mds)(va[0]);
            stress[0][1] = (real_mds)(-mass)*(real_mds)(va[0])*(real_mds)(va[1]);
            stress[0][2] = (real_mds)(-mass)*(real_mds)(va[0])*(real_mds)(va[2]);
            stress[1][0] = (real_mds)(-mass)*(real_mds)(va[1])*(real_mds)(va[0]);
            stress[1][1] = (real_mds)(-mass)*(real_mds)(va[1])*(real_mds)(va[1]);
            stress[1][2] = (real_mds)(-mass)*(real_mds)(va[1])*(real_mds)(va[2]);
            stress[2][0] = (real_mds)(-mass)*(real_mds)(va[2])*(real_mds)(va[0]);
            stress[2][1] = (real_mds)(-mass)*(real_mds)(va[2])*(real_mds)(va[1]);
            stress[2][2] = (real_mds)(-mass)*(real_mds)(va[2])*(real_mds)(va[2]);
            
            elast[0][0] = (real_mds)(mass)*realval_mds(4.0)*(real_mds)(va[0])*(real_mds)(va[0]); //c_xxxx = c_0000
            elast[1][1] = (real_mds)(mass)*realval_mds(4.0)*(real_mds)(va[1])*(real_mds)(va[1]); //c_yyyy = c_1111
            elast[2][2] = (real_mds)(mass)*realval_mds(4.0)*(real_mds)(va[2])*(real_mds)(va[2]); //c_zzzz = c_2222

            elast[0][4] = (real_mds)(mass)*((real_mds)(va[0])*(real_mds)(va[2]) + (real_mds)(va[0])*(real_mds)(va[2])); //c_xxxz = c_0002
            elast[0][5] = (real_mds)(mass)*((real_mds)(va[0])*(real_mds)(va[1]) + (real_mds)(va[0])*(real_mds)(va[1])); //c_xxxy = c_0001
            elast[1][3] = (real_mds)(mass)*((real_mds)(va[1])*(real_mds)(va[2]) + (real_mds)(va[1])*(real_mds)(va[2])); //c_yyyz = c_1112
            elast[1][5] = (real_mds)(mass)*((real_mds)(va[1])*(real_mds)(va[0]) + (real_mds)(va[1])*(real_mds)(va[0])); //c_yyxy = c_1101
            elast[2][3] = (real_mds)(mass)*((real_mds)(va[2])*(real_mds)(va[1]) + (real_mds)(va[2])*(real_mds)(va[1])); //c_zzyz = c_2212
            elast[2][4] = (real_mds)(mass)*((real_mds)(va[2])*(real_mds)(va[0]) + (real_mds)(va[2])*(real_mds)(va[0])); //c_zzxz = c_2202
            elast[4][4] = (real_mds)(mass)*((real_mds)(va[0])*(real_mds)(va[0]) + (real_mds)(va[2])*(real_mds)(va[2])); //c_xzxz = c_0202
            elast[5][5] = (real_mds)(mass)*((real_mds)(va[0])*(real_mds)(va[0]) + (real_mds)(va[1])*(real_mds)(va[1])); //c_xyxy = c_0101
            elast[3][3] = (real_mds)(mass)*((real_mds)(va[1])*(real_mds)(va[1]) + (real_mds)(va[2])*(real_mds)(va[2])); //c_yzyz = c_1212

            elast[3][4] = (real_mds)(mass)*(real_mds)(va[1])*(real_mds)(va[0]); //c_yzxz = c_1202
            elast[3][5] = (real_mds)(mass)*(real_mds)(va[2])*(real_mds)(va[0]); //c_yzxy = c_1201
            elast[4][5] = (real_mds)(mass)*(real_mds)(va[2])*(real_mds)(va[1]); //c_xzxy = c_0201
        }
        else
        {
            stress[0][0] = (real_mds)(-mass)*((real_mds)(va[0])*(real_mds)(va[0]) + (real_mds)(vb[0])*(real_mds)(vb[0]))/realval_mds(2.0);
            stress[0][1] = (real_mds)(-mass)*((real_mds)(va[0])*(real_mds)(va[1]) + (real_mds)(vb[0])*(real_mds)(vb[1]))/realval_mds(2.0);
            stress[0][2] = (real_mds)(-mass)*((real_mds)(va[0])*(real_mds)(va[2]) + (real_mds)(vb[0])*(real_mds)(vb[2]))/realval_mds(2.0);
            stress[1][0] = (real_mds)(-mass)*((real_mds)(va[1])*(real_mds)(va[0]) + (real_mds)(vb[1])*(real_mds)(vb[0]))/realval_mds(2.0);
            stress[1][1] = (real_mds)(-mass)*((real_mds)(va[1])*(real_mds)(va[1]) + (real_mds)(vb[1])*(real_mds)(vb[1]))/realval_mds(2.0);
            stress[1][2] = (real_mds)(-mass)*((real_mds)(va[1])*(real_mds)(va[2]) + (real_mds)(vb[1])*(real_mds)(vb[2]))/realval_mds(2.0);
            stress[2][0] = (real_mds)(-mass)*((real_mds)(va[2])*(real_mds)(va[0]) + (real_mds)(vb[2])*(real_mds)(vb[0]))/realval_mds(2.0);
            stress[2][1] = (real_mds)(-mass)*((real_mds)(va[2])*(real_mds)(va[1]) + (real_mds)(vb[2])*(real_mds)(vb[1]))/realval_mds(2.0);
            stress[2][2] = (real_mds)(-mass)*((real_mds)(va[2])*(real_mds)(va[2]) + (real_mds)(vb[2])*(real_mds)(vb[2]))/realval_mds(2.0);
            
            elast[0][0] = (real_mds)(mass)*realval_mds(4.0)*((real_mds)(va[0])*(real_mds)(va[0]) + (real_mds)(vb[0])*(real_mds)(vb[0]))/realval_mds(2.0); //c_xxxx = c_0000
            elast[1][1] = (real_mds)(mass)*realval_mds(4.0)*((real_mds)(va[1])*(real_mds)(va[1]) + (real_mds)(vb[1])*(real_mds)(vb[1]))/realval_mds(2.0); //c_yyyy = c_1111
            elast[2][2] = (real_mds)(mass)*realval_mds(4.0)*((real_mds)(va[2])*(real_mds)(va[2]) + (real_mds)(vb[2])*(real_mds)(vb[2]))/realval_mds(2.0); //c_zzzz = c_2222

            elast[0][4] = (real_mds)(mass)*((real_mds)(va[0])*(real_mds)(va[2]) + (real_mds)(va[0])*(real_mds)(va[2]) + (real_mds)(vb[0])*(real_mds)(vb[2]) + (real_mds)(vb[0])*(real_mds)(vb[2]))/realval_mds(2.0); //c_xxxz = c_0002
            elast[0][5] = (real_mds)(mass)*((real_mds)(va[0])*(real_mds)(va[1]) + (real_mds)(va[0])*(real_mds)(va[1]) + (real_mds)(vb[0])*(real_mds)(vb[1]) + (real_mds)(vb[0])*(real_mds)(vb[1]))/realval_mds(2.0); //c_xxxy = c_0001
            elast[1][3] = (real_mds)(mass)*((real_mds)(va[1])*(real_mds)(va[2]) + (real_mds)(va[1])*(real_mds)(va[2]) + (real_mds)(vb[1])*(real_mds)(vb[2]) + (real_mds)(vb[1])*(real_mds)(vb[2]))/realval_mds(2.0); //c_yyyz = c_1112
            elast[1][5] = (real_mds)(mass)*((real_mds)(va[1])*(real_mds)(va[0]) + (real_mds)(va[1])*(real_mds)(va[0]) + (real_mds)(vb[1])*(real_mds)(vb[0]) + (real_mds)(vb[1])*(real_mds)(vb[0]))/realval_mds(2.0); //c_yyxy = c_1101
            elast[2][3] = (real_mds)(mass)*((real_mds)(va[2])*(real_mds)(va[1]) + (real_mds)(va[2])*(real_mds)(va[1]) + (real_mds)(vb[2])*(real_mds)(vb[1]) + (real_mds)(vb[2])*(real_mds)(vb[1]))/realval_mds(2.0); //c_zzyz = c_2212
            elast[2][4] = (real_mds)(mass)*((real_mds)(va[2])*(real_mds)(va[0]) + (real_mds)(va[2])*(real_mds)(va[0]) + (real_mds)(vb[2])*(real_mds)(vb[0]) + (real_mds)(vb[2])*(real_mds)(vb[0]))/realval_mds(2.0); //c_zzxz = c_2202
            elast[3][3] = (real_mds)(mass)*((real_mds)(va[1])*(real_mds)(va[1]) + (real_mds)(va[2])*(real_mds)(va[2]) + (real_mds)(vb[1])*(real_mds)(vb[1]) + (real_mds)(vb[2])*(real_mds)(vb[2]))/realval_mds(2.0); //c_yzyz = c_1212
            elast[4][4] = (real_mds)(mass)*((real_mds)(va[0])*(real_mds)(va[0]) + (real_mds)(va[2])*(real_mds)(va[2]) + (real_mds)(vb[0])*(real_mds)(vb[0]) + (real_mds)(vb[2])*(real_mds)(vb[2]))/realval_mds(2.0); //c_xzxz = c_0202
            elast[5][5] = (real_mds)(mass)*((real_mds)(va[0])*(real_mds)(va[0]) + (real_mds)(va[1])*(real_mds)(va[1]) + (real_mds)(vb[0])*(real_mds)(vb[0]) + (real_mds)(vb[1])*(real_mds)(vb[1]))/realval_mds(2.0); //c_xyxy = c_0101

            elast[3][4] = (real_mds)(mass)*((real_mds)(va[1])*(real_mds)(va[0]) + (real_mds)(vb[1])*(real_mds)(vb[0]))/realval_mds(2.0); //c_yzxz = c_1202
            elast[3][5] = (real_mds)(mass)*((real_mds)(va[2])*(real_mds)(va[0]) + (real_mds)(vb[2])*(real_mds)(vb[0]))/realval_mds(2.0); //c_yzxy = c_1201
            elast[4][5] = (real_mds)(mass)*((real_mds)(va[2])*(real_mds)(va[1]) + (real_mds)(vb[2])*(real_mds)(vb[1]))/realval_mds(2.0); //c_xzxy = c_0201
        }

        // Get the coordinates of the point in the grid
        iarray i1 = {
            int(this->state.gridCells[0] * x[0] * this->state.invbox[0][0] - (x[0] < 0.0) ),
            int(this->state.gridCells[1] * x[1] * this->state.invbox[1][1] - (x[1] < 0.0) ),
            int(this->state.gridCells[2] * x[2] * this->state.invbox[2][2] - (x[2] < 0.0) )
        };
        
        // and the index constants
        const int iip1 = ((i1[0] + 1 + this->state.gridCells[0]) % this->state.gridCells[0])*this->state.gridCells[1]*this->state.gridCells[2];
        const int jjp1 = ((i1[1] + 1 + this->state.gridCells[1]) % this->state.gridCells[1])*this->state.gridCells[2];
        const int kkp1 = ((i1[2] + 1 + this->state.gridCells[2]) % this->state.gridCells[2]);
        const int iim1 = ((i1[0] + this->state.gridCells[0]) % this->state.gridCells[0])*this->state.gridCells[1]*this->state.gridCells[2];
        const int jjm1 = ((i1[1] + this->state.gridCells[1]) % this->state.gridCells[1])*this->state.gridCells[2];
        const int kkm1 = ((i1[2] + this->state.gridCells[2]) % this->state.gridCells[2]);
        
        // xc = vector from the corner of the point to the corner of the cell
        const array3_mds xc = {
            (real_mds)(x[0])-this->state.gridsp[0]*i1[0],
            (real_mds)(x[1])-this->state.gridsp[1]*i1[1],
            (real_mds)(x[2])-this->state.gridsp[2]*i1[2]
        };
        const array3_mds xd = {
            xc[0]-this->state.gridsp[0],
            xc[1]-this->state.gridsp[1],
            xc[2]-this->state.gridsp[2]
        };
        
        // Spread it
        const real_mds C = this->state.invgridsp * this->state.invgridsp;
        scalesummatrix3( C*xc[0]*xc[1]*xc[2],stress,stress_grid[iip1+jjp1+kkp1]);
        scalesummatrix3(-C*xc[0]*xc[1]*xd[2],stress,stress_grid[iip1+jjp1+kkm1]);
        scalesummatrix3(-C*xc[0]*xd[1]*xc[2],stress,stress_grid[iip1+jjm1+kkp1]);
        scalesummatrix3( C*xc[0]*xd[1]*xd[2],stress,stress_grid[iip1+jjm1+kkm1]);
        scalesummatrix3(-C*xd[0]*xc[1]*xc[2],stress,stress_grid[iim1+jjp1+kkp1]);
        scalesummatrix3( C*xd[0]*xc[1]*xd[2],stress,stress_grid[iim1+jjp1+kkm1]);
        scalesummatrix3( C*xd[0]*xd[1]*xc[2],stress,stress_grid[iim1+jjm1+kkp1]);
        scalesummatrix3(-C*xd[0]*xd[1]*xd[2],stress,stress_grid[iim1+jjm1+kkm1]);
        
        scalesummatrix6( C*xc[0]*xc[1]*xc[2],elast,elast_grid[iip1+jjp1+kkp1]);
        scalesummatrix6(-C*xc[0]*xc[1]*xd[2],elast,elast_grid[iip1+jjp1+kkm1]);
        scalesummatrix6(-C*xc[0]*xd[1]*xc[2],elast,elast_grid[iip1+jjm1+kkp1]);
        scalesummatrix6( C*xc[0]*xd[1]*xd[2],elast,elast_grid[iip1+jjm1+kkm1]);
        scalesummatrix6(-C*xd[0]*xc[1]*xc[2],elast,elast_grid[iim1+jjp1+kkp1]);
        scalesummatrix6( C*xd[0]*xc[1]*xd[2],elast,elast_grid[iim1+jjp1+kkm1]);
        scalesummatrix6( C*xd[0]*xd[1]*xc[2],elast,elast_grid[iim1+jjm1+kkp1]);
        scalesummatrix6(-C*xd[0]*xd[1]*xd[2],elast,elast_grid[iim1+jjm1+kkm1]);
    }
}

void StressGrid::SumGrid ( )
{
    if (MDS_OK == this->state.ierr)
    {
        // get the thread id
        int thread_id = this->m_thread_map[std::this_thread::get_id()];
        
        // every thread must process this latch before proceeding
        static barrier sumgrid_enter_stress_reduction(this->m_max_threads);
        sumgrid_enter_stress_reduction.count_down_and_wait();
        
        // reduce all stress current grids
        for (int i = thread_id; i < this->state.nCells; i+=this->m_max_threads)
        {
            if (i < this->state.nCells)
            {
                for (int j = 1; j < this->m_max_threads; ++j)
                {
                    summatrix3(this->alloc.current_grid[i], this->alloc.current_grid[i+j*this->state.nCells], this->alloc.current_grid[i] );
                    zeromatrix3(this->alloc.current_grid[i+j*this->state.nCells]);
                    summatrix6(this->alloc.current_grid_elborn[i], this->alloc.current_grid_elborn[i+j*this->state.nCells], this->alloc.current_grid_elborn[i] );
                    zeromatrix6(this->alloc.current_grid_elborn[i+j*this->state.nCells]);
                    summatrix6(this->alloc.current_grid_elkin[i], this->alloc.current_grid_elkin[i+j*this->state.nCells], this->alloc.current_grid_elkin[i] );
                    zeromatrix6(this->alloc.current_grid_elkin[i+j*this->state.nCells]);
                }
            }
        }

        // every thread must process this latch before proceeding
        static barrier sumgrid_continue(this->m_max_threads);
        sumgrid_continue.count_down_and_wait();

        if (thread_id == 0)
        {
#ifdef CUSTRESS_ENABLE
            if (this->settings.cuda)
                custress_sum_grid(this->alloc.current_grid);
#endif//CUSTRESS_ENABLE

            /*
            The covariance of the stress is computed using the online method that uses the co-moment
            as referenced in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

            x represents sigma_local_ij
            y represents sigma_total_kl

            at each iteration we first increment the counter
            n += 1

            then we compute the following:

            dx = x - meanx
            dy = y - meany
            meanx += dx / n
            meany += dy / n
            Covar += dx * dy

            after all iterations we scale Covar by 1/n

            The variance and mean box volume for simulations in the NPT ensemble (where V varies over time)
            are computed using a similar online algorithm

            at each iteraton we first increment the counter
            n += 1

            deltaV = Vnew - Vmean
            Vmean += deltaV / n
            deltaV2 = Vnew - Vmean
            M2 += deltaV * deltaV2

            after all iterations we scale M2 by 1/n to get Vvar
            Vvar = M2/n

            */

            this->state.nframes++;

            summatrix3( this->state.box, this->state.sumbox, this->state.sumbox ); //Update the cummulative system box - only used for box dimensions in output, it does not change elasticity/stress values

            real_mds Vnew, deltaV, deltaV2;
            Vnew = this->state.box[0][0]*this->state.box[1][1]*this->state.box[2][2];
            deltaV = Vnew - this->state.avg_boxvol;
            this->state.avg_boxvol += deltaV/this->state.nframes;
            deltaV2 = Vnew - this->state.avg_boxvol;
            this->state.var_boxvol += deltaV*deltaV2;

            for (int i = 0; i < this->state.nCells; i++)
                summatrix3(this->alloc.current_gridtot[0], this->alloc.current_grid[i], this->alloc.current_gridtot[0]); // compute y = sigma_total_kl = sum(sigma_local_kl)/this->state.nCells

            scalematrix3(this->alloc.current_gridtot[0], realval_mds(1.0)/this->state.nCells, this->alloc.current_gridtot[0]); // scale by this->state.nCells
            scalesummatrix3(realval_mds(-1.0), this->alloc.avg_gridtot[0], this->alloc.current_gridtot[0]); // subtract meany from y and store back into y (which now becomes dy)
            scalesummatrix3(realval_mds(1.0)/this->state.nframes, this->alloc.current_gridtot[0], this->alloc.avg_gridtot[0]); //compute meany
            scalesummatrix3(deltaV, this->alloc.current_gridtot[0], this->alloc.sum_gridtot_volcovar[0]); // accumulate the covar(sigma_total_kl, Vol)

            matrix6_mds tmp_covar[1];
            matrix3_mds dx[1];
            for (int i = 0; i < this->state.nCells; i++) {
                summatrix3( this->alloc.sum_grid[i], this->alloc.current_grid[i], this->alloc.sum_grid[i] );
                summatrix6( this->alloc.sum_grid_elborn[i], this->alloc.current_grid_elborn[i], this->alloc.sum_grid_elborn[i] );
                summatrix6( this->alloc.sum_grid_elkin[i], this->alloc.current_grid_elkin[i], this->alloc.sum_grid_elkin[i] );
                scalematrix3(this->alloc.avg_grid[i], realval_mds(-1.0), dx[0]); // dx = -meanx
                summatrix3(dx[0], this->alloc.current_grid[i], dx[0]); // dx += x
                scalesummatrix3(realval_mds(1.0)/this->state.nframes, dx[0], this->alloc.avg_grid[i]); //compute meanx
                scalesummatrix3(deltaV, dx[0], this->alloc.sum_grid_volcovar[i]); // accumulate covar(sigma_local_ij, Vol)
                matrixouterprod6( dx[0], this->alloc.current_gridtot[0], tmp_covar[0]); // dx*dy - uncomment this line to do local-vs-total fluctuations
                //matrixouterprod6( dx[0], dx[0], tmp_covar[0]); // dx*dy - uncomment this line to do local-vs-local fluctuations
                summatrix6(this->alloc.sum_grid_elcovar[i], tmp_covar[0], this->alloc.sum_grid_elcovar[i]); //this accumulates the covar of sigma_local_ij*sigma_total_kl
            }

            zeromatrix3(this->alloc.current_gridtot[0]);
            for(int i = 0; i < this->state.nCells; ++i) {
                zeromatrix3(this->alloc.current_grid[i]);
                zeromatrix6(this->alloc.current_grid_elborn[i]);
                zeromatrix6(this->alloc.current_grid_elkin[i]);
            }
        }

        // every thread must process this latch before exiting
        static barrier sumgrid_exit(this->m_max_threads);
        sumgrid_exit.count_down_and_wait();
    }
}

void StressGrid::Reset ( )
{
    if (MDS_OK == this->state.ierr)
    {
        // get the thread id
        int thread_id = this->m_thread_map[std::this_thread::get_id()];

        // every thread must process this latch before proceeding
        static barrier reset_entry(this->m_max_threads);
        reset_entry.count_down_and_wait();

        // every thread zeros its own current grid
        for( int i=0; i<this->state.nCells; i++ ) {
            zeromatrix3(this->alloc.current_grid[i+thread_id*this->m_max_threads]);
            zeromatrix6(this->alloc.current_grid_elborn[i+thread_id*this->m_max_threads]);
            zeromatrix6(this->alloc.current_grid_elkin[i+thread_id*this->m_max_threads]);
        }

        if (thread_id == 0) {
            // thread 0 deals with the sum grid
            zeromatrix3(this->alloc.current_gridtot[0]);
            zeromatrix3(this->alloc.avg_gridtot[0]);
            for( int i=0; i<this->state.nCells; i++ ) {
                zeromatrix3( this->alloc.sum_grid[i] );
                zeromatrix3( this->alloc.avg_grid[i] );
                zeromatrix3( this->alloc.sum_grid_volcovar[i] );
                zeromatrix6( this->alloc.sum_grid_elborn[i] );
                zeromatrix6( this->alloc.sum_grid_elcovar[i] );
                zeromatrix6( this->alloc.sum_grid_elkin[i] );
            }

            this->state.nframes = 0;
            this->state.nreset ++;
            this->state.avg_boxvol = 0.0;
            this->state.var_boxvol = 0.0;
        }

        // every thread must process this latch before exiting
        static barrier reset_exit(this->m_max_threads);
        reset_exit.count_down_and_wait();
    }
}

void StressGrid::Write ( )
{
    if (MDS_OK == this->state.ierr) {
        int         Dtype=1;
        std::string outname, elborn_outname, elcovar_outname, elkin_outname, eltotal_outname, eltotalhooke_outname;
        std::ostringstream outnumber;
        FILE *outfile = nullptr;
        FILE *elborn_outfile = nullptr;
        FILE *elcovar_outfile = nullptr;
        FILE *elkin_outfile = nullptr;
        FILE *eltotal_outfile = nullptr;
        FILE *eltotalhooke_outfile = nullptr;

        outnumber << this->state.nreset;
        size_t lastindex = this->m_filename.find_last_of(".");
        std::string rawname = this->m_filename.substr(0, lastindex);

        //Change output format if the user specifies a filename with a .dat extension
        outname = this->m_filename + outnumber.str();
        if (outname.find(".dat") == std::string::npos)
            outname = outname + ".mds";

        elcovar_outname = rawname + "_elcovar.dat" + outnumber.str();
        elkin_outname = rawname + "_elkin.dat" + outnumber.str();
        elborn_outname = rawname + "_elborn.dat" + outnumber.str();
        eltotal_outname = rawname + "_eltotal.dat" + outnumber.str();
        eltotalhooke_outname = rawname + "_eltotalhooke.dat" + outnumber.str();

        // open the main output file
        outfile = fopen(outname.c_str(), "wb" );
        Dtype = 1;
        fwrite(&Dtype, sizeof(int), 1, outfile);

        // use different dtype for elasticity file
        elcovar_outfile = fopen(elcovar_outname.c_str(), "wb" );
        elborn_outfile = fopen(elborn_outname.c_str(), "wb" );
        elkin_outfile = fopen(elkin_outname.c_str(), "wb" );
        eltotal_outfile = fopen(eltotal_outname.c_str(), "wb" );
        eltotalhooke_outfile = fopen(eltotalhooke_outname.c_str(), "wb" );

        Dtype = 6;
        fwrite(&Dtype, sizeof(int), 1, elcovar_outfile);
        fwrite(&Dtype, sizeof(int), 1, elborn_outfile);
        fwrite(&Dtype, sizeof(int), 1, elkin_outfile);
        fwrite(&Dtype, sizeof(int), 1, eltotal_outfile);
        fwrite(&Dtype, sizeof(int), 1, eltotalhooke_outfile);

        //Divide sumbox with respect to the number of frames to get the avg
        matrix3_mds        avgbox;
        scalematrix3( this->state.sumbox, realval_mds(1.0)/this->state.nframes, avgbox);

        //Need to copy to an array of output precision
        matrix3_out        avgbox_out;
        copymatrix3(avgbox, avgbox_out);

        // writing average box to each outfile
        fwrite(avgbox_out, sizeof(matrix3_out), 1, outfile);
        fwrite(avgbox_out, sizeof(matrix3_out), 1, elcovar_outfile);
        fwrite(avgbox_out, sizeof(matrix3_out), 1, elborn_outfile);
        fwrite(avgbox_out, sizeof(matrix3_out), 1, elkin_outfile);
        fwrite(avgbox_out, sizeof(matrix3_out), 1, eltotal_outfile);
        fwrite(avgbox_out, sizeof(matrix3_out), 1, eltotalhooke_outfile);

        // this is the number of grids in all dimensions, also to all files
        fwrite(&this->state.gridCells[0], sizeof(this->state.gridCells[0]), 1, outfile);
        fwrite(&this->state.gridCells[1], sizeof(this->state.gridCells[1]), 1, outfile);
        fwrite(&this->state.gridCells[2], sizeof(this->state.gridCells[2]), 1, outfile);
        fwrite(&this->state.gridCells[0], sizeof(this->state.gridCells[0]), 1, elcovar_outfile);
        fwrite(&this->state.gridCells[1], sizeof(this->state.gridCells[1]), 1, elcovar_outfile);
        fwrite(&this->state.gridCells[2], sizeof(this->state.gridCells[2]), 1, elcovar_outfile);
        fwrite(&this->state.gridCells[0], sizeof(this->state.gridCells[0]), 1, elborn_outfile);
        fwrite(&this->state.gridCells[1], sizeof(this->state.gridCells[1]), 1, elborn_outfile);
        fwrite(&this->state.gridCells[2], sizeof(this->state.gridCells[2]), 1, elborn_outfile);
        fwrite(&this->state.gridCells[0], sizeof(this->state.gridCells[0]), 1, elkin_outfile);
        fwrite(&this->state.gridCells[1], sizeof(this->state.gridCells[1]), 1, elkin_outfile);
        fwrite(&this->state.gridCells[2], sizeof(this->state.gridCells[2]), 1, elkin_outfile);
        fwrite(&this->state.gridCells[0], sizeof(this->state.gridCells[0]), 1, eltotal_outfile);
        fwrite(&this->state.gridCells[1], sizeof(this->state.gridCells[1]), 1, eltotal_outfile);
        fwrite(&this->state.gridCells[2], sizeof(this->state.gridCells[2]), 1, eltotal_outfile);
        fwrite(&this->state.gridCells[0], sizeof(this->state.gridCells[0]), 1, eltotalhooke_outfile);
        fwrite(&this->state.gridCells[1], sizeof(this->state.gridCells[1]), 1, eltotalhooke_outfile);
        fwrite(&this->state.gridCells[2], sizeof(this->state.gridCells[2]), 1, eltotalhooke_outfile);

        // calculate stress factors
        real_mds stressfac, covfac, covfac2;
        stressfac = realval_mds(mds_units);
        covfac = -realval_mds(mds_units)*realval_mds(mds_units)*this->state.avg_boxvol/(this->settings.temperature*realval_mds(KBfac)*this->state.nframes);
        //covfac /= this->state.nCells; // uncomment this for for local-vs-local fluctuations
        covfac2 = realval_mds(0.0);
        if (this->settings.pcoupl == true && this->state.nframes > 1)
            covfac2 = realval_mds(mds_units)*realval_mds(mds_units)*this->state.avg_boxvol/(this->settings.temperature*realval_mds(KBfac)*this->state.var_boxvol*this->state.nframes);

        // need to store matrices in double precision
        matrix3_out sum_grid = {0};
        matrix6_out sum_grid_elcovar = {0};
        matrix6_out sum_grid_elborn = {0};
        matrix6_out sum_grid_elkin = {0};
        matrix3_out s = {0};
        matrix6_out elast = {0};

        // need this for corrections below
        matrix6_mds npt_covar_corr[1];
        for ( int i = 0; i < this->state.nCells; i++ )
        {
            //scalematrix3(this->alloc.sum_grid[i], stressfac, this->alloc.sum_grid[i]);
            scalematrix3(this->alloc.avg_grid[i], stressfac, this->alloc.avg_grid[i]); // Use the online average stress instead of the cummulative stress divided by the number of frames
            scalematrix6(this->alloc.sum_grid_elcovar[i], covfac, this->alloc.sum_grid_elcovar[i]);
            scalematrix6(this->alloc.sum_grid_elborn[i], stressfac/this->state.nframes, this->alloc.sum_grid_elborn[i]);
            scalematrix6(this->alloc.sum_grid_elkin[i], stressfac/this->state.nframes, this->alloc.sum_grid_elkin[i]);

            if (this->settings.pcoupl == true && this->state.nframes > 1)
            {
                matrixouterprod6(this->alloc.sum_grid_volcovar[i], this->alloc.sum_gridtot_volcovar[0], npt_covar_corr[0]); // Cov(sigma_local_ij,V)*Cov(sigma_total_kl,V)
                scalematrix6(npt_covar_corr[0], covfac2, npt_covar_corr[0]); // scale the term above by <V>/kT*Var(V)
                summatrix6(this->alloc.sum_grid_elcovar[i], npt_covar_corr[0], this->alloc.sum_grid_elcovar[i]);
            }

            // copying to dmatrix3/6 here for m_ncell writes to respective files
            //copymatrix3(this->alloc.sum_grid[i], sum_grid);
            copymatrix3(this->alloc.avg_grid[i], sum_grid);
            copymatrix6(this->alloc.sum_grid_elcovar[i], sum_grid_elcovar);
            copymatrix6(this->alloc.sum_grid_elborn[i], sum_grid_elborn);
            copymatrix6(this->alloc.sum_grid_elkin[i], sum_grid_elkin);
            fwrite(&sum_grid[0], sizeof(matrix3_out), 1, outfile);
            fwrite(&sum_grid_elcovar[0], sizeof(matrix6_out), 1, elcovar_outfile);
            fwrite(&sum_grid_elborn[0], sizeof(matrix6_out), 1, elborn_outfile);
            fwrite(&sum_grid_elkin[0], sizeof(matrix6_out), 1, elkin_outfile);
        }

        fclose(outfile);
        fclose(elcovar_outfile);
        fclose(elborn_outfile);
        fclose(elkin_outfile);
        for ( int i = 0; i < this->state.nCells; i++ )
        {
            summatrix6(this->alloc.sum_grid_elcovar[i], this->alloc.sum_grid_elborn[i], this->alloc.sum_grid_elcovar[i]); // add up all the contributions to the total elasticity tensor
            summatrix6(this->alloc.sum_grid_elcovar[i], this->alloc.sum_grid_elkin[i], this->alloc.sum_grid_elcovar[i]);

            // need to store matrices in double precision
            copymatrix6(this->alloc.sum_grid_elcovar[i], elast);
            fwrite(&elast[0], sizeof(matrix6_out), 1, eltotal_outfile);

            // Correct the elasticity tensor using the stress tensor to obtain the Hooke's law elasticity tensor
            //
            // Stiffness matrix in Voigt notation
            // 0 = xx; 1 = yy; 2 = zz; 3 = yz or zy; 4 = xz or zx; 5 = xy or yx
            // All indices                         Voigt indices           Stress indices
            // ( xxxx xxyy xxzz xxyz xxxz xxxy ) = ( 00 01 02 03 04 05 ) = [ 0000 0011 0022 0012 0002 0001 ]
            // ( yyxx yyyy yyzz yyyz yyxz yyxy ) = ( 10 11 12 13 14 15 ) = [ 1100 1111 1122 1112 1102 1101 ]
            // ( zzxx zzyy zzzz zzyz zzxz zzxy ) = ( 20 21 22 23 24 25 ) = [ 2200 2211 2222 2212 2202 2201 ]
            // ( yzxx yzyy yzzz yzyz yzxz yzxy ) = ( 30 31 32 33 34 35 ) = [ 1200 1211 1222 1212 1202 1201 ]
            // ( xzxx xzyy xzzz xzyz xzxz xzxy ) = ( 40 41 42 43 44 45 ) = [ 0200 0211 0222 0212 0202 0201 ]
            // ( xyxx xyyy xyzz xyyz xyxz xyxy ) = ( 50 51 52 53 54 55 ) = [ 0100 0111 0122 0112 0102 0101 ]

            // C_hook[ijkl] = C[ijkl] + 1/2*( s[ik]d[jl] + s[il]d[jk] + s[jk]d[il] + s[jl]d[ik] - 2*s[ij]d[kl])

            copymatrix3(this->alloc.avg_grid[i], s);
                                                       // ijkl
            elast[0][0] +=  s[0][0];                   // 0000
            elast[0][1] += -s[0][0];                   // 0011
            elast[0][2] += -s[0][0];                   // 0022
            elast[0][3] +=  0;                         // 0012
            elast[0][4] +=  s[0][2];                   // 0002
            elast[0][5] +=  s[0][1];                   // 0001
            elast[1][0] += -s[1][1];                   // 1100
            elast[1][1] +=  s[1][1];                   // 1111
            elast[1][2] += -s[1][1];                   // 1122
            elast[1][3] +=  s[1][2];                   // 1112
            elast[1][4] +=  0;                         // 1102
            elast[1][5] +=  s[1][0];                   // 1101
            elast[2][0] += -s[2][2];                   // 2200
            elast[2][1] += -s[2][2];                   // 2211
            elast[2][2] +=  s[2][2];                   // 2222
            elast[2][3] +=  s[2][1];                   // 2212
            elast[2][4] +=  s[2][0];                   // 2202
            elast[2][5] +=  0;                         // 2201
            elast[3][0] += -s[1][2];                   // 1200
            elast[3][1] +=  s[2][1]-s[1][2];           // 1211
            elast[3][2] +=  0;                         // 1222
            elast[3][3] +=  0.5*(s[1][1] + s[2][2]);   // 1212
            elast[3][4] +=  0.5*(s[1][0]);             // 1202
            elast[3][5] +=  0.5*(s[2][0]);             // 1201
            elast[4][0] +=  s[2][0]-s[0][2];           // 0200
            elast[4][1] += -s[0][2];                   // 0211
            elast[4][2] +=  0;                         // 0222
            elast[4][3] +=  0.5*(s[0][1]);             // 0212
            elast[4][4] +=  0.5*(s[0][0] + s[2][2]);   // 0202
            elast[4][5] +=  0.5*(s[2][1]);             // 0201
            elast[5][0] +=  s[1][0]-s[0][1];           // 0100
            elast[5][1] +=  0;                         // 0111
            elast[5][2] += -s[0][1];                   // 0122
            elast[5][3] +=  0.5*(s[0][2]);             // 0112
            elast[5][4] +=  0.5*(s[1][2]);             // 0102
            elast[5][5] +=  0.5*(s[0][0] + s[1][1]);   // 0101

            // need to store matrices in double precision
            fwrite(&elast[0], sizeof(matrix6_out), 1, eltotalhooke_outfile);
        }

        fclose(eltotal_outfile);
        fclose(eltotalhooke_outfile);
    }
}


/**
 * PRIVATE CLASS FUNCTIONS
 */
void StressGrid::Clear() {
    // free any allocated memory
    if (this->alloc.current_grid         != nullptr ) delete [] this->alloc.current_grid;
    if (this->alloc.current_gridtot      != nullptr ) delete [] this->alloc.current_gridtot;
    if (this->alloc.current_grid_elborn  != nullptr ) delete [] this->alloc.current_grid_elborn;
    if (this->alloc.current_grid_elkin   != nullptr ) delete [] this->alloc.current_grid_elkin;
    if (this->alloc.sum_grid             != nullptr ) delete [] this->alloc.sum_grid;
    if (this->alloc.avg_grid             != nullptr ) delete [] this->alloc.avg_grid;
    if (this->alloc.avg_gridtot          != nullptr ) delete [] this->alloc.avg_gridtot;
    if (this->alloc.sum_grid_elcovar     != nullptr ) delete [] this->alloc.sum_grid_elcovar;
    if (this->alloc.sum_grid_elkin       != nullptr ) delete [] this->alloc.sum_grid_elkin;
    if (this->alloc.sum_grid_elborn      != nullptr ) delete [] this->alloc.sum_grid_elborn;
    if (this->alloc.sum_grid_volcovar    != nullptr ) delete [] this->alloc.sum_grid_volcovar;
    if (this->alloc.sum_gridtot_volcovar != nullptr ) delete [] this->alloc.sum_gridtot_volcovar;

    // clear the  pointers
    memset(&this->alloc, 0, sizeof(alloc_t) );

    // clear portions of the state
    this->state.nframes = 0;
    this->state.avg_boxvol = realval_mds(0.0);
    this->state.var_boxvol = realval_mds(0.0);

    if (this->settings.initialized)
        this->settings.initialized = false;
    
#ifdef CUSTRESS_ENABLE
    if (this->settings.cuda)
        custress_clear();
#endif//CUSTRESS_ENABLE
}
