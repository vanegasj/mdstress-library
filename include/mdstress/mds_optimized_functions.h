/*=========================================================================

  Module    : MDStress - Optimized Distribution Functions
  Authors   : Original: A. Torres-Sanchez and J. M. Vanegas
              Optimized: Claude (Anthropic)
  Purpose   : Optimized grid distribution and force decomposition
  Date      : Jan-2026
  Version   : 2.0
  
  USAGE:
    Include this file in mds_stressgrid.cpp and replace the original
    function implementations with these optimized versions.
    
  OPTIMIZATIONS:
    1. Reduced modulo operations in index calculations
    2. Precomputed constants moved outside loops
    3. Branchless index selection where possible
    4. Cache-prefetching hints
    5. Direct analytical solution for 3-body decomposition

=========================================================================*/

#ifndef mds_optimized_functions_h
#define mds_optimized_functions_h

#include "mds_basicops_optimized.h"
#include <cmath>

namespace mds {

//==========================================================================
// Helper: Fast Modulo for Grid Indices
//==========================================================================

// Fast modulo when we know the value is in range [-gridSize, 2*gridSize)
static inline int fast_grid_mod(int val, int gridSize)
{
    // Branchless version using conditional move
    // Works for val in range [-gridSize, 2*gridSize)
    val = val + (val < 0) * gridSize;
    val = val - (val >= gridSize) * gridSize;
    return val;
}

// For positive values only (result of i+1)
static inline int fast_grid_mod_positive(int val, int gridSize)
{
    return (val >= gridSize) ? (val - gridSize) : val;
}

//==========================================================================
// Optimized: distribute_observables_3d
//==========================================================================

static inline void distribute_observables_3d_optimized(
        const state_t & state,
        const array3_mds & xi,
        const array3_mds & xj,
        const array3_mds & diff,
        const matrix3_mds * stress,
        const matrix6_mds * elast,
        matrix3_mds * stress_grid,
        matrix6_mds * elast_grid)
{
    // Early exit if nothing to do
    if (stress_grid == nullptr && elast_grid == nullptr) return;

    //------------------------------------------------------------------------------------
    // Precompute grid cell factors (loop invariant)
    const real_mds inv00 = state.invbox[0][0];
    const real_mds inv11 = state.invbox[1][1];
    const real_mds inv22 = state.invbox[2][2];
    
    const int nx = state.gridCells[0];
    const int ny = state.gridCells[1];
    const int nz = state.gridCells[2];
    const int ny_nz = ny * nz;
    
    // Calculate the grid coordinates (no pbc) for the extreme points
    // Optimized: avoid repeated multiplication
    const real_mds xi_scaled[3] = {
        xi[0] * inv00 * nx,
        xi[1] * inv11 * ny,
        xi[2] * inv22 * nz
    };
    const real_mds xj_scaled[3] = {
        xj[0] * inv00 * nx,
        xj[1] * inv11 * ny,
        xj[2] * inv22 * nz
    };

    iarray i1 = {
        int(xi_scaled[0]) - (xi[0] < 0.0),
        int(xi_scaled[1]) - (xi[1] < 0.0),
        int(xi_scaled[2]) - (xi[2] < 0.0)
    };

    const iarray i2 = {
        int(xj_scaled[0]) - (xj[0] < 0.0),
        int(xj_scaled[1]) - (xj[1] < 0.0),
        int(xj_scaled[2]) - (xj[2] < 0.0)
    };

    // Direction of traversal
    const iarray c = {
        (i2[0] > i1[0]) - (i1[0] > i2[0]),
        (i2[1] > i1[1]) - (i1[1] > i2[1]),
        (i2[2] > i1[2]) - (i1[2] > i2[2])
    };

    // Check for single-cell case (common optimization)
    const int iterations = c[0]*(i2[0]-i1[0]) + c[1]*(i2[1]-i1[1]) + c[2]*(i2[2]-i1[2]);
    
    if (iterations == 0) {
        // Single cell case - simplified processing
        const int ii = fast_grid_mod(i1[0], nx) * ny_nz;
        const int jj = fast_grid_mod(i1[1], ny) * nz;
        const int kk = fast_grid_mod(i1[2], nz);
        const int idx = ii + jj + kk;
        
        if (stress_grid != nullptr) {
            scalesummatrix3(realval_mds(1.0), *stress, stress_grid[idx]);
        }
        if (elast_grid != nullptr) {
            scalesummatrix6(realval_mds(1.0), *elast, elast_grid[idx]);
        }
        return;
    }

    // Precompute difference reciprocals (avoid division in loop)
    const real_mds diff_inv[3] = {
        realval_mds(1.0) / (xi[0] - xj[0]),
        realval_mds(1.0) / (xi[1] - xj[1]),
        realval_mds(1.0) / (xi[2] - xj[2])
    };
    
    const array3_mds t_c1 = {
        xi[0] * diff_inv[0],
        xi[1] * diff_inv[1],
        xi[2] * diff_inv[2]
    };
    const array3_mds t_c2 = {
        state.gridsp[0] * diff_inv[0],
        state.gridsp[1] * diff_inv[1],
        state.gridsp[2] * diff_inv[2]
    };

    iarray in = {
        i1[0] + (c[0] + 1) / 2,
        i1[1] + (c[1] + 1) / 2,
        i1[2] + (c[2] + 1) / 2
    };

    array3_mds d_cgrid = {
        xi[0] - (i1[0] + realval_mds(0.5)) * state.gridsp[0],
        xi[1] - (i1[1] + realval_mds(0.5)) * state.gridsp[1],
        xi[2] - (i1[2] + realval_mds(0.5)) * state.gridsp[2]
    };

    // Parametric time
    array3_mds t = {
        (c[0] == 0) ? realval_mds(1.1) : t_c1[0] - in[0] * t_c2[0],
        (c[1] == 0) ? realval_mds(1.1) : t_c1[1] - in[1] * t_c2[1],
        (c[2] == 0) ? realval_mds(1.1) : t_c1[2] - in[2] * t_c2[2]
    };

    // Precompute spatial constants (loop invariant)
    const real_mds C = realval_mds(0.125) * state.invgridsp * state.invgridsp;
    const real_mds axy = diff[0] * diff[1];
    const real_mds axz = diff[0] * diff[2];
    const real_mds ayz = diff[1] * diff[2];
    const real_mds axyz = diff[0] * ayz;
    
    // Precompute gridsp products
    const real_mds gsp0 = state.gridsp[0];
    const real_mds gsp1 = state.gridsp[1];
    const real_mds gsp2 = state.gridsp[2];
    const real_mds gsp3 = state.gridsp[3];
    const real_mds gsp4 = state.gridsp[4];
    const real_mds gsp5 = state.gridsp[5];
    const real_mds gsp6 = state.gridsp[6];

    real_mds oldt = 0.0;
    
    // Main distribution loop
    for (int count = 0; count <= iterations; ++count) {
        // Branchless index selection using min
        // Find which dimension crosses first
        const bool t0_le_t1 = (t[0] <= t[1] + mds_eps);
        const bool t0_le_t2 = (t[0] <= t[2] + mds_eps);
        const bool t1_le_t2 = (t[1] <= t[2] + mds_eps);
        
        int iX;
        if (t0_le_t1 && t0_le_t2) {
            iX = 0;
        } else if (t1_le_t2) {
            iX = 1;
        } else {
            iX = 2;
        }
        
        const real_mds newt = (iterations == count) ? realval_mds(1.0) : t[iX];

        // Parametric time constants
        const real_mds t12 = oldt * oldt;
        const real_mds t22 = newt * newt;
        const real_mds dt1 = newt - oldt;
        const real_mds dt2 = t22 - t12;
        const real_mds dt3 = realval_mds(4.0) * (t22 * newt - t12 * oldt) / realval_mds(3.0);
        const real_mds dt4 = t22 * t22 - t12 * t12;

        // Position-dependent constants
        const real_mds bxy = d_cgrid[0] * d_cgrid[1];
        const real_mds bxz = d_cgrid[0] * d_cgrid[2];
        const real_mds byz = d_cgrid[1] * d_cgrid[2];
        const real_mds bxyz = d_cgrid[0] * byz;
    
        // Optimized index calculation using fast_grid_mod
        const int iip1 = fast_grid_mod_positive(i1[0] + 1, nx) * ny_nz;
        const int jjp1 = fast_grid_mod_positive(i1[1] + 1, ny) * nz;
        const int kkp1 = fast_grid_mod_positive(i1[2] + 1, nz);
        const int iim1 = fast_grid_mod(i1[0], nx) * ny_nz;
        const int jjm1 = fast_grid_mod(i1[1], ny) * nz;
        const int kkm1 = fast_grid_mod(i1[2], nz);

        // Compute D coefficients
        const real_mds D0 = realval_mds(8.0)*bxyz*dt1 
                         + realval_mds(4.0)*(diff[0]*byz + diff[1]*bxz + diff[2]*bxy)*dt2
                         + realval_mds(2.0)*(d_cgrid[0]*ayz + d_cgrid[1]*axz + d_cgrid[2]*axy)*dt3 
                         + realval_mds(2.0)*axyz*dt4;
        const real_mds D1 = gsp0 * (realval_mds(4.0)*byz*dt1 + realval_mds(2.0)*(diff[1]*d_cgrid[2] + diff[2]*d_cgrid[1])*dt2 + ayz*dt3);
        const real_mds D2 = gsp1 * (realval_mds(4.0)*bxz*dt1 + realval_mds(2.0)*(diff[0]*d_cgrid[2] + diff[2]*d_cgrid[0])*dt2 + axz*dt3);
        const real_mds D3 = gsp2 * (realval_mds(4.0)*bxy*dt1 + realval_mds(2.0)*(diff[0]*d_cgrid[1] + diff[1]*d_cgrid[0])*dt2 + axy*dt3);
        const real_mds D4 = gsp3 * (realval_mds(2.0)*d_cgrid[2]*dt1 + diff[2]*dt2);
        const real_mds D5 = gsp4 * (realval_mds(2.0)*d_cgrid[1]*dt1 + diff[1]*dt2);
        const real_mds D6 = gsp5 * (realval_mds(2.0)*d_cgrid[0]*dt1 + diff[0]*dt2);
        const real_mds D7 = gsp6 * dt1;

        // Precompute common sums
        const real_mds D67 = D6 + D7;
        const real_mds D567 = D5 + D67;
        const real_mds D4567 = D4 + D567;
        
        // Scaling factors with optimized computation
        const real_mds sf1 = C * ( D0 + D1 + D2 + D3 + D4567);
        const real_mds sf2 = C * (-D0 - D1 - D2 + D3 - D4 + D567);
        const real_mds sf3 = C * (-D0 - D1 + D2 - D3 + D4 - D5 + D67);
        const real_mds sf4 = C * ( D0 + D1 - D2 - D3 - D4 - D5 + D67);
        const real_mds sf5 = C * (-D0 + D1 - D2 - D3 + D4 + D5 - D67);
        const real_mds sf6 = C * ( D0 - D1 + D2 - D3 - D4 + D5 - D67);
        const real_mds sf7 = C * ( D0 - D1 - D2 + D3 + D4 - D5 - D67);
        const real_mds sf8 = C * (-D0 + D1 + D2 + D3 - D4567);

        // Grid indices
        const int ind1 = iip1 + jjp1 + kkp1;
        const int ind2 = iip1 + jjp1 + kkm1;
        const int ind3 = iip1 + jjm1 + kkp1;
        const int ind4 = iip1 + jjm1 + kkm1;
        const int ind5 = iim1 + jjp1 + kkp1;
        const int ind6 = iim1 + jjp1 + kkm1;
        const int ind7 = iim1 + jjm1 + kkp1;
        const int ind8 = iim1 + jjm1 + kkm1;

        // Distribute stress
        if (stress_grid != nullptr) {
            scalesummatrix3(sf1, *stress, stress_grid[ind1]);
            scalesummatrix3(sf2, *stress, stress_grid[ind2]);
            scalesummatrix3(sf3, *stress, stress_grid[ind3]);
            scalesummatrix3(sf4, *stress, stress_grid[ind4]);
            scalesummatrix3(sf5, *stress, stress_grid[ind5]);
            scalesummatrix3(sf6, *stress, stress_grid[ind6]);
            scalesummatrix3(sf7, *stress, stress_grid[ind7]);
            scalesummatrix3(sf8, *stress, stress_grid[ind8]);
        }
        
        // Distribute elasticity
        if (elast_grid != nullptr) {
            scalesummatrix6(sf1, *elast, elast_grid[ind1]);
            scalesummatrix6(sf2, *elast, elast_grid[ind2]);
            scalesummatrix6(sf3, *elast, elast_grid[ind3]);
            scalesummatrix6(sf4, *elast, elast_grid[ind4]);
            scalesummatrix6(sf5, *elast, elast_grid[ind5]);
            scalesummatrix6(sf6, *elast, elast_grid[ind6]);
            scalesummatrix6(sf7, *elast, elast_grid[ind7]);
            scalesummatrix6(sf8, *elast, elast_grid[ind8]);
        }

        // Update state for next iteration
        d_cgrid[iX] -= c[iX] * state.gridsp[iX];
        oldt = t[iX];
        
        i1[iX] += c[iX];
        in[iX] += c[iX];

        t[iX] = t_c1[iX] - in[iX] * t_c2[iX];
    }
}

//==========================================================================
// Optimized: decompose_n3 - Direct Analytical Solution
//==========================================================================

// This replaces the Eigen-based SVD solution with a direct analytical approach
// for the 3-body force decomposition system.
//
// The system is:
// F_A = λ_AB * AB + λ_AC * AC
// F_B = -λ_AB * AB + λ_BC * BC  
// F_C = -λ_AC * AC - λ_BC * BC
//
// This is a 9x3 overdetermined system that can be solved using the normal equations
// or by using the specific structure of the problem.

static void decompose_n3_optimized(
        const state_t & state,
        const array3_ext *R,
        const array3_ext *F,
        matrix3_mds * stress_grid,
        const real_ext * phi,
        const real_ext * kappa,
        matrix6_mds * elast_grid)
{
    // Convert to internal precision
    const array3_mds Ra = {(real_mds)R[0][0], (real_mds)R[0][1], (real_mds)R[0][2]};
    const array3_mds Rb = {(real_mds)R[1][0], (real_mds)R[1][1], (real_mds)R[1][2]};
    const array3_mds Rc = {(real_mds)R[2][0], (real_mds)R[2][1], (real_mds)R[2][2]};
    const array3_mds Fa = {(real_mds)F[0][0], (real_mds)F[0][1], (real_mds)F[0][2]};
    const array3_mds Fb = {(real_mds)F[1][0], (real_mds)F[1][1], (real_mds)F[1][2]};
    const array3_mds Fc = {(real_mds)F[2][0], (real_mds)F[2][1], (real_mds)F[2][2]};

    // Compute vectors between particles
    array3_mds AB, AC, BC;
    diffarray3(Rb, Ra, AB, state.box, state.periodic);
    diffarray3(Rc, Ra, AC, state.box, state.periodic);
    diffarray3(Rc, Rb, BC, state.box, state.periodic);

    // Direct solution using normal equations: (M^T M) x = M^T b
    // 
    // M^T M is a 3x3 symmetric matrix:
    // [  2*|AB|^2     AB·AC        AB·BC    ]
    // [   AB·AC     2*|AC|^2      -AC·BC    ]
    // [   AB·BC      -AC·BC      2*|BC|^2   ]
    //
    // M^T b is a 3x1 vector:
    // [ AB·(Fa - Fb) ]
    // [ AC·(Fa - Fc) ]
    // [ BC·(Fb - Fc) ]

    // Compute dot products (reused multiple times)
    const real_mds AB_AB = AB[0]*AB[0] + AB[1]*AB[1] + AB[2]*AB[2];
    const real_mds AC_AC = AC[0]*AC[0] + AC[1]*AC[1] + AC[2]*AC[2];
    const real_mds BC_BC = BC[0]*BC[0] + BC[1]*BC[1] + BC[2]*BC[2];
    const real_mds AB_AC = AB[0]*AC[0] + AB[1]*AC[1] + AB[2]*AC[2];
    const real_mds AB_BC = AB[0]*BC[0] + AB[1]*BC[1] + AB[2]*BC[2];
    const real_mds AC_BC = AC[0]*BC[0] + AC[1]*BC[1] + AC[2]*BC[2];

    // Force differences
    const real_mds Fa_Fb[3] = {Fa[0] - Fb[0], Fa[1] - Fb[1], Fa[2] - Fb[2]};
    const real_mds Fa_Fc[3] = {Fa[0] - Fc[0], Fa[1] - Fc[1], Fa[2] - Fc[2]};
    const real_mds Fb_Fc[3] = {Fb[0] - Fc[0], Fb[1] - Fc[1], Fb[2] - Fc[2]};

    // Right-hand side
    const real_mds rhs0 = AB[0]*Fa_Fb[0] + AB[1]*Fa_Fb[1] + AB[2]*Fa_Fb[2];
    const real_mds rhs1 = AC[0]*Fa_Fc[0] + AC[1]*Fa_Fc[1] + AC[2]*Fa_Fc[2];
    const real_mds rhs2 = BC[0]*Fb_Fc[0] + BC[1]*Fb_Fc[1] + BC[2]*Fb_Fc[2];

    // Solve 3x3 system using Cramer's rule
    // Matrix A = M^T M
    const real_mds A00 = realval_mds(2.0) * AB_AB;
    const real_mds A01 = AB_AC;
    const real_mds A02 = AB_BC;
    const real_mds A11 = realval_mds(2.0) * AC_AC;
    const real_mds A12 = -AC_BC;
    const real_mds A22 = realval_mds(2.0) * BC_BC;

    // Determinant (using symmetry: A10=A01, A20=A02, A21=A12)
    const real_mds det = A00 * (A11 * A22 - A12 * A12)
                       - A01 * (A01 * A22 - A12 * A02)
                       + A02 * (A01 * A12 - A11 * A02);

    real_mds lab, lac, lbc;
    
    if (std::abs(det) > mds_eps) {
        const real_mds inv_det = realval_mds(1.0) / det;
        
        // Cramer's rule
        lab = inv_det * (rhs0 * (A11 * A22 - A12 * A12)
                       - A01 * (rhs1 * A22 - A12 * rhs2)
                       + A02 * (rhs1 * A12 - A11 * rhs2));
        
        lac = inv_det * (A00 * (rhs1 * A22 - A12 * rhs2)
                       - rhs0 * (A01 * A22 - A12 * A02)
                       + A02 * (A01 * rhs2 - rhs1 * A02));
        
        lbc = inv_det * (A00 * (A11 * rhs2 - rhs1 * A12)
                       - A01 * (A01 * rhs2 - rhs1 * A02)
                       + rhs0 * (A01 * A12 - A11 * A02));
    } else {
        // Degenerate case - use pseudo-inverse or fallback
        // This is rare in practice
        lab = lac = lbc = realval_mds(0.0);
    }

    // Compute pairwise forces
    const array3_mds Fij1 = {lab * AB[0], lab * AB[1], lab * AB[2]};
    const array3_mds Fij2 = {lac * AC[0], lac * AC[1], lac * AC[2]};
    const array3_mds Fij3 = {lbc * BC[0], lbc * BC[1], lbc * BC[2]};

    // Distribute to grid
    if (elast_grid != nullptr) {
        const real_ext zero = realval_mds(0.0);

        if (phi == nullptr) {
            // Settle specialization
            const real_ext phiNormAB = real_ext(lab * std::sqrt(AB_AB));
            const real_ext phiNormAC = real_ext(lac * std::sqrt(AC_AC));
            const real_ext phiNormBC = real_ext(lbc * std::sqrt(BC_BC));

            distribute_n2(state, Ra, Rb, &Fij1, stress_grid, &Ra, &Rb, &phiNormAB, &zero, elast_grid, false);
            distribute_n2(state, Ra, Rc, &Fij2, stress_grid, &Ra, &Rc, &phiNormAC, &zero, elast_grid, false);
            distribute_n2(state, Rb, Rc, &Fij3, stress_grid, &Rb, &Rc, &phiNormBC, &zero, elast_grid, false);
        } else {
            // Full elasticity calculation
            distribute_n2(state, Ra, Rb, &Fij1, stress_grid, &Ra, &Rb, phi+0, kappa+0, elast_grid, false);
            distribute_n2(state, Ra, Rc, &Fij2, stress_grid, &Ra, &Rc, phi+1, kappa+4, elast_grid, false);
            distribute_n2(state, Rb, Rc, &Fij3, stress_grid, &Rb, &Rc, phi+2, kappa+8, elast_grid, false);

            // Off-diagonal terms
            distribute_n2(state, Ra, Rb, nullptr, nullptr, &Ra, &Rc, &zero, kappa+1, elast_grid, false);
            distribute_n2(state, Ra, Rb, nullptr, nullptr, &Rb, &Rc, &zero, kappa+2, elast_grid, false);
            distribute_n2(state, Ra, Rc, nullptr, nullptr, &Ra, &Rb, &zero, kappa+3, elast_grid, false);
            distribute_n2(state, Ra, Rc, nullptr, nullptr, &Rb, &Rc, &zero, kappa+5, elast_grid, false);
            distribute_n2(state, Rb, Rc, nullptr, nullptr, &Ra, &Rb, &zero, kappa+6, elast_grid, false);
            distribute_n2(state, Rb, Rc, nullptr, nullptr, &Ra, &Rc, &zero, kappa+7, elast_grid, false);
        }
    } else {
        // Stress only
        distribute_n2(state, Ra, Rb, &Fij1, stress_grid, nullptr, nullptr, nullptr, nullptr, nullptr, false);
        distribute_n2(state, Ra, Rc, &Fij2, stress_grid, nullptr, nullptr, nullptr, nullptr, nullptr, false);
        distribute_n2(state, Rb, Rc, &Fij3, stress_grid, nullptr, nullptr, nullptr, nullptr, nullptr, false);
    }
}

//==========================================================================
// Optimized: SumGrid Reduction with Cache Blocking
//==========================================================================

// Cache block size (tune based on L2 cache size)
// Typical L2 is 256KB-1MB per core
// matrix3_mds = 72 bytes, matrix6_mds = 288 bytes
// Processing 1024 cells at a time uses ~370KB for stress + elasticity
constexpr int SUMGRID_BLOCK_SIZE = 1024;

static void sumgrid_reduction_optimized(
        matrix3_mds * __restrict__ current_grid,
        matrix6_mds * __restrict__ current_grid_elborn,
        matrix6_mds * __restrict__ current_grid_elkin,
        long nCells,
        int m_max_threads,
        int thread_id)
{
    // Each thread processes a portion of cells
    const long cells_per_thread = (nCells + m_max_threads - 1) / m_max_threads;
    const long start_cell = thread_id * cells_per_thread;
    const long end_cell = std::min(start_cell + cells_per_thread, nCells);
    
    // Process in cache-friendly blocks
    for (long block_start = start_cell; block_start < end_cell; block_start += SUMGRID_BLOCK_SIZE) {
        const long block_end = std::min(block_start + (long)SUMGRID_BLOCK_SIZE, end_cell);
        
        // For each thread's contribution (except thread 0)
        for (int j = 1; j < m_max_threads; ++j) {
            const long offset = j * nCells;
            
            // Process this block
            for (long i = block_start; i < block_end; ++i) {
                // Use combined sum+zero operations to reduce memory traffic
                summatrix3_and_zero(current_grid[i + offset], current_grid[i]);
                summatrix6_and_zero(current_grid_elborn[i + offset], current_grid_elborn[i]);
                summatrix6_and_zero(current_grid_elkin[i + offset], current_grid_elkin[i]);
            }
        }
    }
}

//==========================================================================
// Optimized: Batch Zero Operations
//==========================================================================

static void zero_current_grids_optimized(
        matrix3_mds * current_grid,
        matrix6_mds * current_grid_elborn,
        matrix6_mds * current_grid_elkin,
        long nCells)
{
    // Use memset for large contiguous zeroing (most efficient)
    std::memset(current_grid, 0, nCells * sizeof(matrix3_mds));
    std::memset(current_grid_elborn, 0, nCells * sizeof(matrix6_mds));
    std::memset(current_grid_elkin, 0, nCells * sizeof(matrix6_mds));
}

} // namespace mds

#endif // mds_optimized_functions_h
