#include <string.h>
#include <assert.h>
#include <cuda.h>

#include "mds_custress.h"

// some macros to check for errors
#define checkCuda(a) __checkCuda(a, __FILE__, __LINE__)

// convenience function for checking CUDA runtime API results
inline
cudaError_t __checkCuda(cudaError_t result, const char * file, int line)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s in %s, line %d\n", cudaGetErrorString(result), file, line);
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

// constant memory is cached
__constant__ mds::dmatrix c_box;
__constant__ mds::dmatrix c_invbox;
__constant__ mds::barray  c_periodic;
__constant__ mds::darray  c_gridsp;
__constant__ mds::iarray  c_nxyz;

// global memory is not, so minimize access here
mds::batcharrays *d_batch    = nullptr;
mds::dmatrix     *d_sum_grid = nullptr;

// global functions
__device__ static inline void spread_line_source(
        double t1, double t2,
        mds::darray & a,
        mds::darray & b,
        mds::iarray & x,
        mds::dmatrix & stress,
        mds::dmatrix * current_grid)
{
    // attempt to auto-vectorize code
    const double ijks[8][8] = {
      // ijk,  i',  j',  k', ij', ik', jk', ijk'
       { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,},
       {-1.0,-1.0,-1.0, 1.0,-1.0, 1.0, 1.0, 1.0,},
       {-1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0, 1.0,},
       { 1.0, 1.0,-1.0,-1.0,-1.0,-1.0, 1.0, 1.0,},
       {-1.0, 1.0,-1.0,-1.0, 1.0, 1.0,-1.0, 1.0,},
       { 1.0,-1.0, 1.0,-1.0,-1.0, 1.0,-1.0, 1.0,},
       { 1.0,-1.0,-1.0, 1.0, 1.0,-1.0,-1.0, 1.0,},
       {-1.0, 1.0, 1.0, 1.0,-1.0,-1.0,-1.0, 1.0,},
    };

    // scalars used to prepare vectors
    double t12,t22, dt1, dt2, dt3, dt4;
    double axy, axz, ayz, axyz;
    double bxy, bxz, byz, bxyz;
    double lxy, lxz, lyz, lxyz;
    int iip1, iim1, jjp1, jjm1, kkp1, kkm1;

    // vectors and a single coefficient
    double C, invgridsp;
    double D[8], factor[8];

    // work out the parametric time constants
    t12 = t1*t1;
    t22 = t2*t2;
    dt1 = t2 - t1;
    dt2 = t22 - t12;
    dt3 = 4.0*(t22*t2 - t12*t1)/3.0;
    dt4 = t22*t22 - t12*t12;

    // now the position/spatial constants
    invgridsp = 1.0/(c_gridsp[0]*c_gridsp[1]*c_gridsp[2]);
    axy = a[0]*a[1]; axz = a[0]*a[2]; ayz = a[1]*a[2];
    bxy = b[0]*b[1]; bxz = b[0]*b[2]; byz = b[1]*b[2];
    lxy = c_gridsp[0]*c_gridsp[1];
    lxz = c_gridsp[0]*c_gridsp[2];
    lyz = c_gridsp[1]*c_gridsp[2];
    axyz = a[0]*ayz; bxyz = b[0]*byz; lxyz = c_gridsp[0]*lyz;

    // and the index constants
    iip1 = ((x[0] + 1 + c_nxyz[0]) % c_nxyz[0])*c_nxyz[1]*c_nxyz[2];
    jjp1 = ((x[1] + 1 + c_nxyz[1]) % c_nxyz[1])*c_nxyz[2];
    kkp1 = ((x[2] + 1 + c_nxyz[2]) % c_nxyz[2]);
    iim1 = ((x[0] + c_nxyz[0]) % c_nxyz[0])*c_nxyz[1]*c_nxyz[2];
    jjm1 = ((x[1] + c_nxyz[1]) % c_nxyz[1])*c_nxyz[2];
    kkm1 = ((x[2] + c_nxyz[2]) % c_nxyz[2]);

    // the composite constants in terms of i, j, k
    C = 0.125*invgridsp*invgridsp;
    D[0] = 8.0*bxyz*dt1 + 4.0*(a[0]*byz+a[1]*bxz+a[2]*bxy)*dt2
        + 2.0*(b[0]*ayz+b[1]*axz+b[2]*axy)*dt3 + 2.0*axyz*dt4;
    D[1] = c_gridsp[0]*(4.0*byz*dt1 + 2.0*(a[1]*b[2]+a[2]*b[1])*dt2 + ayz*dt3);
    D[2] = c_gridsp[1]*(4.0*bxz*dt1 + 2.0*(a[0]*b[2]+a[2]*b[0])*dt2 + axz*dt3);
    D[3] = c_gridsp[2]*(4.0*bxy*dt1 + 2.0*(a[0]*b[1]+a[1]*b[0])*dt2 + axy*dt3);
    D[4] = lxy*(2.0*b[2]*dt1+a[2]*dt2);
    D[5] = lxz*(2.0*b[1]*dt1+a[1]*dt2);
    D[6] = lyz*(2.0*b[0]*dt1+a[0]*dt2);
    D[7] = lxyz*dt1;

    // prepare the factors
    factor[0] = C*(ijks[0][0]*D[0] + ijks[0][1]*D[1] + ijks[0][2]*D[2] + ijks[0][3]*D[3] + ijks[0][4]*D[4] + ijks[0][5]*D[5] + ijks[0][6]*D[6] + ijks[0][7]*D[7]);
    factor[1] = C*(ijks[1][0]*D[0] + ijks[1][1]*D[1] + ijks[1][2]*D[2] + ijks[1][3]*D[3] + ijks[1][4]*D[4] + ijks[1][5]*D[5] + ijks[1][6]*D[6] + ijks[1][7]*D[7]);
    factor[2] = C*(ijks[2][0]*D[0] + ijks[2][1]*D[1] + ijks[2][2]*D[2] + ijks[2][3]*D[3] + ijks[2][4]*D[4] + ijks[2][5]*D[5] + ijks[2][6]*D[6] + ijks[2][7]*D[7]);
    factor[3] = C*(ijks[3][0]*D[0] + ijks[3][1]*D[1] + ijks[3][2]*D[2] + ijks[3][3]*D[3] + ijks[3][4]*D[4] + ijks[3][5]*D[5] + ijks[3][6]*D[6] + ijks[3][7]*D[7]);
    factor[4] = C*(ijks[4][0]*D[0] + ijks[4][1]*D[1] + ijks[4][2]*D[2] + ijks[4][3]*D[3] + ijks[4][4]*D[4] + ijks[4][5]*D[5] + ijks[4][6]*D[6] + ijks[4][7]*D[7]);
    factor[5] = C*(ijks[5][0]*D[0] + ijks[5][1]*D[1] + ijks[5][2]*D[2] + ijks[5][3]*D[3] + ijks[5][4]*D[4] + ijks[5][5]*D[5] + ijks[5][6]*D[6] + ijks[5][7]*D[7]);
    factor[6] = C*(ijks[6][0]*D[0] + ijks[6][1]*D[1] + ijks[6][2]*D[2] + ijks[6][3]*D[3] + ijks[6][4]*D[4] + ijks[6][5]*D[5] + ijks[6][6]*D[6] + ijks[6][7]*D[7]);
    factor[7] = C*(ijks[7][0]*D[0] + ijks[7][1]*D[1] + ijks[7][2]*D[2] + ijks[7][3]*D[3] + ijks[7][4]*D[4] + ijks[7][5]*D[5] + ijks[7][6]*D[6] + ijks[7][7]*D[7]);

    // perform the sums into the grid
    ssmatm(factor[0], stress, current_grid[iip1 + jjp1 + kkp1]);
    ssmatm(factor[1], stress, current_grid[iip1 + jjp1 + kkm1]);
    ssmatm(factor[2], stress, current_grid[iip1 + jjm1 + kkp1]);
    ssmatm(factor[3], stress, current_grid[iip1 + jjm1 + kkm1]);
    ssmatm(factor[4], stress, current_grid[iim1 + jjp1 + kkp1]);
    ssmatm(factor[5], stress, current_grid[iim1 + jjp1 + kkm1]);
    ssmatm(factor[6], stress, current_grid[iim1 + jjm1 + kkp1]);
    ssmatm(factor[7], stress, current_grid[iim1 + jjm1 + kkm1]);
}

__device__ static inline void diff_array(
        const mds::darray & a,
        const mds::darray & b,
        mds::darray & c)
{
    for (int i = 0; i < mds_ndim; i++)
        c[i] = a[i]-b[i];

    for (int i = 0; i < mds_ndim; i++)
    {
        if (c_periodic[i] == true)
        {
            while (c[i] > 0.5*c_box[i][i])
                c[i] -= c_box[i][i];
            while (c[i] <= -0.5*c_box[i][i])
                c[i] += c_box[i][i];
        }
    }
}

__device__ static inline void grid_coord(const mds::darray & pt, int & i, int & j, int & k )
{
    i = c_nxyz[0] * pt[0] * c_invbox[0][0];
    j = c_nxyz[1] * pt[1] * c_invbox[1][1];
    k = c_nxyz[2] * pt[2] * c_invbox[2][2];

    if(pt[0] < 0) i -= 1;
    if(pt[1] < 0) j -= 1;
    if(pt[2] < 0) k -= 1;
}

__device__ static inline void BatchPairInteraction(
        mds::darray & xi,
        mds::darray & xj,
        mds::darray & F,
        mds::dmatrix * current_grid)
{
    double oldt;
    int cmp0x,cmp1x,cmp2x,iX;
    mds::darray t, d_cgrid, diff;

    mds::iarray i1; //grid cell corresponding to particle I (A)
    mds::iarray i2; //grid cell corresponding to particle J (B)
    mds::iarray x;  //cell during spreding
    mds::iarray xn; //next cell during spreading
    mds::iarray c;  //director
    
    mds::dmatrix stress;

    int timesInLoop; // avoid getting stuck in the while loop

    //------------------------------------------------------------------------------------
    // Calculate the stress tensor
    diff_array(xj, xi, diff);
    stress[0][0] = F[0]*diff[0];
    stress[1][0] = F[1]*diff[0];
    stress[2][0] = F[2]*diff[0];
    stress[0][1] = F[0]*diff[1];
    stress[1][1] = F[1]*diff[1];
    stress[2][1] = F[2]*diff[1];
    stress[0][2] = F[0]*diff[2];
    stress[1][2] = F[1]*diff[2];
    stress[2][2] = F[2]*diff[2];

    //------------------------------------------------------------------------------------
    // Distribute the stress

    // calculate the grid coordinates (no pbc) for the extreme points
    grid_coord(xi, i1[0], i1[1], i1[2]);
    grid_coord(xj, i2[0], i2[1], i2[2]);

    // d_cgrid = vector from the center of the present cell to the initial point
    d_cgrid[0] = xi[0]-(i1[0]+0.5)*c_gridsp[0];
    d_cgrid[1] = xi[1]-(i1[1]+0.5)*c_gridsp[1];
    d_cgrid[2] = xi[2]-(i1[2]+0.5)*c_gridsp[2];
    
    // First cross point with aplane (if there is at least one i.e. c[i] != 0)
    x[0] = i1[0];
    x[1] = i1[1];
    x[2] = i1[2];

    // c is a vector that guide the advance in each coordinate (+1 if it has to advance in this coordinate, -1 if it has to go back or 0 if it has to do nothing)
    c[0] = (i2[0]>i1[0])-(i1[0]>i2[0]);
    c[1] = (i2[1]>i1[1])-(i1[1]>i2[1]);
    c[2] = (i2[2]>i1[2])-(i1[2]>i2[2]);
    
    // label of the next cell is 1 step further than the previous in this direction
    xn[0] = i1[0]+(c[0]+1)/2;
    xn[1] = i1[1]+(c[1]+1)/2;
    xn[2] = i1[2]+(c[2]+1)/2;

    // parametric time of crossing
    t[0] = (xi[0]-xn[0] * c_gridsp[0])/(xi[0]-xj[0]);
    t[1] = (xi[1]-xn[1] * c_gridsp[1])/(xi[1]-xj[1]);
    t[2] = (xi[2]-xn[2] * c_gridsp[2])/(xi[2]-xj[2]);
        
    // this sets the time larger than 1 if there is no crossing
    t[0] = (c[0] == 0.0)*1.1 + (c[0] != 0.0)*t[0];
    t[1] = (c[1] == 0.0)*1.1 + (c[1] != 0.0)*t[1];
    t[2] = (c[2] == 0.0)*1.1 + (c[2] != 0.0)*t[2];
    
    // track previous time of crossing and check that sum is complete (?)
    oldt      = 0.0; 

    // while we don't reach the last point...
    timesInLoop = 0;
    while( (c[0]*x[0]<c[0]*i2[0])||(c[1]*x[1]<c[1]*i2[1])||(c[2]*x[2]<c[2]*i2[2]) )
    {
        // iterate loop
        timesInLoop ++;

        // figure out index
        cmp0x = ((t[0]<t[1]+mds_eps) + (t[0]<t[2]+mds_eps))/2;
        cmp1x = ((t[1]<t[0]+mds_eps) + (t[1]<t[2]+mds_eps))/2;
        cmp2x = ((t[2]<t[0]+mds_eps) + (t[2]<t[1]+mds_eps))/2;
        iX = 0*cmp0x+1*cmp1x+2*cmp2x;

        // distribute the contribution
        spread_line_source(oldt,t[iX],diff,d_cgrid,x,stress,current_grid);

        // move to next cross point
        d_cgrid[iX] -= c[iX] * c_gridsp[iX];
        oldt = t[iX];
        
        x[iX] += c[iX];
        xn[iX] += c[iX];

        // Next cross point:
        t[iX] = (xi[iX]-xn[iX] * c_gridsp[iX])/(xi[iX]-xj[iX]);

        if(iX > 2 || timesInLoop > 10000000) //To avoid infinite loops
            return;
    }

    // Distribute the last contribution
    spread_line_source(oldt,1,diff,d_cgrid,x,stress,current_grid);
}

// public functions
void custress_init(long ncells, mds::barray periodic, mds::iarray nxyz)
{
    // initialize device
    checkCuda(cudaSetDevice(0));

    // clear it
    custress_clear();

    // allocate global memory
    checkCuda(cudaMalloc((void**)&d_sum_grid,ncells*sizeof(mds::dmatrix)));
    checkCuda(cudaMalloc((void**)&d_batch,sizeof(mds::batcharrays)));
    
    // initialize period boundary conditions
    checkCuda(cudaMemcpyToSymbol(&c_periodic, &periodic, sizeof(mds::barray)));
    checkCuda(cudaMemcpyToSymbol(&c_nxyz,     &nxyz,     sizeof(mds::iarray)));

    // initialize global memory
    checkCuda(cudaMemset(&d_sum_grid, 0, ncells*sizeof(mds::dmatrix)));
    checkCuda(cudaMemset(&d_batch,    0, sizeof(mds::batcharrays)));
}

// public functions
void custress_clear()
{
    // allocate memory
    if (d_sum_grid != nullptr) cudaFree(d_sum_grid);
    if (d_batch != nullptr) cudaFree(d_batch);
    
    // set to null
    d_sum_grid = nullptr;
    d_batch = nullptr;
}

void custress_update_box_spacings(mds::dmatrix box, mds::dmatrix invbox, mds::darray gridsp)
{
    // initialize constant memory variables
    checkCuda(cudaMemcpyToSymbol(&c_box,       &box, sizeof(mds::dmatrix)));
    checkCuda(cudaMemcpyToSymbol(&c_invbox, &invbox, sizeof(mds::dmatrix)));
    checkCuda(cudaMemcpyToSymbol(&c_gridsp, &gridsp, sizeof(mds::darray)));
}

void custress_process_batch(int batch_index, mds::batcharrays * h_batch)
{
}
