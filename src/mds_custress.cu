#include <string.h>
#include <assert.h>
#include <cuda.h>
#include <stdio.h>

#include "mds_custress.h"
#include "mds_basicops.h"

// some macros to check for errors
#define checkCuda(a) __checkCuda(a, __FILE__, __LINE__)

// need atomic add for FP64 (only for older hardware):
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// cuda specific ssmatm
#define cu_ssmatm(a,b,c) \
atomicAdd(&c[0][0],a*b[0][0]); \
atomicAdd(&c[0][1],a*b[0][1]); \
atomicAdd(&c[0][2],a*b[0][2]); \
atomicAdd(&c[1][0],a*b[1][0]); \
atomicAdd(&c[1][1],a*b[1][1]); \
atomicAdd(&c[1][2],a*b[1][2]); \
atomicAdd(&c[2][0],a*b[2][0]); \
atomicAdd(&c[2][1],a*b[2][1]); \
atomicAdd(&c[2][2],a*b[2][2])

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

// toy with single precision
#define CUSTRESS_SINGLE
#ifndef CUSTRESS_SINGLE
typedef double real_t;
#define realval(a) (a)
#define cu_eps realval(mds_eps)
#else
typedef float real_t;
#define realval(a) (a ## f)
#define cu_eps 1.0e-6f
#endif//CUSTRESS_SINGLE

// typedefs for custress 
typedef int32_t   int_t;
typedef uint32_t uint_t;

// copies of mdstresslib typedefs
typedef bool cu_barray[4];
typedef int_t cu_iarray[3];
typedef real_t cu_darray[3];
typedef real_t cu_dmatrix[3][3];

#define cu_batchsize 16384
#define cu_threads_per_block 32
typedef struct {
    cu_darray Ri[cu_batchsize];
    cu_darray Rj[cu_batchsize];
    cu_darray Fij[cu_batchsize];
} batcharrays_t;

// device constant memory is cached
__constant__ cu_barray  c_periodic;
__constant__ cu_iarray  c_nxyz;
__constant__ cu_dmatrix c_box;
__constant__ cu_dmatrix c_invbox;
__constant__ cu_darray  c_gridsp;

// device global memory is not, so minimize access here
batcharrays_t *d_batch    = nullptr;
cu_dmatrix    *d_sum_grid = nullptr;

// host memory
uint_t h_bindex = 0;
uint_t h_ncells = 0;
batcharrays_t *h_batch    = nullptr;
cu_dmatrix    *h_sum_grid = nullptr;

// global functions
__device__ static inline void spread_line_source(
        real_t t1, real_t t2,
        const cu_darray & a,
        const cu_darray & b,
        const cu_iarray & x,
        const cu_dmatrix & stress,
        cu_dmatrix * current_grid)
{
    // attempt to auto-vectorize code
    const real_t ijks[8][8] = {
      // ijk,  i',  j',  k', ij', ik', jk', ijk'
       {realval( 1.0),realval( 1.0),realval( 1.0),realval( 1.0),realval( 1.0),realval( 1.0),realval( 1.0),realval( 1.0),},
       {realval(-1.0),realval(-1.0),realval(-1.0),realval( 1.0),realval(-1.0),realval( 1.0),realval( 1.0),realval( 1.0),},
       {realval(-1.0),realval(-1.0),realval( 1.0),realval(-1.0),realval( 1.0),realval(-1.0),realval( 1.0),realval( 1.0),},
       {realval( 1.0),realval( 1.0),realval(-1.0),realval(-1.0),realval(-1.0),realval(-1.0),realval( 1.0),realval( 1.0),},
       {realval(-1.0),realval( 1.0),realval(-1.0),realval(-1.0),realval( 1.0),realval( 1.0),realval(-1.0),realval( 1.0),},
       {realval( 1.0),realval(-1.0),realval( 1.0),realval(-1.0),realval(-1.0),realval( 1.0),realval(-1.0),realval( 1.0),},
       {realval( 1.0),realval(-1.0),realval(-1.0),realval( 1.0),realval( 1.0),realval(-1.0),realval(-1.0),realval( 1.0),},
       {realval(-1.0),realval( 1.0),realval( 1.0),realval( 1.0),realval(-1.0),realval(-1.0),realval(-1.0),realval( 1.0),},
    };

    // scalars used to prepare vectors
    real_t t12,t22, dt1, dt2, dt3, dt4;
    real_t axy, axz, ayz, axyz;
    real_t bxy, bxz, byz, bxyz;
    real_t lxy, lxz, lyz, lxyz;
    int_t iip1, iim1, jjp1, jjm1, kkp1, kkm1;

    // vectors and a single coefficient
    real_t C, invgridsp;
    real_t D[8], factor[8];

    // work out the parametric time constants
    t12 = t1*t1;
    t22 = t2*t2;
    dt1 = t2 - t1;
    dt2 = t22 - t12;
    dt3 = realval(4.0)*(t22*t2 - t12*t1)/realval(3.0);
    dt4 = t22*t22 - t12*t12;
    
    //printf("c_gridsp 0: %f, 1: %f,2: %f\n",c_gridsp[0],c_gridsp[1],c_gridsp[2]);

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
    
    //printf("invgridsp 0: %f\n",invgridsp);

    // the composite constants in terms of i, j, k
    C = realval(0.125)*invgridsp*invgridsp;
    //printf("C 0: %f\n",C);
    D[0] = realval(8.0)*bxyz*dt1 + realval(4.0)*(a[0]*byz+a[1]*bxz+a[2]*bxy)*dt2
        + realval(2.0)*(b[0]*ayz+b[1]*axz+b[2]*axy)*dt3 + realval(2.0)*axyz*dt4;
    D[1] = c_gridsp[0]*(realval(4.0)*byz*dt1 + realval(2.0)*(a[1]*b[2]+a[2]*b[1])*dt2 + ayz*dt3);
    D[2] = c_gridsp[1]*(realval(4.0)*bxz*dt1 + realval(2.0)*(a[0]*b[2]+a[2]*b[0])*dt2 + axz*dt3);
    D[3] = c_gridsp[2]*(realval(4.0)*bxy*dt1 + realval(2.0)*(a[0]*b[1]+a[1]*b[0])*dt2 + axy*dt3);
    D[4] = lxy*(realval(2.0)*b[2]*dt1+a[2]*dt2);
    D[5] = lxz*(realval(2.0)*b[1]*dt1+a[1]*dt2);
    D[6] = lyz*(realval(2.0)*b[0]*dt1+a[0]*dt2);
    D[7] = lxyz*dt1;
    //printf("D 0: %f, 1: %f,2: %f, 3: %f,4: %f, 5: %f,6: %f, 7: %f\n",D[0],D[1],D[2],D[3],D[4],D[5],D[6],D[7]);

    // prepare the factors
    factor[0] = C*(ijks[0][0]*D[0] + ijks[0][1]*D[1] + ijks[0][2]*D[2] + ijks[0][3]*D[3] + ijks[0][4]*D[4] + ijks[0][5]*D[5] + ijks[0][6]*D[6] + ijks[0][7]*D[7]);
    factor[1] = C*(ijks[1][0]*D[0] + ijks[1][1]*D[1] + ijks[1][2]*D[2] + ijks[1][3]*D[3] + ijks[1][4]*D[4] + ijks[1][5]*D[5] + ijks[1][6]*D[6] + ijks[1][7]*D[7]);
    factor[2] = C*(ijks[2][0]*D[0] + ijks[2][1]*D[1] + ijks[2][2]*D[2] + ijks[2][3]*D[3] + ijks[2][4]*D[4] + ijks[2][5]*D[5] + ijks[2][6]*D[6] + ijks[2][7]*D[7]);
    factor[3] = C*(ijks[3][0]*D[0] + ijks[3][1]*D[1] + ijks[3][2]*D[2] + ijks[3][3]*D[3] + ijks[3][4]*D[4] + ijks[3][5]*D[5] + ijks[3][6]*D[6] + ijks[3][7]*D[7]);
    factor[4] = C*(ijks[4][0]*D[0] + ijks[4][1]*D[1] + ijks[4][2]*D[2] + ijks[4][3]*D[3] + ijks[4][4]*D[4] + ijks[4][5]*D[5] + ijks[4][6]*D[6] + ijks[4][7]*D[7]);
    factor[5] = C*(ijks[5][0]*D[0] + ijks[5][1]*D[1] + ijks[5][2]*D[2] + ijks[5][3]*D[3] + ijks[5][4]*D[4] + ijks[5][5]*D[5] + ijks[5][6]*D[6] + ijks[5][7]*D[7]);
    factor[6] = C*(ijks[6][0]*D[0] + ijks[6][1]*D[1] + ijks[6][2]*D[2] + ijks[6][3]*D[3] + ijks[6][4]*D[4] + ijks[6][5]*D[5] + ijks[6][6]*D[6] + ijks[6][7]*D[7]);
    factor[7] = C*(ijks[7][0]*D[0] + ijks[7][1]*D[1] + ijks[7][2]*D[2] + ijks[7][3]*D[3] + ijks[7][4]*D[4] + ijks[7][5]*D[5] + ijks[7][6]*D[6] + ijks[7][7]*D[7]);

    //printf("FACTOR 0: %f, 1: %f,2: %f, 3: %f,4: %f, 5: %f,6: %f, 7: %f\n",factor[0],factor[1],factor[2],factor[3],factor[4],factor[5],factor[6],factor[7]);

    // perform the sums into the grid
    cu_ssmatm(factor[0], stress, current_grid[iip1 + jjp1 + kkp1]);
    cu_ssmatm(factor[1], stress, current_grid[iip1 + jjp1 + kkm1]);
    cu_ssmatm(factor[2], stress, current_grid[iip1 + jjm1 + kkp1]);
    cu_ssmatm(factor[3], stress, current_grid[iip1 + jjm1 + kkm1]);
    cu_ssmatm(factor[4], stress, current_grid[iim1 + jjp1 + kkp1]);
    cu_ssmatm(factor[5], stress, current_grid[iim1 + jjp1 + kkm1]);
    cu_ssmatm(factor[6], stress, current_grid[iim1 + jjm1 + kkp1]);
    cu_ssmatm(factor[7], stress, current_grid[iim1 + jjm1 + kkm1]);
}

__device__ static inline void diff_array(
        const cu_darray a,
        const cu_darray b,
        cu_darray c)
{
    for (int_t i = 0; i < mds_ndim; i++)
        c[i] = a[i]-b[i];

    if (c_periodic[3] == true)
    {
        for (int_t i = 0; i < mds_ndim; i++)
        {
            if (c_periodic[i] == true)
            {
                while (c[i] >   realval(0.5)*c_box[i][i])
                    c[i] -= c_box[i][i];
                while (c[i] <= -realval(0.5)*c_box[i][i])
                    c[i] += c_box[i][i];
            }
        }
    }
}

__device__ static inline void grid_coord(
        const cu_darray pt,
        int_t & i, int_t & j, int_t & k )
{
    i = c_nxyz[0] * pt[0] * c_invbox[0][0];
    j = c_nxyz[1] * pt[1] * c_invbox[1][1];
    k = c_nxyz[2] * pt[2] * c_invbox[2][2];

    if(pt[0] < 0) i -= 1;
    if(pt[1] < 0) j -= 1;
    if(pt[2] < 0) k -= 1;
}

__device__ static inline void BatchPairInteraction(
        const cu_darray xi,
        const cu_darray xj,
        const cu_darray F,
        cu_dmatrix * current_grid)
{
    real_t oldt;
    int_t cmp0x,cmp1x,cmp2x,iX;

    cu_darray t, d_cgrid, diff;

    cu_iarray i1; //grid cell corresponding to particle I (A)
    cu_iarray i2; //grid cell corresponding to particle J (B)
    cu_iarray x;  //cell during spreading
    cu_iarray xn; //next cell during spreading
    cu_iarray c;  //director
    
    cu_dmatrix stress;

    uint_t timesInLoop; // avoid getting stuck in the while loop

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
    t[0] = (c[0] == realval(0.0))*realval(1.1) + (c[0] != realval(0.0))*t[0];
    t[1] = (c[1] == realval(0.0))*realval(1.1) + (c[1] != realval(0.0))*t[1];
    t[2] = (c[2] == realval(0.0))*realval(1.1) + (c[2] != realval(0.0))*t[2];
    
    // track previous time of crossing and check that sum is complete (?)
    oldt = realval(0.0); 

    // while we don't reach the last point...
    timesInLoop = 0;
    while( (c[0]*x[0]<c[0]*i2[0])||(c[1]*x[1]<c[1]*i2[1])||(c[2]*x[2]<c[2]*i2[2]) )
    {
        // iterate loop
        timesInLoop ++;

        // figure out index
        cmp0x = ((t[0]<t[1]+cu_eps) + (t[0]<t[2]+cu_eps))/2;
        cmp1x = ((t[1]<t[0]+cu_eps) + (t[1]<t[2]+cu_eps))/2;
        cmp2x = ((t[2]<t[0]+cu_eps) + (t[2]<t[1]+cu_eps))/2;
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
void custress_init(size_t ncells, int nx, int ny, int nz)
{
    // initialize device
    checkCuda(cudaSetDevice(0));

    // clear it
    custress_clear();

    // allocate global memory
    checkCuda(cudaMalloc((void**)&d_sum_grid,ncells*sizeof(cu_dmatrix)));
    checkCuda(cudaMalloc((void**)&d_batch,sizeof(batcharrays_t)));

    // allocate host memory
    h_batch = new batcharrays_t;
    h_sum_grid = new cu_dmatrix[ncells];
    
    cu_iarray cu_nxyz = {(int_t)nx, (int_t)ny, (int_t)nz};
    checkCuda(cudaMemcpyToSymbol(c_nxyz, cu_nxyz, sizeof(mds::iarray)));

    // initialize global memory
    checkCuda(cudaMemset(d_sum_grid, 0, ncells*sizeof(cu_dmatrix)));
    checkCuda(cudaMemset(d_batch,    0, sizeof(batcharrays_t)));

    // initialize host memory
    h_ncells = (uint_t)ncells;
    h_bindex = 0;
    memset(h_sum_grid, 0, ncells*sizeof(cu_dmatrix));
    memset(h_batch,    0, sizeof(cu_dmatrix));
}

void custress_set_periodic(bool x, bool y, bool z, bool enforce)
{
    // initialize period boundary conditions
    cu_barray cu_periodic = { x, y, z ,enforce };
    checkCuda(cudaMemcpyToSymbol(c_periodic, cu_periodic, sizeof(cu_barray)));
}

// public functions
void custress_clear()
{
    // free global memory
    if (d_sum_grid != nullptr) cudaFree(d_sum_grid);
    if (d_batch != nullptr) cudaFree(d_batch);

    // free host memory
    if (h_sum_grid != nullptr) free(h_sum_grid);
    if (h_batch != nullptr) free(h_batch);
    
    // set to null
    d_sum_grid = nullptr;
    d_batch = nullptr;
    h_sum_grid = nullptr;
    h_batch = nullptr;
}

void custress_update_box_spacings(const mds::dmatrix box, const mds::dmatrix invbox, const mds::darray gridsp)
{
    // convert to custress types
    cu_dmatrix cu_box = {
        {(real_t)box[0][0], (real_t)box[0][1], (real_t)box[0][2]},
        {(real_t)box[1][0], (real_t)box[1][1], (real_t)box[1][2]},
        {(real_t)box[2][0], (real_t)box[2][1], (real_t)box[2][2]},
    };
    cu_dmatrix cu_invbox = {
        {(real_t)invbox[0][0], (real_t)invbox[0][1], (real_t)invbox[0][2]},
        {(real_t)invbox[1][0], (real_t)invbox[1][1], (real_t)invbox[1][2]},
        {(real_t)invbox[2][0], (real_t)invbox[2][1], (real_t)invbox[2][2]},
    };
    cu_darray cu_gridsp = {
        (real_t)gridsp[0], (real_t)gridsp[1], (real_t)gridsp[2],
    };

    // initialize constant memory variables
    checkCuda(cudaMemcpyToSymbol(c_box,    cu_box,    sizeof(cu_dmatrix)));
    checkCuda(cudaMemcpyToSymbol(c_invbox, cu_invbox, sizeof(cu_dmatrix)));
    checkCuda(cudaMemcpyToSymbol(c_gridsp, cu_gridsp, sizeof(cu_darray)));
}

// gpu kernel
__global__ static void process_batch(uint_t batch_index, const batcharrays_t * __restrict__ batch, cu_dmatrix * current_grid)
{
    auto index = blockIdx.x*cu_threads_per_block+threadIdx.x;

    // guard execution
    if (index < batch_index)
        BatchPairInteraction(batch->Ri[index], batch->Rj[index], batch->Fij[index], current_grid);
}
/*__global__ static void process_batch(uint_t batch_index, const batcharrays_t * __restrict__ batch, cu_dmatrix * current_grid)
{
    BatchPairInteraction(batch->Ri[batch_index], batch->Rj[batch_index], batch->Fij[batch_index], current_grid);
}*/

// launches GPU kernel
void custress_distribute_pair_interaction(const mds::darray xi, const mds::darray xj, const mds::darray Fij)
{
    // store for later processing
    if (h_bindex < cu_batchsize)
    {
        h_batch->Ri[h_bindex][0]  = (real_t)xi[0];
        h_batch->Ri[h_bindex][1]  = (real_t)xi[1];
        h_batch->Ri[h_bindex][2]  = (real_t)xi[2];
        h_batch->Rj[h_bindex][0]  = (real_t)xj[0];
        h_batch->Rj[h_bindex][1]  = (real_t)xj[1];
        h_batch->Rj[h_bindex][2]  = (real_t)xj[2];
        h_batch->Fij[h_bindex][0] = (real_t)Fij[0];
        h_batch->Fij[h_bindex][1] = (real_t)Fij[1];
        h_batch->Fij[h_bindex][2] = (real_t)Fij[2];
        h_bindex += 1;
    }

    // if bindex is cu_batchsize we process
    if (h_bindex == cu_batchsize)
    {
        // transfer the grid to host
        checkCuda(cudaMemcpyAsync(
                d_batch,
                h_batch,
                sizeof(batcharrays_t),
                cudaMemcpyHostToDevice) );

        // execute with a single element processed by each streaming multiprocessor
        process_batch<<<cu_batchsize/cu_threads_per_block,cu_threads_per_block>>>(h_bindex, d_batch, d_sum_grid);

        h_bindex = 0;
    }
}

void custress_sum_grid(mds::dmatrix * current_grid)
{
    // finish off any remaining pairs
    if (h_bindex > 0)
    {
        // transfer the batch to device
        checkCuda(cudaMemcpyAsync(
                d_batch,
                h_batch,
                sizeof(batcharrays_t),
                cudaMemcpyHostToDevice) );

        // execute with a single element processed by each streaming multiprocessor
        process_batch<<<cu_batchsize/cu_threads_per_block,cu_threads_per_block>>>(h_bindex, d_batch, d_sum_grid);

        h_bindex = 0;
    }
    
    // transfer the grid to host
    checkCuda(cudaMemcpy(
            h_sum_grid,
            d_sum_grid,
            h_ncells*sizeof(cu_dmatrix),
            cudaMemcpyDeviceToHost) );
    checkCuda(cudaDeviceSynchronize());

    // sum into mdstress current_grid
    for (uint_t i = 0; i < h_ncells; ++i)
    {
        current_grid[i][0][0] += h_sum_grid[i][0][0];
        current_grid[i][0][1] += h_sum_grid[i][0][1];
        current_grid[i][0][2] += h_sum_grid[i][0][2];
        current_grid[i][1][0] += h_sum_grid[i][1][0];
        current_grid[i][1][1] += h_sum_grid[i][1][1];
        current_grid[i][1][2] += h_sum_grid[i][1][2];
        current_grid[i][2][0] += h_sum_grid[i][2][0];
        current_grid[i][2][1] += h_sum_grid[i][2][1];
        current_grid[i][2][2] += h_sum_grid[i][2][2];
    }
}
