#include <string.h>
#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <mutex>
#include <iostream>

#include "mds_custress.h"
#include "mds_basicops.h"

// some macros to check for errors
#define checkCuda(a) __checkCuda(a, __FILE__, __LINE__)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                                   __longlong_as_double(assumed)));
        } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

#define SINGLE_PRECISION
#ifdef SINGLE_PRECISION
typedef double double_t;
#define doubleval(a) (a)
typedef float single_t;
#define singleval(a) (a ## f)
#else
typedef double double_t;
#define doubleval(a) (a)
typedef double single_t;
#define singleval(a) (a)
#endif

// cuda specific ssmatm
#define cu_ssmatm(a,b,c) \
atomicAdd(&c[0][0],(single_t)(a*b[0][0])); \
atomicAdd(&c[0][1],(single_t)(a*b[0][1])); \
atomicAdd(&c[0][2],(single_t)(a*b[0][2])); \
atomicAdd(&c[1][0],(single_t)(a*b[1][0])); \
atomicAdd(&c[1][1],(single_t)(a*b[1][1])); \
atomicAdd(&c[1][2],(single_t)(a*b[1][2])); \
atomicAdd(&c[2][0],(single_t)(a*b[2][0])); \
atomicAdd(&c[2][1],(single_t)(a*b[2][1])); \
atomicAdd(&c[2][2],(single_t)(a*b[2][2]))

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

#define cu_eps singleval(1.0e-10)

// typedefs for custress 
typedef int32_t   int_t;
typedef uint32_t uint_t;

// copies of mdstresslib typedefs
typedef bool cu_barray[4];
typedef int_t cu_iarray[3];
typedef single_t cu_sarray[3];
typedef single_t cu_smatrix[3][3];
typedef double_t cu_darray[3];
typedef double_t cu_dmatrix[3][3];

#define cu_batches 16
#define cu_batchsize (2*262144)
#define cu_threads_per_block 64
typedef struct {
    cu_sarray Ri[cu_batchsize];
    cu_sarray Rj[cu_batchsize];
    cu_sarray Fij[cu_batchsize];
} batcharrays_t;

// device constant memory is cached
__constant__ cu_barray  c_periodic;
__constant__ cu_iarray  c_nxyz;
__constant__ cu_smatrix c_box;
__constant__ cu_smatrix c_invbox;
__constant__ single_t   c_gridsp[8];

// device global memory is not, so minimize access here
batcharrays_t *d_batch    = nullptr;
cu_smatrix    *d_sum_grid = nullptr;

// host memory
uint_t h_ncells = 0;
double_t h_length_max[cu_batches] = { 0 };
uint_t h_bindex[cu_batches] = { 0 };
batcharrays_t *h_batch    = nullptr;
cu_smatrix    *h_sum_grid = nullptr;

// mutex for threads
std::mutex h_mutex_kernel[cu_batches];
cudaEvent_t h_mem_event[cu_batches] = { nullptr };

// global functions
__device__ static inline void spread_line_source(
        single_t t1, single_t t2,
        const cu_sarray & a,
        const cu_sarray & b,
        const cu_iarray & x,
        const cu_smatrix & stress,
        cu_smatrix * current_grid)
{
    // scalars used to prepare vectors
    single_t t12,t22, dt1, dt2, dt3, dt4;
    single_t axy, axz, ayz, axyz;
    single_t bxy, bxz, byz, bxyz;
    int_t iip1, iim1, jjp1, jjm1, kkp1, kkm1;

    // vectors and a single coefficient
    single_t D[8];
    single_t factor[8];

    // work out the parametric time constants
    t12 = t1*t1;
    t22 = t2*t2;
    dt1 = t2 - t1;
    dt2 = t22 - t12;
    dt3 = singleval(4.0)*(t22*t2 - t12*t1)/singleval(3.0);
    dt4 = t22*t22 - t12*t12;
    
    // now the position/spatial constants
    axy = a[0]*a[1]; axz = a[0]*a[2]; ayz = a[1]*a[2];
    bxy = b[0]*b[1]; bxz = b[0]*b[2]; byz = b[1]*b[2];
    axyz = a[0]*ayz; bxyz = b[0]*byz;

    // and the index constants
    iip1 = ((x[0] + 1 + c_nxyz[0]) % c_nxyz[0])*c_nxyz[1]*c_nxyz[2];
    jjp1 = ((x[1] + 1 + c_nxyz[1]) % c_nxyz[1])*c_nxyz[2];
    kkp1 = ((x[2] + 1 + c_nxyz[2]) % c_nxyz[2]);
    iim1 = ((x[0] + c_nxyz[0]) % c_nxyz[0])*c_nxyz[1]*c_nxyz[2];
    jjm1 = ((x[1] + c_nxyz[1]) % c_nxyz[1])*c_nxyz[2];
    kkm1 = ((x[2] + c_nxyz[2]) % c_nxyz[2]);
    
    // the composite constants in terms of i, j, k
    D[0] = singleval(8.0)*bxyz*dt1 + singleval(4.0)*(a[0]*byz+a[1]*bxz+a[2]*bxy)*dt2
        + singleval(2.0)*(b[0]*ayz+b[1]*axz+b[2]*axy)*dt3 + singleval(2.0)*axyz*dt4;
    D[1] = c_gridsp[0]*(singleval(4.0)*byz*dt1 + singleval(2.0)*(a[1]*b[2]+a[2]*b[1])*dt2 + ayz*dt3);
    D[2] = c_gridsp[1]*(singleval(4.0)*bxz*dt1 + singleval(2.0)*(a[0]*b[2]+a[2]*b[0])*dt2 + axz*dt3);
    D[3] = c_gridsp[2]*(singleval(4.0)*bxy*dt1 + singleval(2.0)*(a[0]*b[1]+a[1]*b[0])*dt2 + axy*dt3);
    D[4] = c_gridsp[3]*(singleval(2.0)*b[2]*dt1+a[2]*dt2);
    D[5] = c_gridsp[4]*(singleval(2.0)*b[1]*dt1+a[1]*dt2);
    D[6] = c_gridsp[5]*(singleval(2.0)*b[0]*dt1+a[0]*dt2);
    D[7] = c_gridsp[6]*dt1;

    // prepare the factors
    factor[0] = c_gridsp[7]*( D[0] + D[1] + D[2] + D[3] + D[4] + D[5] + D[6] + D[7]);
    factor[1] = c_gridsp[7]*(-D[0] - D[1] - D[2] + D[3] - D[4] + D[5] + D[6] + D[7]);
    factor[2] = c_gridsp[7]*(-D[0] - D[1] + D[2] - D[3] + D[4] - D[5] + D[6] + D[7]);
    factor[3] = c_gridsp[7]*( D[0] + D[1] - D[2] - D[3] - D[4] - D[5] + D[6] + D[7]);
    factor[4] = c_gridsp[7]*(-D[0] + D[1] - D[2] - D[3] + D[4] + D[5] - D[6] + D[7]);
    factor[5] = c_gridsp[7]*( D[0] - D[1] + D[2] - D[3] - D[4] + D[5] - D[6] + D[7]);
    factor[6] = c_gridsp[7]*( D[0] - D[1] - D[2] + D[3] + D[4] - D[5] - D[6] + D[7]);
    factor[7] = c_gridsp[7]*(-D[0] + D[1] + D[2] + D[3] - D[4] - D[5] - D[6] + D[7]);

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
        const cu_sarray a,
        const cu_sarray b,
        cu_sarray c)
{
    c[0] = a[0]-b[0];
    c[1] = a[1]-b[1];
    c[2] = a[2]-b[2];

    if (c_periodic[3] == true)
    {
        c[0] -= (c_periodic[0] == true && c[0] >   doubleval(0.5)*c_box[0][0]) ? c_box[0][0] : singleval(0.0);
        c[0] += (c_periodic[0] == true && c[0] <= -doubleval(0.5)*c_box[0][0]) ? c_box[0][0] : singleval(0.0);
        c[1] -= (c_periodic[1] == true && c[1] >   doubleval(0.5)*c_box[1][1]) ? c_box[1][1] : singleval(0.0);
        c[1] += (c_periodic[1] == true && c[1] <= -doubleval(0.5)*c_box[1][1]) ? c_box[1][1] : singleval(0.0);
        c[2] -= (c_periodic[2] == true && c[2] >   doubleval(0.5)*c_box[2][2]) ? c_box[2][2] : singleval(0.0);
        c[2] += (c_periodic[2] == true && c[2] <= -doubleval(0.5)*c_box[2][2]) ? c_box[2][2] : singleval(0.0);
    }
}

__device__ static inline void grid_coord(
        const cu_sarray pt,
        int_t & i, int_t & j, int_t & k )
{
    i = c_nxyz[0] * pt[0] * c_invbox[0][0] - (pt[0] < doubleval(0.0));
    j = c_nxyz[1] * pt[1] * c_invbox[1][1] - (pt[1] < doubleval(0.0));
    k = c_nxyz[2] * pt[2] * c_invbox[2][2] - (pt[2] < doubleval(0.0));
}

__device__ static inline void BatchPairInteraction(
        const cu_sarray xi,
        const cu_sarray xj,
        const cu_sarray F,
        cu_smatrix * current_grid)
{
    double_t oldt;
    int_t cmp0x,cmp1x,cmp2x,iX;

    cu_darray t;
    cu_sarray d_cgrid;
    cu_sarray diff;

    cu_iarray i1; //grid cell corresponding to particle I (A)
    cu_iarray i2; //grid cell corresponding to particle J (B)
    cu_iarray x;  //cell during spreading
    cu_iarray xn; //next cell during spreading
    cu_iarray c;  //director
    
    cu_smatrix stress;

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
    d_cgrid[0] = xi[0]-(i1[0]+singleval(0.5))*c_gridsp[0];
    d_cgrid[1] = xi[1]-(i1[1]+singleval(0.5))*c_gridsp[1];
    d_cgrid[2] = xi[2]-(i1[2]+singleval(0.5))*c_gridsp[2];
    
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
    t[0] = (xi[0]-xn[0] * (double_t)c_gridsp[0])/(xi[0]-xj[0]);
    t[1] = (xi[1]-xn[1] * (double_t)c_gridsp[1])/(xi[1]-xj[1]);
    t[2] = (xi[2]-xn[2] * (double_t)c_gridsp[2])/(xi[2]-xj[2]);
        
    // this sets the time larger than 1 if there is no crossing
    t[0] = (c[0] == 0)*doubleval(1.1) + (c[0] != 0)*t[0];
    t[1] = (c[1] == 0)*doubleval(1.1) + (c[1] != 0)*t[1];
    t[2] = (c[2] == 0)*doubleval(1.1) + (c[2] != 0)*t[2];
    
    // track previous time of crossing and check that sum is complete (?)
    oldt = doubleval(0.0); 

    // while we don't reach the last point...
    while( c[0]*x[0] + c[1]*x[1] + c[2]*x[2] < c[0]*i2[0] + c[1]*i2[1] + c[2]*i2[2] )
    {
        // figure out index
        cmp0x = ((t[0]<t[1]+cu_eps) + (t[0]<t[2]+cu_eps))/2;
        cmp1x = ((t[1]<t[0]+cu_eps) + (t[1]<t[2]+cu_eps))/2;
        cmp2x = ((t[2]<t[0]+cu_eps) + (t[2]<t[1]+cu_eps))/2;
        iX = 0*cmp0x+1*cmp1x+2*cmp2x;

        // distribute the contribution
        spread_line_source(oldt,t[iX],diff,d_cgrid,x,stress,current_grid);

        // move to next cross point
        d_cgrid[iX] -= c[iX] * (single_t)c_gridsp[iX];
        oldt = t[iX];
        
        x[iX] += c[iX];
        xn[iX] += c[iX];

        // Next cross point:
        t[iX] = (xi[iX]-xn[iX] * (double_t)c_gridsp[iX])/(xi[iX]-xj[iX]);

        if(iX > 2) //To avoid infinite loops
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

    // create event
    for (int i = 0; i < cu_batches; ++i) 
    {
        checkCuda(cudaEventCreate(&h_mem_event[i]));
        checkCuda(cudaEventRecord(h_mem_event[i]));
    }

    // clear it
    custress_clear();

    // allocate global memory
    checkCuda(cudaMalloc((void**)&d_sum_grid,ncells*sizeof(cu_smatrix)));
    checkCuda(cudaMalloc((void**)&d_batch,sizeof(batcharrays_t[cu_batches])));

    // allocate host memory
    h_batch = new batcharrays_t[cu_batches];
    h_sum_grid = new cu_smatrix[ncells];
    
    cu_iarray cu_nxyz = {(int_t)nx, (int_t)ny, (int_t)nz};
    checkCuda(cudaMemcpyToSymbol(c_nxyz, cu_nxyz, sizeof(mds::iarray)));

    // initialize global memory
    checkCuda(cudaMemset(d_sum_grid, 0, ncells*sizeof(cu_smatrix)));
    checkCuda(cudaMemset(d_batch,    0, sizeof(batcharrays_t[cu_batches])));

    // initialize host memory
    h_ncells = (uint_t)ncells;
    for (int i = 0; i < cu_batches; ++i) h_bindex[i] = 0;
    memset(h_sum_grid, 0, ncells*sizeof(cu_smatrix));
    memset(h_batch,    0, sizeof(batcharrays_t[cu_batches]));
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
    
    for (int i = 0; i < cu_batches; ++i)
    {
        checkCuda(cudaEventDestroy(h_mem_event[i]));
        h_mem_event[i] = nullptr;
    }
}

void custress_update_box_spacings(const mds::dmatrix box, const mds::dmatrix invbox, const mds::darray gridsp)
{
    // convert to custress types
    cu_smatrix cu_box = {
        {(single_t)box[0][0], (single_t)box[0][1], (single_t)box[0][2]},
        {(single_t)box[1][0], (single_t)box[1][1], (single_t)box[1][2]},
        {(single_t)box[2][0], (single_t)box[2][1], (single_t)box[2][2]},
    };
    cu_smatrix cu_invbox = {
        {(single_t)invbox[0][0], (single_t)invbox[0][1], (single_t)invbox[0][2]},
        {(single_t)invbox[1][0], (single_t)invbox[1][1], (single_t)invbox[1][2]},
        {(single_t)invbox[2][0], (single_t)invbox[2][1], (single_t)invbox[2][2]},
    };
    single_t cu_gridsp [] = {
        (single_t)(gridsp[0]),                     // lx
        (single_t)(gridsp[1]),                     // ly
        (single_t)(gridsp[2]),                     // lz
        (single_t)(gridsp[0]*gridsp[1]),           // lxy
        (single_t)(gridsp[0]*gridsp[2]),           // lxz
        (single_t)(gridsp[1]*gridsp[2]),           // lyz
        (single_t)(gridsp[0]*gridsp[1]*gridsp[2]), // lxyz
        (single_t)(0.125/((gridsp[0]*gridsp[1]*gridsp[2])*(gridsp[0]*gridsp[1]*gridsp[2]))) // C
    };

    // initialize constant memory variables
    checkCuda(cudaMemcpyToSymbol(c_box,    cu_box,    sizeof(cu_smatrix)));
    checkCuda(cudaMemcpyToSymbol(c_invbox, cu_invbox, sizeof(cu_smatrix)));
    checkCuda(cudaMemcpyToSymbol(c_gridsp, cu_gridsp, sizeof(cu_gridsp)));
}

// gpu kernel
__global__ static void process_batch(uint_t batch_index, const batcharrays_t * __restrict__ batch, cu_smatrix * current_grid)
{
    auto index = blockIdx.x*cu_threads_per_block+threadIdx.x;

    // guard execution
    if (index < batch_index)
        BatchPairInteraction(batch->Ri[index], batch->Rj[index], batch->Fij[index], current_grid);
}

// launches GPU kernel
void custress_distribute_pair_interaction(const mds::darray xi, const mds::darray xj, const mds::darray Fij, std::mutex * state_mutex)
{
    // calculate the diff
    double_t this_length =
        (xi[0]-xj[0])*(xi[0]-xj[0])+
        (xi[1]-xj[1])*(xi[1]-xj[1])+
        (xi[2]-xj[2])*(xi[2]-xj[2]);
    
    if (this_length > h_length_max[cu_batches-1u])
    {
        state_mutex->lock();
        for (int i = 0; i < cu_batches; ++i)
            h_mutex_kernel[i].lock();

        for (int i = 0; i < cu_batches; ++i)
            h_length_max[i] = (i+1)*(this_length/cu_batches);

        for (int i = 0; i < cu_batches; ++i)
            h_mutex_kernel[i].unlock();
        state_mutex->unlock();
    }
    
    // calculate the possible maximum length
    int i = 0;
    do
    {
        if (this_length < h_length_max[i])
            break;
        else
            ++i;
    } while(true); 
    i = (i >= cu_batches) ? cu_batches-1u : i;
    
    h_mutex_kernel[i].lock();
    checkCuda(cudaEventSynchronize(h_mem_event[i]));
    
    // store for later processing
    h_batch[i].Ri[h_bindex[i]][0] = (single_t)xi[0];
    h_batch[i].Ri[h_bindex[i]][1] = (single_t)xi[1];
    h_batch[i].Ri[h_bindex[i]][2] = (single_t)xi[2];
    h_batch[i].Rj[h_bindex[i]][0] = (single_t)xj[0];
    h_batch[i].Rj[h_bindex[i]][1] = (single_t)xj[1];
    h_batch[i].Rj[h_bindex[i]][2] = (single_t)xj[2];
    h_batch[i].Fij[h_bindex[i]][0] = (single_t)Fij[0];
    h_batch[i].Fij[h_bindex[i]][1] = (single_t)Fij[1];
    h_batch[i].Fij[h_bindex[i]][2] = (single_t)Fij[2];
    h_bindex[i] += 1;

    // if bindex is cu_batchsize we process
    if (h_bindex[i] == cu_batchsize)
    {
        //state_mutex->lock();

        // transfer the grid to host
        checkCuda(cudaMemcpyAsync(
                &d_batch[i],
                &h_batch[i],
                sizeof(batcharrays_t),
                cudaMemcpyHostToDevice) );
        checkCuda(cudaEventRecord(h_mem_event[i]));

        // execute with a single element processed by each streaming multiprocessor
        process_batch<<<cu_batchsize/cu_threads_per_block,cu_threads_per_block>>>(h_bindex[i], &d_batch[i], d_sum_grid);
        h_bindex[i] = 0;
        
        //state_mutex->unlock();
    }
    h_mutex_kernel[i].unlock();
}

void custress_sum_grid(mds::dmatrix * current_grid)
{
    // finish off any remaining pairs
    for (int i = 0; i < cu_batches; ++i)
    {
        if (h_bindex[i] > 0)
        {
            // transfer the batch to device
            checkCuda(cudaMemcpyAsync(
                    &d_batch[i],
                    &h_batch[i],
                    sizeof(batcharrays_t),
                    cudaMemcpyHostToDevice) );

            // execute with a single element processed by each streaming multiprocessor
            process_batch<<<cu_batchsize/cu_threads_per_block,cu_threads_per_block>>>(h_bindex[i], &d_batch[i], d_sum_grid);

            h_bindex[i] = 0;
        }
    }
    
    // transfer the grid to host
    checkCuda(cudaMemcpy(
            h_sum_grid,
            d_sum_grid,
            h_ncells*sizeof(cu_smatrix),
            cudaMemcpyDeviceToHost) );
    
    // sum into mdstress current_grid
    for (uint_t i = 0; i < h_ncells; ++i)
    {
        current_grid[i][0][0] += (double)h_sum_grid[i][0][0];//.x + (double)h_sum_grid[i][0][0].y;
        current_grid[i][0][1] += (double)h_sum_grid[i][0][1];//.x + (double)h_sum_grid[i][0][1].y;
        current_grid[i][0][2] += (double)h_sum_grid[i][0][2];//.x + (double)h_sum_grid[i][0][2].y;
        current_grid[i][1][0] += (double)h_sum_grid[i][1][0];//.x + (double)h_sum_grid[i][1][0].y;
        current_grid[i][1][1] += (double)h_sum_grid[i][1][1];//.x + (double)h_sum_grid[i][1][1].y;
        current_grid[i][1][2] += (double)h_sum_grid[i][1][2];//.x + (double)h_sum_grid[i][1][2].y;
        current_grid[i][2][0] += (double)h_sum_grid[i][2][0];//.x + (double)h_sum_grid[i][2][0].y;
        current_grid[i][2][1] += (double)h_sum_grid[i][2][1];//.x + (double)h_sum_grid[i][2][1].y;
        current_grid[i][2][2] += (double)h_sum_grid[i][2][2];//.x + (double)h_sum_grid[i][2][2].y;
    }
}
