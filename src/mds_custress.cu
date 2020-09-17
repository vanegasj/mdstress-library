#include "mds_custress.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>

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

//#define SINGLE_PRECISION
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

#define cu_batchsize 8192
#define cu_threads_per_block 32

typedef struct {
    cu_sarray Ri;
    cu_sarray Rj;
    cu_sarray Fij;
} cu_pair_t;

typedef struct {
    cu_pair_t pair[cu_batchsize];
} cu_batches_t;

// device constant memory is cached
__constant__ cu_barray  c_periodic;
__constant__ cu_iarray  c_nxyz;
__constant__ cu_smatrix c_box;
__constant__ cu_smatrix c_invbox;
__constant__ single_t   c_gridsp[8];

// host parameters
size_t h_ncells = 0;
size_t h_nbatches = 0;
size_t h_griddim = 3;

// host memory
uint_t        *h_bindex       = nullptr;
cu_batches_t  *h_batch        = nullptr;
cu_smatrix    *h_sum_grid     = nullptr;
cudaEvent_t   *h_mem_event    = nullptr;
cudaStream_t  *h_stream       = nullptr;

// device global memory is not, so minimize access here
cu_batches_t  *d_batch    = nullptr;
cu_smatrix    *d_sum_grid = nullptr;

// cuda context
const dim3 batch_blocks = {cu_batchsize/cu_threads_per_block,1,1};
const dim3 batch_threads = {cu_threads_per_block,1,1};

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
        c[0] -= (c_periodic[0] == true && c[0] >   singleval(0.5)*c_box[0][0]) ? c_box[0][0] : singleval(0.0);
        c[0] += (c_periodic[0] == true && c[0] <= -singleval(0.5)*c_box[0][0]) ? c_box[0][0] : singleval(0.0);
        c[1] -= (c_periodic[1] == true && c[1] >   singleval(0.5)*c_box[1][1]) ? c_box[1][1] : singleval(0.0);
        c[1] += (c_periodic[1] == true && c[1] <= -singleval(0.5)*c_box[1][1]) ? c_box[1][1] : singleval(0.0);
        c[2] -= (c_periodic[2] == true && c[2] >   singleval(0.5)*c_box[2][2]) ? c_box[2][2] : singleval(0.0);
        c[2] += (c_periodic[2] == true && c[2] <= -singleval(0.5)*c_box[2][2]) ? c_box[2][2] : singleval(0.0);
    }
}

// public functions
void custress_init(size_t nbatches_min, size_t ncells, int nx, int ny, int nz)
{
    // clear it
    custress_clear();

    // set host parameters
    h_ncells = ncells;
    h_nbatches = nbatches_min;

    // allocate host memory
    h_bindex    = new uint_t[h_nbatches];
    h_sum_grid  = new cu_smatrix[h_ncells];
    h_mem_event = new cudaEvent_t[h_nbatches];
    h_stream    = new cudaStream_t[h_nbatches];
    
    // note grid type
    if (nz == ncells)
        h_griddim = 2;
    else
    if (ny == ncells)
        h_griddim = 1;
    else
    if (nx == ncells)
        h_griddim = 0;
    else
        h_griddim = 3;

    // initialize device
    checkCuda(cudaSetDevice(0));
    checkCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    //checkCuda(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    checkCuda(cudaSetDeviceFlags(cudaDeviceScheduleYield));
    checkCuda(cudaSetDeviceFlags(cudaDeviceLmemResizeToMax));

    // allocate host memory with cuda to ensure it is pinned
    checkCuda(cudaMallocHost((void**)&h_batch, sizeof(cu_batches_t[h_nbatches])));

    // allocate global memory
    checkCuda(cudaMalloc((void**)&d_batch,sizeof(cu_batches_t[h_nbatches])));
    checkCuda(cudaMalloc((void**)&d_sum_grid,sizeof(cu_smatrix[h_ncells])));
    
    // initialize global memory
    checkCuda(cudaMemset(d_sum_grid, 0, sizeof(cu_smatrix[h_ncells])));
    checkCuda(cudaMemset(d_batch,    0, sizeof(cu_batches_t[h_nbatches])));
    
    // create events and streams
    for (int i = 0; i < h_nbatches; ++i) 
    {
        checkCuda(cudaStreamCreate(&h_stream[i]));
        checkCuda(cudaEventCreateWithFlags(&h_mem_event[i], cudaEventDisableTiming));
        checkCuda(cudaEventRecord(h_mem_event[i],h_stream[i]));
    }
    
    // copy symbols
    cu_iarray cu_nxyz = {(int_t)nx, (int_t)ny, (int_t)nz};
    checkCuda(cudaMemcpyToSymbol(c_nxyz, cu_nxyz, sizeof(mds::iarray)));

    // initialize host memory
    for (int i = 0; i < h_nbatches; ++i)
        h_bindex[i] = 0;
    memset(h_sum_grid, 0, sizeof(cu_smatrix[h_ncells]));
    memset(h_batch,    0, sizeof(cu_batches_t[h_nbatches]));
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
    if (h_ncells != 0 && h_nbatches != 0)
    {
        for (int i = 0; i < h_nbatches; ++i)
        {
            checkCuda(cudaEventDestroy(h_mem_event[i]));
            checkCuda(cudaStreamDestroy(h_stream[i]));
        }
        
        // free global memory
        if (d_sum_grid != nullptr) cudaFree(d_sum_grid);
        if (d_batch != nullptr) cudaFree(d_batch);
        
        // free host memory
        if (h_batch != nullptr) checkCuda(cudaFreeHost(h_batch));
        if (h_sum_grid != nullptr) free(h_sum_grid);
        if (h_mem_event != nullptr) free(h_mem_event);
        if (h_stream != nullptr) free(h_stream);
        
        // set host and device pointers to null
        h_bindex       = nullptr;
        h_batch        = nullptr;
        h_sum_grid     = nullptr;
        h_mem_event    = nullptr;
        h_stream       = nullptr;
        d_batch        = nullptr;
        d_sum_grid     = nullptr;

        // zero the host parameters
        h_ncells   = 0u;
        h_nbatches = 0u;
        h_griddim = 3;
    }
}

// gpu kernel
__global__ static void
__launch_bounds__(cu_threads_per_block, 32)
process_batch_1d(uint_t dim, uint_t max_index, const cu_batches_t * __restrict__ batch, cu_smatrix * current_grid)
{
    int index = blockIdx.x*cu_threads_per_block+threadIdx.x;
    if (index < max_index)
    {
        cu_pair_t this_pair = batch->pair[index];
        
        double_t oldt, newt, t_c1, t_c2;
        double_t d_cgrid;
        cu_sarray diff;

        int_t i2; //grid cell corresponding to particle J (B)
        int_t x;  //cell during spreading
        int_t xn; //next cell during spreading
        int_t c;  //director
        
        cu_smatrix stress;
        
        // scalars used to prepare vectors
        single_t t12,t22, dt1, dt2;
        int_t kkp1, kkm1;

        // vectors and a single coefficient
        single_t D[2];
        single_t factor[2];

        //------------------------------------------------------------------------------------
        // Calculate the stress tensor
        diff_array(this_pair.Rj, this_pair.Ri, diff);
        stress[0][0] = this_pair.Fij[0]*diff[0];
        stress[1][0] = this_pair.Fij[1]*diff[0];
        stress[2][0] = this_pair.Fij[2]*diff[0];
        stress[0][1] = this_pair.Fij[0]*diff[1];
        stress[1][1] = this_pair.Fij[1]*diff[1];
        stress[2][1] = this_pair.Fij[2]*diff[1];
        stress[0][2] = this_pair.Fij[0]*diff[2];
        stress[1][2] = this_pair.Fij[1]*diff[2];
        stress[2][2] = this_pair.Fij[2]*diff[2];

        //------------------------------------------------------------------------------------
        // Distribute the stress

        // calculate the grid coordinates (no pbc) for the extreme points
        x = c_nxyz[dim] * this_pair.Ri[dim] * c_invbox[dim][dim] - (this_pair.Ri[dim] < doubleval(0.0));
        i2 = c_nxyz[dim] * this_pair.Rj[dim] * c_invbox[dim][dim] - (this_pair.Rj[dim] < doubleval(0.0));

        // d_cgrid = vector from the center of the present cell to the initial point
        d_cgrid = this_pair.Ri[dim]-(x+singleval(0.5))*c_gridsp[dim];
        
        // c is a vector that guide the advance in each coordinate (+1 if it has to advance in this coordinate, -1 if it has to go back or 0 if it has to do nothing)
        c = (i2>x)-(x>i2);
        
        // label of the next cell is 1 step further than the previous in this direction
        xn = x+(c+1)/2;
        
        t_c1 = this_pair.Ri[dim] / (this_pair.Ri[dim]-this_pair.Rj[dim]);
        t_c2 = c_gridsp[dim] / (this_pair.Ri[dim]-this_pair.Rj[dim]);

        // parametric time of crossing
        oldt = doubleval(0.0); 
        newt = (c == 0) ? doubleval(1.1) : t_c1-xn*t_c2;

        // while we don't reach the last point...
        int iterations = c*(i2-x);
        for (int count = 0; count <= iterations; ++count)
        {
            // figure out index
            newt = (iterations == count) ? doubleval(1.0) : newt;

            // distribute the contribution
            // work out the parametric time constants
            t12 = oldt*oldt;
            t22 = newt*newt;
            dt1 = newt-oldt;
            dt2 = t22 - t12;
            
            // and the index constants
            kkp1 = ((x + 1 + c_nxyz[dim]) % c_nxyz[dim]);
            kkm1 = ((x + c_nxyz[dim]) % c_nxyz[dim]);

            // the composite constants in terms of i, j, k
            D[0] = c_gridsp[5-dim]*(singleval(2.0)*d_cgrid*dt1+diff[dim]*dt2);
            D[1] = c_gridsp[6]*dt1;

            // prepare the factors
            factor[0] = singleval(4.0)*c_gridsp[7]*( D[0] + D[1]);
            factor[1] = singleval(4.0)*c_gridsp[7]*(-D[0] + D[1]);

            // perform the sums into the grid
            cu_ssmatm(factor[0], stress, current_grid[kkp1]);
            cu_ssmatm(factor[1], stress, current_grid[kkm1]);

            // move to next cross point
            d_cgrid -= c * (single_t)c_gridsp[dim];
            oldt = newt;
            
            x += c;
            xn += c;

            // Next cross point:
            newt = t_c1-xn*t_c2;
        }
    }
}

__global__ static void
process_batch_3d(uint_t max_index, const cu_batches_t * __restrict__ batch, cu_smatrix * __restrict__ current_grid)
{
    int index = blockIdx.x*cu_threads_per_block+threadIdx.x;
    if (index < max_index)
    {
        cu_pair_t this_pair = batch->pair[index];

        double_t oldt, newt;
        int_t cmp0x,cmp1x,cmp2x,iX;

        cu_darray t, t_c1, t_c2;
        cu_sarray d_cgrid;
        cu_sarray diff;

        cu_iarray i2; //grid cell corresponding to particle J (B)
        cu_iarray x;  //cell during spreading
        cu_iarray xn; //next cell during spreading
        cu_iarray c;  //director
        
        cu_smatrix stress;
        
        // scalars used to prepare vectors
        single_t t12,t22, dt1, dt2, dt3, dt4;
        single_t axy, axz, ayz, axyz;
        single_t bxy, bxz, byz, bxyz;
        int_t iip1, iim1, jjp1, jjm1, kkp1, kkm1;

        // vectors and a single coefficient
        single_t D[8];
        single_t factor[8];

        //------------------------------------------------------------------------------------
        // Calculate the stress tensor
        diff_array(this_pair.Rj, this_pair.Ri, diff);
        stress[0][0] = this_pair.Fij[0]*diff[0];
        stress[1][0] = this_pair.Fij[1]*diff[0];
        stress[2][0] = this_pair.Fij[2]*diff[0];
        stress[0][1] = this_pair.Fij[0]*diff[1];
        stress[1][1] = this_pair.Fij[1]*diff[1];
        stress[2][1] = this_pair.Fij[2]*diff[1];
        stress[0][2] = this_pair.Fij[0]*diff[2];
        stress[1][2] = this_pair.Fij[1]*diff[2];
        stress[2][2] = this_pair.Fij[2]*diff[2];

        //------------------------------------------------------------------------------------
        // Distribute the stress

        // calculate the grid coordinates (no pbc) for the extreme points
        x[0] = c_nxyz[0] * this_pair.Ri[0] * c_invbox[0][0] - (this_pair.Ri[0] < doubleval(0.0));
        x[1] = c_nxyz[1] * this_pair.Ri[1] * c_invbox[1][1] - (this_pair.Ri[1] < doubleval(0.0));
        x[2] = c_nxyz[2] * this_pair.Ri[2] * c_invbox[2][2] - (this_pair.Ri[2] < doubleval(0.0));
        i2[0] = c_nxyz[0] * this_pair.Rj[0] * c_invbox[0][0] - (this_pair.Rj[0] < doubleval(0.0));
        i2[1] = c_nxyz[1] * this_pair.Rj[1] * c_invbox[1][1] - (this_pair.Rj[1] < doubleval(0.0));
        i2[2] = c_nxyz[2] * this_pair.Rj[2] * c_invbox[2][2] - (this_pair.Rj[2] < doubleval(0.0));

        // d_cgrid = vector from the center of the present cell to the initial point
        d_cgrid[0] = this_pair.Ri[0]-(x[0]+singleval(0.5))*c_gridsp[0];
        d_cgrid[1] = this_pair.Ri[1]-(x[1]+singleval(0.5))*c_gridsp[1];
        d_cgrid[2] = this_pair.Ri[2]-(x[2]+singleval(0.5))*c_gridsp[2];
        
        // c is a vector that guide the advance in each coordinate (+1 if it has to advance in this coordinate, -1 if it has to go back or 0 if it has to do nothing)
        c[0] = (i2[0]>x[0])-(x[0]>i2[0]);
        c[1] = (i2[1]>x[1])-(x[1]>i2[1]);
        c[2] = (i2[2]>x[2])-(x[2]>i2[2]);
        
        // label of the next cell is 1 step further than the previous in this direction
        xn[0] = x[0]+(c[0]+1)/2;
        xn[1] = x[1]+(c[1]+1)/2;
        xn[2] = x[2]+(c[2]+1)/2;
        
        t_c1[0] = this_pair.Ri[0] / (this_pair.Ri[0]-this_pair.Rj[0]);
        t_c1[1] = this_pair.Ri[1] / (this_pair.Ri[1]-this_pair.Rj[1]);
        t_c1[2] = this_pair.Ri[2] / (this_pair.Ri[2]-this_pair.Rj[2]);
        t_c2[0] = c_gridsp[0] / (this_pair.Ri[0]-this_pair.Rj[0]);
        t_c2[1] = c_gridsp[1] / (this_pair.Ri[1]-this_pair.Rj[1]);
        t_c2[2] = c_gridsp[2] / (this_pair.Ri[2]-this_pair.Rj[2]);

        // parametric time of crossing
        t[0] = t_c1[0]-xn[0]*t_c2[0];
        t[1] = t_c1[1]-xn[1]*t_c2[1];
        t[2] = t_c1[2]-xn[2]*t_c2[2];
            
        // this sets the time larger than 1 if there is no crossing
        t[0] = (c[0] == 0)*doubleval(1.1) + (c[0] != 0)*t[0];
        t[1] = (c[1] == 0)*doubleval(1.1) + (c[1] != 0)*t[1];
        t[2] = (c[2] == 0)*doubleval(1.1) + (c[2] != 0)*t[2];
        
        // track previous time of crossing
        oldt = doubleval(0.0); 

        // while we don't reach the last point...
        int iterations = c[0]*(i2[0]-x[0]) + c[1]*(i2[1]-x[1]) + c[2]*(i2[2]-x[2]);
        for (int count = 0; count <= iterations; ++count)
        {
            // figure out index
            cmp0x = ((t[0]<t[1]+cu_eps) + (t[0]<t[2]+cu_eps))/2;
            cmp1x = ((t[1]<t[0]+cu_eps) + (t[1]<t[2]+cu_eps))/2;
            cmp2x = ((t[2]<t[0]+cu_eps) + (t[2]<t[1]+cu_eps))/2;
            iX = (1-cmp0x)*(cmp1x+2*(1-cmp1x)*cmp2x);

            // last iteration of loop distributes the remainder
            newt = (iterations == count) ? doubleval(1.0) : t[iX];

            // distribute the contribution
            // work out the parametric time constants
            t12 = oldt*oldt;
            t22 = newt*newt;
            dt1 = newt - oldt;
            dt2 = t22 - t12;
            dt3 = singleval(4.0)*(t22*newt - t12*oldt)/singleval(3.0);
            dt4 = t22*t22 - t12*t12;
            
            // now the position/spatial constants
            axy = diff[0]*diff[1]; axz = diff[0]*diff[2]; ayz = diff[1]*diff[2];
            bxy = d_cgrid[0]*d_cgrid[1]; bxz = d_cgrid[0]*d_cgrid[2]; byz = d_cgrid[1]*d_cgrid[2];
            axyz = diff[0]*ayz; bxyz = d_cgrid[0]*byz;

            // and the index constants
            iip1 = ((x[0] + 1 + c_nxyz[0]) % c_nxyz[0])*c_nxyz[1]*c_nxyz[2];
            jjp1 = ((x[1] + 1 + c_nxyz[1]) % c_nxyz[1])*c_nxyz[2];
            kkp1 = ((x[2] + 1 + c_nxyz[2]) % c_nxyz[2]);
            iim1 = ((x[0] + c_nxyz[0]) % c_nxyz[0])*c_nxyz[1]*c_nxyz[2];
            jjm1 = ((x[1] + c_nxyz[1]) % c_nxyz[1])*c_nxyz[2];
            kkm1 = ((x[2] + c_nxyz[2]) % c_nxyz[2]);
            
            // the composite constants in terms of i, j, k
            D[0] = singleval(8.0)*bxyz*dt1 + singleval(4.0)*(diff[0]*byz+diff[1]*bxz+diff[2]*bxy)*dt2
                + singleval(2.0)*(d_cgrid[0]*ayz+d_cgrid[1]*axz+d_cgrid[2]*axy)*dt3 + singleval(2.0)*axyz*dt4;
            D[1] = c_gridsp[0]*(singleval(4.0)*byz*dt1 + singleval(2.0)*(diff[1]*d_cgrid[2]+diff[2]*d_cgrid[1])*dt2 + ayz*dt3);
            D[2] = c_gridsp[1]*(singleval(4.0)*bxz*dt1 + singleval(2.0)*(diff[0]*d_cgrid[2]+diff[2]*d_cgrid[0])*dt2 + axz*dt3);
            D[3] = c_gridsp[2]*(singleval(4.0)*bxy*dt1 + singleval(2.0)*(diff[0]*d_cgrid[1]+diff[1]*d_cgrid[0])*dt2 + axy*dt3);
            D[4] = c_gridsp[3]*(singleval(2.0)*d_cgrid[2]*dt1+diff[2]*dt2);
            D[5] = c_gridsp[4]*(singleval(2.0)*d_cgrid[1]*dt1+diff[1]*dt2);
            D[6] = c_gridsp[5]*(singleval(2.0)*d_cgrid[0]*dt1+diff[0]*dt2);
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

            // move to next cross point
            d_cgrid[iX] -= c[iX] * (single_t)c_gridsp[iX];
            oldt = t[iX];
            
            x[iX] += c[iX];
            xn[iX] += c[iX];

            // Next cross point:
            t[iX] = t_c1[iX]-xn[iX]*t_c2[iX];
        }
    }
}

void custress_update_box_spacings(const mds::dmatrix box, const mds::dmatrix invbox, const mds::darray gridsp)
{
    // empty current batches
    for (int i = 0; i < h_nbatches; ++i)
    {
        if (h_bindex[i] > 0 && h_bindex[i] != cu_batchsize)
        {
            // transfer the batch to device
            checkCuda(cudaMemcpyAsync(
                    &d_batch[i],
                    &h_batch[i],
                    sizeof(cu_batches_t),
                    cudaMemcpyHostToDevice,
                    h_stream[i]) );
            checkCuda(cudaEventRecord(h_mem_event[i],h_stream[i]));

            // execute with a single element processed by each streaming multiprocessor
            if (h_griddim != 3)
                process_batch_1d<<<batch_blocks,batch_threads,0u,h_stream[i]>>>(h_griddim, h_bindex[i], &d_batch[i], d_sum_grid);
            else
                process_batch_3d<<<batch_blocks,batch_threads,0u,h_stream[i]>>>(h_bindex[i], &d_batch[i], d_sum_grid);
            
            // set the batchindex to batchsize to trigger an event sync in distribute pair
            h_bindex[i] = cu_batchsize;
        }
    }

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

// launches GPU kernel
bool custress_distribute_pair_interaction(const mds::darray xi, const mds::darray xj, const mds::darray Fij, int batch_id)
{
    if (h_bindex[batch_id] == cu_batchsize)
    {
        //checkCuda(cudaEventSynchronize(h_mem_event[batch_id]));
        if (cudaSuccess != checkCuda(cudaEventQuery(h_mem_event[batch_id])))
            return false;
        else
            h_bindex[batch_id] = 0;
    }

    // store for later processing
    h_batch[batch_id].pair[h_bindex[batch_id]].Ri[0] = (single_t)xi[0];
    h_batch[batch_id].pair[h_bindex[batch_id]].Ri[1] = (single_t)xi[1];
    h_batch[batch_id].pair[h_bindex[batch_id]].Ri[2] = (single_t)xi[2];
    h_batch[batch_id].pair[h_bindex[batch_id]].Rj[0] = (single_t)xj[0];
    h_batch[batch_id].pair[h_bindex[batch_id]].Rj[1] = (single_t)xj[1];
    h_batch[batch_id].pair[h_bindex[batch_id]].Rj[2] = (single_t)xj[2];
    h_batch[batch_id].pair[h_bindex[batch_id]].Fij[0] = (single_t)Fij[0];
    h_batch[batch_id].pair[h_bindex[batch_id]].Fij[1] = (single_t)Fij[1];
    h_batch[batch_id].pair[h_bindex[batch_id]].Fij[2] = (single_t)Fij[2];
    h_bindex[batch_id] += 1;

    // if bindex is cu_batchsize we process
    if (h_bindex[batch_id] == cu_batchsize)
    {
        // transfer the grid to host
        checkCuda(cudaMemcpyAsync(
                &d_batch[batch_id],
                &h_batch[batch_id],
                sizeof(cu_batches_t),
                cudaMemcpyHostToDevice,
                h_stream[batch_id]));
        checkCuda(cudaEventRecord(h_mem_event[batch_id],h_stream[batch_id]));

        // execute with a single element processed by each streaming multiprocessor
        if (h_griddim != 3)
            process_batch_1d<<<batch_blocks,batch_threads,0u,h_stream[batch_id]>>>(h_griddim, h_bindex[batch_id], &d_batch[batch_id], d_sum_grid);
        else
            process_batch_3d<<<batch_blocks,batch_threads,0u,h_stream[batch_id]>>>(h_bindex[batch_id], &d_batch[batch_id], d_sum_grid);
    }

    return true;
}

void custress_sum_grid(mds::dmatrix * current_grid)
{
    // finish off any remaining pairs
    for (int i = 0; i < h_nbatches; ++i)
    {
        if (h_bindex[i] > 0 && h_bindex[i] != cu_batchsize)
        {
            // transfer the batch to device
            checkCuda(cudaMemcpyAsync(
                    &d_batch[i],
                    &h_batch[i],
                    sizeof(cu_batches_t),
                    cudaMemcpyHostToDevice,
                    h_stream[i]) );

            // execute with a single element processed by each streaming multiprocessor
            if (h_griddim != 3)
                process_batch_1d<<<batch_blocks,batch_threads,0u,h_stream[i]>>>(h_griddim, h_bindex[i], &d_batch[i], d_sum_grid);
            else
                process_batch_3d<<<batch_blocks,batch_threads,0u,h_stream[i]>>>(h_bindex[i], &d_batch[i], d_sum_grid);
            
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
        current_grid[i][0][0] += (double)h_sum_grid[i][0][0];
        current_grid[i][0][1] += (double)h_sum_grid[i][0][1];
        current_grid[i][0][2] += (double)h_sum_grid[i][0][2];
        current_grid[i][1][0] += (double)h_sum_grid[i][1][0];
        current_grid[i][1][1] += (double)h_sum_grid[i][1][1];
        current_grid[i][1][2] += (double)h_sum_grid[i][1][2];
        current_grid[i][2][0] += (double)h_sum_grid[i][2][0];
        current_grid[i][2][1] += (double)h_sum_grid[i][2][1];
        current_grid[i][2][2] += (double)h_sum_grid[i][2][2];
    }
    
    // zero grids on device
    checkCuda(cudaMemset(d_sum_grid, 0, sizeof(cu_smatrix[h_ncells])));
}
