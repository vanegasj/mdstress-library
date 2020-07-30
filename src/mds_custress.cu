#include "mds_custress.h"
#include <cuda.h>

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
#define cu_threads_per_block 128
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

// host parameters
uint_t h_ncells = 0;
uint_t h_nbatches = 0;

// host memory
uint_t        *h_bindex       = nullptr;
batcharrays_t *h_batch        = nullptr;
cu_smatrix    *h_sum_grid     = nullptr;
cudaEvent_t   *h_mem_event    = nullptr;
cudaStream_t  *h_stream       = nullptr;
double_t      *h_length_max   = nullptr;

// device global memory is not, so minimize access here
batcharrays_t *d_batch    = nullptr;
cu_smatrix    *d_sum_grid = nullptr;

// cuda context
const dim3 batch_blocks = {cu_batchsize/cu_threads_per_block,1,1};
const dim3 batch_threads = {cu_threads_per_block,1,1};

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
        iX = (1-cmp0x)*(cmp1x+2*(1-cmp1x)*cmp2x);

        // distribute the contribution
        spread_line_source(oldt,t[iX],diff,d_cgrid,x,stress,current_grid);

        // move to next cross point
        d_cgrid[iX] -= c[iX] * (single_t)c_gridsp[iX];
        oldt = t[iX];
        
        x[iX] += c[iX];
        xn[iX] += c[iX];

        // Next cross point:
        t[iX] = (xi[iX]-xn[iX] * (double_t)c_gridsp[iX])/(xi[iX]-xj[iX]);
    }

    // Distribute the last contribution
    spread_line_source(oldt,1,diff,d_cgrid,x,stress,current_grid);
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
    h_batch     = new batcharrays_t[h_nbatches];
    h_sum_grid  = new cu_smatrix[h_nbatches*h_ncells];
    h_mem_event = new cudaEvent_t[h_nbatches];
    h_stream    = new cudaStream_t[h_nbatches];
    h_length_max = new double_t[h_nbatches];

    // initialize device
    checkCuda(cudaSetDevice(0));

    // create events and streams
    for (int i = 0; i < h_nbatches; ++i) 
    {
        checkCuda(cudaStreamCreate(&h_stream[i]));
        checkCuda(cudaEventCreateWithFlags(&h_mem_event[i], cudaEventDisableTiming));
        checkCuda(cudaEventRecord(h_mem_event[i],h_stream[i]));
    }

    // allocate global memory
    checkCuda(cudaMalloc((void**)&d_batch,sizeof(batcharrays_t[h_nbatches])));
    checkCuda(cudaMalloc((void**)&d_sum_grid,sizeof(cu_smatrix[h_nbatches*h_ncells])));
    
    // copy symbols
    cu_iarray cu_nxyz = {(int_t)nx, (int_t)ny, (int_t)nz};
    checkCuda(cudaMemcpyToSymbol(c_nxyz, cu_nxyz, sizeof(mds::iarray)));

    // initialize global memory
    checkCuda(cudaMemset(d_sum_grid, 0, sizeof(cu_smatrix[h_nbatches*h_ncells])));
    checkCuda(cudaMemset(d_batch,    0, sizeof(batcharrays_t[h_nbatches])));

    // initialize host memory
    for (int i = 0; i < h_nbatches; ++i) h_bindex[i] = 0;
    for (int i = 0; i < h_nbatches; ++i) h_length_max[i] = doubleval(0.0);
    memset(h_sum_grid, 0, sizeof(cu_smatrix[h_nbatches*h_ncells]));
    memset(h_batch,    0, sizeof(batcharrays_t[h_nbatches]));
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
        if (h_batch != nullptr) free(h_batch);
        if (h_sum_grid != nullptr) free(h_sum_grid);
        if (h_mem_event != nullptr) free(h_mem_event);
        if (h_stream != nullptr) free(h_stream);
        if (h_length_max != nullptr)  free(h_length_max);
        
        // set host and device pointers to null
        h_bindex       = nullptr;
        h_batch        = nullptr;
        h_sum_grid     = nullptr;
        h_mem_event    = nullptr;
        h_stream       = nullptr;
        d_batch        = nullptr;
        d_sum_grid     = nullptr;
        h_length_max   = nullptr;

        // zero the host parameters
        h_ncells   = 0u;
        h_nbatches = 0u;
    }
}

// gpu kernel
__global__ static void process_batch(uint_t max_index, const batcharrays_t * __restrict__ batch, cu_smatrix * current_grid)
{
    auto index = blockIdx.x*cu_threads_per_block+threadIdx.x;

    // guard execution
    if (index < max_index)
        BatchPairInteraction(batch->Ri[index], batch->Rj[index], batch->Fij[index], current_grid);
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
                    sizeof(batcharrays_t),
                    cudaMemcpyHostToDevice,
                    h_stream[i]) );
            checkCuda(cudaEventRecord(h_mem_event[i],h_stream[i]));

            // execute with a single element processed by each streaming multiprocessor
            process_batch<<<batch_blocks,batch_threads,0u,h_stream[i]>>>(h_bindex[i], &d_batch[i], d_sum_grid+i*h_ncells);
            
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

__global__ static void reduce_grids(uint_t nbatches, uint_t ncells, cu_smatrix * current_grid)
{
    auto index = blockIdx.x*cu_threads_per_block+threadIdx.x;
    cu_smatrix this_cell = { { 0 } };

    if (index < ncells)
    {
        for (int_t i = nbatches-1; i >= 0 ; --i)
        {
            this_cell[0][0] += current_grid[index+i*ncells][0][0];
            this_cell[0][1] += current_grid[index+i*ncells][0][1];
            this_cell[0][2] += current_grid[index+i*ncells][0][2];
            this_cell[1][0] += current_grid[index+i*ncells][1][0];
            this_cell[1][1] += current_grid[index+i*ncells][1][1];
            this_cell[1][2] += current_grid[index+i*ncells][1][2];
            this_cell[2][0] += current_grid[index+i*ncells][2][0];
            this_cell[2][1] += current_grid[index+i*ncells][2][1];
            this_cell[2][2] += current_grid[index+i*ncells][2][2];
        }

        current_grid[index][0][0] = this_cell[0][0];
        current_grid[index][0][1] = this_cell[0][1];
        current_grid[index][0][2] = this_cell[0][2];
        current_grid[index][1][0] = this_cell[1][0];
        current_grid[index][1][1] = this_cell[1][1];
        current_grid[index][1][2] = this_cell[1][2];
        current_grid[index][2][0] = this_cell[2][0];
        current_grid[index][2][1] = this_cell[2][1];
        current_grid[index][2][2] = this_cell[2][2];
    }
}

// launches GPU kernel
//std::mutex h_length_mutex;
void custress_distribute_pair_interaction(const mds::darray xi, const mds::darray xj, const mds::darray Fij, int batch_id)
{
    // calculate the diff
    /*double_t this_length =
        (xi[0]-xj[0])*(xi[0]-xj[0])+
        (xi[1]-xj[1])*(xi[1]-xj[1])+
        (xi[2]-xj[2])*(xi[2]-xj[2]);
    
    if (this_length > h_length_max[h_nbatches-1u])
    {
        std::lock_guard<std::mutex> lock(h_length_mutex);
        for (int bin = 0; bin < h_nbatches; ++bin)
            h_length_max[bin] = (bin+1)*(this_length/h_nbatches);
    }
    
    int batch_id = 0;
    while (this_length > h_length_max[batch_id] && batch_id < h_nbatches-1u) batch_id++;*/
    
    // acquire lock, or signal that lock was not acquired
    if (h_bindex[batch_id] == cu_batchsize)
    {
        checkCuda(cudaEventSynchronize(h_mem_event[batch_id]));
        h_bindex[batch_id] = 0;
    }

    // store for later processing
    h_batch[batch_id].Ri[h_bindex[batch_id]][0] = (single_t)xi[0];
    h_batch[batch_id].Ri[h_bindex[batch_id]][1] = (single_t)xi[1];
    h_batch[batch_id].Ri[h_bindex[batch_id]][2] = (single_t)xi[2];
    h_batch[batch_id].Rj[h_bindex[batch_id]][0] = (single_t)xj[0];
    h_batch[batch_id].Rj[h_bindex[batch_id]][1] = (single_t)xj[1];
    h_batch[batch_id].Rj[h_bindex[batch_id]][2] = (single_t)xj[2];
    h_batch[batch_id].Fij[h_bindex[batch_id]][0] = (single_t)Fij[0];
    h_batch[batch_id].Fij[h_bindex[batch_id]][1] = (single_t)Fij[1];
    h_batch[batch_id].Fij[h_bindex[batch_id]][2] = (single_t)Fij[2];
    h_bindex[batch_id] += 1;

    // if bindex is cu_batchsize we process
    if (h_bindex[batch_id] == cu_batchsize)
    {
        // transfer the grid to host
        checkCuda(cudaMemcpyAsync(
                &d_batch[batch_id],
                &h_batch[batch_id],
                sizeof(batcharrays_t),
                cudaMemcpyHostToDevice,
                h_stream[batch_id]));
        checkCuda(cudaEventRecord(h_mem_event[batch_id],h_stream[batch_id]));

        // execute with a single element processed by each streaming multiprocessor
        process_batch<<<batch_blocks,batch_threads,0u,h_stream[batch_id]>>>(h_bindex[batch_id], &d_batch[batch_id], d_sum_grid+batch_id*h_ncells);
    }
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
                    sizeof(batcharrays_t),
                    cudaMemcpyHostToDevice,
                    h_stream[i]) );

            // execute with a single element processed by each streaming multiprocessor
            process_batch<<<batch_blocks,batch_threads,0u,h_stream[i]>>>(h_bindex[i], &d_batch[i], d_sum_grid+i*h_ncells);
            
            h_bindex[i] = 0;
        }
    }
    
    // synchronize entire devies
    checkCuda(cudaDeviceSynchronize());

    // sum grids on device and transfer
    uint_t reduce_blocks = h_ncells/cu_threads_per_block;
    reduce_blocks += (reduce_blocks*cu_threads_per_block < h_ncells) ? 1 : 0;
    
    reduce_grids<<<reduce_blocks, cu_threads_per_block>>>(h_nbatches, h_ncells, d_sum_grid);
    //checkCuda(cudaDeviceSynchronize());
    
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
    checkCuda(cudaMemset(d_sum_grid, 0, sizeof(cu_smatrix[h_nbatches*h_ncells])));
}
