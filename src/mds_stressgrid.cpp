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
#include "voro++.hh"

//#define CUSTRESS_ENABLE
//#ifdef CUSTRESS_ENABLE
//#include "mds_custress.h"
//#endif//CUSTRESS_ENABLE

using namespace mds;

template<class matrix_type>
inline void distribute_observable_3d(
        const StressGrid::state_t & state,
        const array3_mds & xi,
        const array3_mds & xj,
        const array3_mds & diff,
        const matrix_type & observable,
        matrix_type * grid)
{
    //------------------------------------------------------------------------------------
    // calculate the grid coordinates (no pbc) for the extreme points
    iarray x = {
        int(state.nxyz[0] * xi[0] * state.invbox[0][0] - (xi[0] < 0.0) ),
        int(state.nxyz[1] * xi[1] * state.invbox[1][1] - (xi[1] < 0.0) ),
        int(state.nxyz[2] * xi[2] * state.invbox[2][2] - (xi[2] < 0.0) )
    };

    const iarray i2 = {
        int(state.nxyz[0] * xj[0] * state.invbox[0][0] - (xj[0] < 0.0) ),
        int(state.nxyz[1] * xj[1] * state.invbox[1][1] - (xj[1] < 0.0) ),
        int(state.nxyz[2] * xj[2] * state.invbox[2][2] - (xj[2] < 0.0) ),
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
        int((i2[0]>x[0])-(x[0]>i2[0]) ),
        int((i2[1]>x[1])-(x[1]>i2[1]) ),
        int((i2[2]>x[2])-(x[2]>i2[2]) )
    };

    iarray xn = {
        int(x[0]+(c[0]+1)/2 ),
        int(x[1]+(c[1]+1)/2 ),
        int(x[2]+(c[2]+1)/2 )
    };

    array3_mds d_cgrid = {
        xi[0]-(x[0]+0.5)*state.gridsp[0],
        xi[1]-(x[1]+0.5)*state.gridsp[1],
        xi[2]-(x[2]+0.5)*state.gridsp[2]
    };

    // calculate parametric time in each dimension, and related constants
    array3_mds t = {
        (c[0] == 0) ? realval_mds(1.1) : t_c1[0]-xn[0]*t_c2[0],
        (c[1] == 0) ? realval_mds(1.1) : t_c1[1]-xn[1]*t_c2[1],
        (c[2] == 0) ? realval_mds(1.1) : t_c1[2]-xn[2]*t_c2[2]
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
    
    const int iterations = c[0]*(i2[0]-x[0]) + c[1]*(i2[1]-x[1]) + c[2]*(i2[2]-x[2]);
    for (int count = 0; count <= iterations; ++count)
    {
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
    
        const int iip1 = ((x[0] + 1 + state.nxyz[0]) % state.nxyz[0])*state.nxyz[1]*state.nxyz[2];
        const int jjp1 = ((x[1] + 1 + state.nxyz[1]) % state.nxyz[1])*state.nxyz[2];
        const int kkp1 = ((x[2] + 1 + state.nxyz[2]) % state.nxyz[2]);
        const int iim1 = ((x[0] + state.nxyz[0]) % state.nxyz[0])*state.nxyz[1]*state.nxyz[2];
        const int jjm1 = ((x[1] + state.nxyz[1]) % state.nxyz[1])*state.nxyz[2];
        const int kkm1 = ((x[2] + state.nxyz[2]) % state.nxyz[2]);

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

        // perform the sums into the grid
        if constexpr (sizeof(matrix_type) == sizeof(matrix3_mds) ) {
            scalesummatrix3(C*( D[0] + D[1] + D[2] + D[3] + D[4] + D[5] + D[6] + D[7]), observable, grid[iip1 + jjp1 + kkp1]);
            scalesummatrix3(C*(-D[0] - D[1] - D[2] + D[3] - D[4] + D[5] + D[6] + D[7]), observable, grid[iip1 + jjp1 + kkm1]);
            scalesummatrix3(C*(-D[0] - D[1] + D[2] - D[3] + D[4] - D[5] + D[6] + D[7]), observable, grid[iip1 + jjm1 + kkp1]);
            scalesummatrix3(C*( D[0] + D[1] - D[2] - D[3] - D[4] - D[5] + D[6] + D[7]), observable, grid[iip1 + jjm1 + kkm1]);
            scalesummatrix3(C*(-D[0] + D[1] - D[2] - D[3] + D[4] + D[5] - D[6] + D[7]), observable, grid[iim1 + jjp1 + kkp1]);
            scalesummatrix3(C*( D[0] - D[1] + D[2] - D[3] - D[4] + D[5] - D[6] + D[7]), observable, grid[iim1 + jjp1 + kkm1]);
            scalesummatrix3(C*( D[0] - D[1] - D[2] + D[3] + D[4] - D[5] - D[6] + D[7]), observable, grid[iim1 + jjm1 + kkp1]);
            scalesummatrix3(C*(-D[0] + D[1] + D[2] + D[3] - D[4] - D[5] - D[6] + D[7]), observable, grid[iim1 + jjm1 + kkm1]);
        } else {
            scalesummatrix6(C*( D[0] + D[1] + D[2] + D[3] + D[4] + D[5] + D[6] + D[7]), observable, grid[iip1 + jjp1 + kkp1]);
            scalesummatrix6(C*(-D[0] - D[1] - D[2] + D[3] - D[4] + D[5] + D[6] + D[7]), observable, grid[iip1 + jjp1 + kkm1]);
            scalesummatrix6(C*(-D[0] - D[1] + D[2] - D[3] + D[4] - D[5] + D[6] + D[7]), observable, grid[iip1 + jjm1 + kkp1]);
            scalesummatrix6(C*( D[0] + D[1] - D[2] - D[3] - D[4] - D[5] + D[6] + D[7]), observable, grid[iip1 + jjm1 + kkm1]);
            scalesummatrix6(C*(-D[0] + D[1] - D[2] - D[3] + D[4] + D[5] - D[6] + D[7]), observable, grid[iim1 + jjp1 + kkp1]);
            scalesummatrix6(C*( D[0] - D[1] + D[2] - D[3] - D[4] + D[5] - D[6] + D[7]), observable, grid[iim1 + jjp1 + kkm1]);
            scalesummatrix6(C*( D[0] - D[1] - D[2] + D[3] + D[4] - D[5] - D[6] + D[7]), observable, grid[iim1 + jjm1 + kkp1]);
            scalesummatrix6(C*(-D[0] + D[1] + D[2] + D[3] - D[4] - D[5] - D[6] + D[7]), observable, grid[iim1 + jjm1 + kkm1]);
        }

        d_cgrid[iX] -= c[iX] * state.gridsp[iX];
        oldt = t[iX];
        
        x[iX] += c[iX];
        xn[iX] += c[iX];

        // Next cross point:
        t[iX] = t_c1[iX]-xn[iX]*t_c2[iX];
    }
}

template<class matrix_type>
inline void distribute_observable_1d(
        const StressGrid::state_t & state,
        const array3_mds & xi,
        const array3_mds & xj,
        const array3_mds & diff,
        const matrix_type & observable,
        matrix_type * grid)
{
    // calculate the grid coordinates (no pbc) for the extreme points
    int x = state.nxyz[state.griddim] * xi[state.griddim] * state.invbox[state.griddim][state.griddim] - (xi[state.griddim] < 0.0);
    const int i2 = state.nxyz[state.griddim] * xj[state.griddim] * state.invbox[state.griddim][state.griddim] - (xj[state.griddim] < 0.0);
    const int c = (i2>x)-(x>i2);
    const real_mds t_c1 = xi[state.griddim] / (xi[state.griddim]-xj[state.griddim]);
    const real_mds t_c2 = state.gridsp[state.griddim] / (xi[state.griddim]-xj[state.griddim]);
    const real_mds C = realval_mds(0.5)*state.invgridsp*state.invgridsp;
    real_mds d_cgrid = xi[state.griddim]-(x+realval_mds(0.5))*state.gridsp[state.griddim];
    int xn = x+(c+1)/2;

    // track previous time of crossing and check that sum is complete (?)
    real_mds oldt = 0.0; 

    // fix the number of iterations
    const int iterations = c*(i2-x);
    for (int count = 0; count <= iterations; ++count)
    {
        // there is always iterations+1, where the last iteration deals
        // with any residual
        real_mds newt = (iterations == count) ? realval_mds(1.0) : t_c1-xn*t_c2;

        // work out the parametric time constants
        const real_mds t12 = oldt*oldt;
        const real_mds t22 = newt*newt;
        const real_mds dt1 = newt - oldt;
        const real_mds dt2 = t22 - t12;

        const int p1 = ((x + 1 + state.nxyz[state.griddim]) % state.nxyz[state.griddim]);
        const int m1 = ((x + state.nxyz[state.griddim]) % state.nxyz[state.griddim]);

        // the composite constants in terms of i, j, k
        const real_mds D1 = state.gridsp[5-state.griddim]*(realval_mds(2.0)*d_cgrid*dt1+diff[state.griddim]*dt2);
        const real_mds D2 = state.gridsp[6]*dt1;
        if constexpr (sizeof(matrix_type) == sizeof(matrix3_mds) ) {
            scalesummatrix3(C*( D1 + D2), observable, grid[p1]);
            scalesummatrix3(C*(-D1 + D2), observable, grid[m1]);
        } else {
            scalesummatrix6(C*( D1 + D2), observable, grid[p1]);
            scalesummatrix6(C*(-D1 + D2), observable, grid[m1]);
        }

        
        d_cgrid -= c * state.gridsp[state.griddim];
        oldt = newt;
        
        x += c;
        xn += c;
    }
}


//Constructor
StressGrid::StressGrid() :
    state({0}),
    alloc({0}),
    m_filename(),
    m_max_threads(0),
    m_thread_map(),
    m_mutex_state()
{
    this->state.griddim = mds_griddim_xyz;
    this->state.maxClust = mds_maxpart;
}

//Destructor
StressGrid::~StressGrid()
{
    this->Clear();
}

//This function initialize the grid depending on the settings. If the settings are incorrect, 
//it throws an error
void StressGrid::Init()
{
    if (true == this->state.disable) return;
    std::lock_guard<std::mutex> lock(this->m_mutex_state);
    
    // Check some common settings before attempting initialization
    this->state.ierr = MDS_OK;
    if (mds_spat == this->state.spatatom) {
        // grid size must always be greater than or equal to 0
        this->state.ierr |= checkError(
                this->state.nxyz[0] < 0,
                "SetNumberOfGridCellsX(int) - int must not be less than 0");
        this->state.ierr |= checkError(
                this->state.nxyz[1] < 0,
                "SetNumberOfGridCellsY(int) - int must not be less than 0");
        this->state.ierr |= checkError(
                this->state.nxyz[2] < 0,
                "SetNumberOfGridCellsZ(int) - int must not be less than 0");

        // if grid size is 0 in all dimensions, we will use spacing to generate grid size
        // and so we check it here
        if (0 == this->state.nxyz[0] && 0 == this->state.nxyz[1] && 0 == this->state.nxyz[2]) {
            this->state.ierr |= checkError(
                    this->state.spacing < mds_eps,
                    "SetSpacing(float) - float must be greater than mds_eps");
            this->state.ierr |= checkError(
                    iszeromatrix3(this->state.box),
                    "SetBox(mat3,int) - all mat3 elements must be greater than mds_eps");
        }
        
        // mds_spat requires a force decomposition algorithm be selected
        this->state.ierr |= checkError(
                mds_ccfd != this->state.fdecomp &&
                mds_ncfd != this->state.fdecomp &&
                mds_gld  != this->state.fdecomp,
                "SetForceDecomposition(type) - type must be mds_{ccfd, ncfd or gld}");
    } else if (mds_atom == this->state.spatatom) {
        this->state.ierr |= checkError(
                this->state.nAtoms <= 0,
                "SetNumberOfAtoms(int) - int must be greater than 0");
    } else /* not mds_spat or mds_atom */ {
        this->state.ierr |= checkError(
                true,
                "SetStressType(type) - type must be mds_spat or mds_atom");
    }

    // parameters unrelated to stress algorithm and md
    this->state.ierr |= checkError(
            0 == this->m_filename.size(),
            "SetFileName(string) - len(string) must be greater than 0");
    this->state.ierr |= checkError(
            this->state.maxClust <= 1,
            "SetMaxCluster(int) - int must be greater than 1");

    if (MDS_OK == this->state.ierr) {
        // Clear all allocations, keeping settings
        this->Clear();
    
        // Set grid sizes before allocation
        if (this->state.spatatom == mds_spat) {
            if(this->state.nxyz[0] == 0)
                this->state.nxyz[0] = static_cast<int>(this->state.box[0][0]/this->state.spacing);
            if(this->state.nxyz[1] == 0)
                this->state.nxyz[1] = static_cast<int>(this->state.box[1][1]/this->state.spacing);
            if(this->state.nxyz[2] == 0)
                this->state.nxyz[2] = static_cast<int>(this->state.box[2][2]/this->state.spacing);
            
            this->state.ncells = this->state.nxyz[0]*this->state.nxyz[1]*this->state.nxyz[2];

            if (this->state.ncells == this->state.nxyz[0]) {
                this->state.griddim = mds_griddim_xxx;
            } else if (this->state.ncells == this->state.nxyz[1]) {
                this->state.griddim = mds_griddim_yyy;
            } else if (this->state.ncells == this->state.nxyz[2]) {
                this->state.griddim = mds_griddim_zzz;
            } else {
                this->state.griddim = mds_griddim_xyz;
            }
        } else {
            this->state.ncells = this->state.nAtoms;

            // create the molecule_id array
            this->alloc.molecule_id = new int[this->state.ncells];
            
            // create the radii array
            this->alloc.radii = new real_mds[this->state.ncells];
            this->alloc.positions = new real_mds[3*this->state.ncells];
            this->alloc.sum_volume = new real_mds[this->state.ncells];

            // set them to defaults
            for ( int i=0; i < this->state.ncells; ++i ) {
                this->alloc.molecule_id[i] = 0;
                this->alloc.radii[i] = realval_mds(0.001);
                this->alloc.sum_volume[i] = realval_mds(0.0);
        
                this->alloc.positions[3*i] = realval_mds(0.0);
                this->alloc.positions[3*i+1] = realval_mds(0.0);
                this->alloc.positions[3*i+2] = realval_mds(0.0);
            }
        }

        // Allocate all space necessary for grids
        this->alloc.sum_grid             = new matrix3_mds [this->state.ncells];
        this->alloc.avg_grid             = new matrix3_mds [this->state.ncells];
        this->alloc.avg_gridtot          = new matrix3_mds [1];
        this->alloc.sum_grid_elcovar     = new matrix6_mds [this->state.ncells];
        this->alloc.sum_grid_elkin       = new matrix6_mds [this->state.ncells];
        this->alloc.sum_grid_elborn      = new matrix6_mds [this->state.ncells];
        this->alloc.sum_grid_volcovar    = new matrix3_mds [this->state.ncells];
        this->alloc.sum_gridtot_volcovar = new matrix3_mds [1];
        this->alloc.current_grid_elkin   = new matrix6_mds [this->state.ncells*this->m_max_threads];
        this->alloc.current_grid_elborn  = new matrix6_mds [this->state.ncells*this->m_max_threads];
        this->alloc.current_grid         = new matrix3_mds [this->state.ncells*this->m_max_threads];
        this->alloc.current_gridtot      = new matrix3_mds [1];

        // Zero all allocated buffers and counters
        this->state.nframes = 0;
        this->state.avg_boxvol = realval_mds(0.0);
        this->state.var_boxvol = realval_mds(0.0);
        zeromatrix3(this->alloc.current_gridtot[0]);
        zeromatrix3(this->alloc.avg_gridtot[0]);
        zeromatrix3(this->alloc.sum_gridtot_volcovar[0]);
        for (int i=0; i < this->state.ncells; i++) {
            zeromatrix3(this->alloc.sum_grid[i]);
            zeromatrix3(this->alloc.avg_grid[i]);
            zeromatrix3(this->alloc.sum_grid_volcovar[i]);
            zeromatrix6(this->alloc.sum_grid_elcovar[i]);
            zeromatrix6(this->alloc.sum_grid_elborn[i]);
            zeromatrix6(this->alloc.sum_grid_elkin[i]);
        }
        for (int i=0; i < this->state.ncells*this->m_max_threads; i++) {
            zeromatrix3(this->alloc.current_grid[i]);
            zeromatrix6(this->alloc.current_grid_elborn[i]);
            zeromatrix6(this->alloc.current_grid_elkin[i]);
        }

        // Create the lapack objects to deal with linear solvers and projections
        this->alloc.lapack = new Lapack*[m_max_threads];
        for (int i=0; i <this->m_max_threads; ++i) {
            this->alloc.lapack[i] = new Lapack (
                    mds_ndim*this->state.maxClust,(this->state.maxClust*(this->state.maxClust-1))/2);
        }

#ifdef CUSTRESS_ENABLE
        if (this->state.cuda) {
            custress_init(
                    this->m_max_threads,
                    this->state.ncells,
                    this->state.nxyz[0],
                    this->state.nxyz[1],
                    this->state.nxyz[2]);
        }
#endif//CUSTRESS_ENABLE

        // Let us know what precision is being used
        std::cout << "STRESSLIB: Internal Floating Points Precision - " << sizeof(real_mds) << " Bytes" << std::endl;
        std::cout << "STRESSLIB: External Floating Points Precision - " << sizeof(real_ext) << " Bytes" << std::endl;
        std::cout << "STRESSLIB: Output Floating Points Precision - " << sizeof(real_out) << " Bytes" << std::endl;

        // we have successfully initialized
        this->state.initialized = true;
    }
}

void StressGrid::SetPeriodicBoundaries(bool x, bool y, bool z, bool enforce)
{ 
    if (true == this->state.disable) return;
    std::lock_guard<std::mutex> lock(m_mutex_state);

#ifdef CUSTRESS_ENABLE
    if (this->state.cuda)
        custress_set_periodic(x, y, z, enforce);
#endif//CUSTRESS_ENABLE

    this->state.periodic[0] = x;
    this->state.periodic[1] = y;
    this->state.periodic[2] = z;
    this->state.periodic[3] = enforce;
}

// This function updates the box, invbox and computes the new spacings.
void StressGrid::UpdateBoxSpacings ( matrix3_ext box )
{
    if (true == this->state.disable)
        return;

    if (MDS_OK == this->state.ierr) {
        // every thread must process this latch before proceeding
        //static barrier ubs_entry(this->m_max_threads);
        //ubs_entry.count_down_and_wait();

        // only thread 0 performs update
        if (this->m_thread_map[std::this_thread::get_id()] == 0) {
            copymatrix3( box, this->state.box);
            inversematrix3( this->state.box, this->state.invbox );

            this->state.gridsp[0] = this->state.box[0][0]/static_cast<real_mds>(this->state.nxyz[0]);
            this->state.gridsp[1] = this->state.box[1][1]/static_cast<real_mds>(this->state.nxyz[1]);
            this->state.gridsp[2] = this->state.box[2][2]/static_cast<real_mds>(this->state.nxyz[2]);
            this->state.gridsp[3] = this->state.gridsp[0]*this->state.gridsp[1];
            this->state.gridsp[4] = this->state.gridsp[0]*this->state.gridsp[2];
            this->state.gridsp[5] = this->state.gridsp[1]*this->state.gridsp[2];
            this->state.gridsp[6] = this->state.gridsp[0]*this->state.gridsp[1]*this->state.gridsp[2];
            this->state.invgridsp =  realval_mds(1.0)/(this->state.gridsp[0]*this->state.gridsp[1]*this->state.gridsp[2]);

            //this->state.sum_boxvol = this->state.sum_boxvol + box[0][0]*box[1][1]*box[2][2];
#ifdef CUSTRESS_ENABLE
            if (this->state.cuda)
                custress_update_box_spacings(this->state.box, this->state.invbox, this->state.gridsp);
#endif//CUSTRESS_ENABLE
        }
    }
}

// This function sums the current grid to sum_grid and sets current_grid
// to zero.
void StressGrid::SumGrid ( )
{
    if (true == this->state.disable)
        return;

    if (MDS_OK == this->state.ierr)
    {
        // get the thread id
        int thread_id = this->m_thread_map[std::this_thread::get_id()];

        // every thread must process this latch before proceeding
        static barrier sumgrid_enter_stress_reduction(this->m_max_threads);
        sumgrid_enter_stress_reduction.count_down_and_wait();
        
        // reduce all stress current grids
        for (int i = thread_id; i < this->state.ncells; i+=this->m_max_threads)
        {
            if (i < this->state.ncells)
            {
                for (int j = 1; j < this->m_max_threads; ++j)
                {
                    summatrix3(this->alloc.current_grid[i], this->alloc.current_grid[i+j*this->state.ncells], this->alloc.current_grid[i] );
                    zeromatrix3(this->alloc.current_grid[i+j*this->state.ncells]);
                    summatrix6(this->alloc.current_grid_elborn[i], this->alloc.current_grid_elborn[i+j*this->state.ncells], this->alloc.current_grid_elborn[i] );
                    zeromatrix6(this->alloc.current_grid_elborn[i+j*this->state.ncells]);
                    summatrix6(this->alloc.current_grid_elkin[i], this->alloc.current_grid_elkin[i+j*this->state.ncells], this->alloc.current_grid_elkin[i] );
                    zeromatrix6(this->alloc.current_grid_elkin[i+j*this->state.ncells]);
                }
            }
        }

        // every thread must process this latch before proceeding
        static barrier sumgrid_continue(this->m_max_threads);
        sumgrid_continue.count_down_and_wait();

        if (thread_id == 0)
        {
#ifdef CUSTRESS_ENABLE
            if (this->state.cuda)
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

            if (this->state.spatatom == mds_spat) {
                for (int i = 0; i < this->state.ncells; i++)
                    summatrix3(this->alloc.current_gridtot[0], this->alloc.current_grid[i], this->alloc.current_gridtot[0]); // compute y = sigma_total_kl = sum(sigma_local_kl)/this->state.ncells

                scalematrix3(this->alloc.current_gridtot[0], realval_mds(1.0)/this->state.ncells, this->alloc.current_gridtot[0]); // scale by this->state.ncells
                scalesummatrix3(realval_mds(-1.0), this->alloc.avg_gridtot[0], this->alloc.current_gridtot[0]); // subtract meany from y and store back into y (which now becomes dy)
                scalesummatrix3(realval_mds(1.0)/this->state.nframes, this->alloc.current_gridtot[0], this->alloc.avg_gridtot[0]); //compute meany
                scalesummatrix3(deltaV, this->alloc.current_gridtot[0], this->alloc.sum_gridtot_volcovar[0]); // accumulate the covar(sigma_total_kl, Vol)

                matrix6_mds tmp_covar[1];
                matrix3_mds dx[1];
                for (int i = 0; i < this->state.ncells; i++) {
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
            } else {
                // initialize the voro container
                array3_mds vor_box;
                vor_box[0] = (real_mds)this->state.box[0][0];
                vor_box[1] = (real_mds)this->state.box[1][1];
                vor_box[2] = (real_mds)this->state.box[2][2];

                real_mds gfxy,gfxz;
                gfxy = vor_box[1]/vor_box[0];
                gfxz = vor_box[2]/vor_box[0];

                // need to count the number of sites without radius 0.0
                int vcells = 0;
                for (int i = 0; i < this->state.ncells; ++i) {
                    if (this->alloc.radii[i] > 0.0)
                        vcells += 1;
                }

                int gridn[3];
                gridn[0] = pow(this->state.ncells/(3*gfxy*gfxz), 1/3.0);
                gridn[1] = gridn[0]*gfxy;
                gridn[2] = gridn[0]*gfxz;

                // create the voronoi objects
                voro::particle_order vorpo = voro::particle_order(this->state.ncells);
                voro::container_poly vorcon = voro::container_poly(0.0, vor_box[0], 0.0, vor_box[1], 0.0, vor_box[2], gridn[0], gridn[1], gridn[2], this->state.periodic[0], this->state.periodic[1], this->state.periodic[2], 8);

                // fill the container
                int voro_cells = 0;
                for (int i = 0; i < this->state.ncells; ++i) {
                    if (this->alloc.radii[i] > 0.0) {
                        voro_cells += 1;
                        double px = (double)this->alloc.positions[3*i];
                        double py = (double)this->alloc.positions[3*i+1];
                        double pz = (double)this->alloc.positions[3*i+2];

                        // we are scaling the radii down to 1/100th the size here so that the number of voronoi cells is the same as
                        // the number of atoms with non-zero radii. This is because voro++ does a check and if two particles are within
                        // a distance less than the sum of their radii, they are put in the same voronoi cell, which does not allow
                        // the calculation of volumes of every particle in the system
                        vorcon.put(vorpo, i, px, py, pz, 0.01*this->alloc.radii[i]);
                    }
                }

                voro::voronoicell c;
                voro::c_loop_order vl(vorcon, vorpo);

                // the particle/cell id
                int pid = 0;

                // track the particle/cell id of atom with largest volume in this molecule
                int this_molecule = this->alloc.molecule_id[pid];
                int last_pid = 0;
                real_mds last_volume = realval_mds(0.0);

                int cells_computed = 0;
                if (vl.start()) {
                    do {
                        if (this->alloc.radii[pid] > 0.0 && vorcon.compute_cell(c,vl)) {
                            // count the cells
                            cells_computed += 1;

                            // get the volume of this cell
                            last_volume = (real_mds)(c.volume());
                            last_pid = pid;

                            // mark the last volume encountered
                            this_molecule = this->alloc.molecule_id[pid];

                            scalematrix3(this->alloc.current_grid[pid], realval_mds(1.0)/last_volume, this->alloc.current_grid[pid] );
                            summatrix3(this->alloc.sum_grid[pid], this->alloc.current_grid[pid], this->alloc.sum_grid[pid] );

                            // add the volume
                            this->alloc.sum_volume[pid] += last_volume;
                        } else {
                            // check that we are working on the same molecule
                            if (this_molecule != this->alloc.molecule_id[pid]) {
                                this->state.ierr |= checkError(true,
                                        "ERROR:: radius of zero encountered, but last atom molecule was not this molecule");
                            } else {
                                scalematrix3( this->alloc.current_grid[pid], realval_mds(1.0)/last_volume, this->alloc.current_grid[pid] );
                                summatrix3( this->alloc.sum_grid[last_pid], this->alloc.current_grid[pid], this->alloc.sum_grid[last_pid] );
                            }
                        }

                        // always increment the particle count
                        pid += 1;
                    } while(vl.inc());

                    // should do an error check here
                    this->state.ierr |= checkError(
                            pid != this->state.nAtoms,
                            "ERROR:: number of atoms processed does not match number of sites");
                }
            }

            zeromatrix3(this->alloc.current_gridtot[0]);
            for(int i = 0; i < this->state.ncells; ++i) {
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

void StressGrid::DispersionCorrection (real_ext shift)
{
    if (true == this->state.disable)
        return;

    if (this->state.nodispcor == false) {
        // select the correct grid
        int batch_id = this->m_thread_map[std::this_thread::get_id()];
        matrix3_mds * grid = this->alloc.current_grid+batch_id*this->state.ncells;

        // add shift to grid
        for(int i = 0; i < this->state.ncells; i++) {
            for(int m = 0; m < 3; m++)
                grid[i][m][m] += shift;
        }
    }
}

//Set both sum_grid and current_grid to zero. Sum the number of resets (this is used for 
//printing files) and set the number of frames to zero
void StressGrid::Reset ( )
{
    if (true == this->state.disable)
        return;

    if (MDS_OK == this->state.ierr)
    {
        // get the thread id
        int thread_id = this->m_thread_map[std::this_thread::get_id()];

        // every thread must process this latch before proceeding
        static barrier reset_entry(this->m_max_threads);
        reset_entry.count_down_and_wait();

        // every thread zeros its own current grid
        for( int i=0; i<this->state.ncells; i++ ) {
            zeromatrix3(this->alloc.current_grid[i+thread_id*this->m_max_threads]);
            zeromatrix6(this->alloc.current_grid_elborn[i+thread_id*this->m_max_threads]);
            zeromatrix6(this->alloc.current_grid_elkin[i+thread_id*this->m_max_threads]);
        }

        if (thread_id == 0) {
            // thread 0 deals with the sum grid
            zeromatrix3(this->alloc.current_gridtot[0]);
            zeromatrix3(this->alloc.avg_gridtot[0]);
            for( int i=0; i<this->state.ncells; i++ ) {
                zeromatrix3( this->alloc.sum_grid[i] );
                zeromatrix3( this->alloc.avg_grid[i] );
                zeromatrix3( this->alloc.sum_grid_volcovar[i] );
                zeromatrix6( this->alloc.sum_grid_elborn[i] );
                zeromatrix6( this->alloc.sum_grid_elcovar[i] );
                zeromatrix6( this->alloc.sum_grid_elkin[i] );
            }

            if (this->state.spatatom == mds_atom) {
                for( int i=0; i<this->state.ncells; i++ )
                    this->alloc.sum_volume[i] = 0.0;
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


//Writes file with average stress to grid using the filename set by the user
void StressGrid::Write ( )
{
    if (true == this->state.disable)
        return;

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
            outname = outname + "." + mds_fileext;

        elcovar_outname = rawname + "_elcovar.dat" + outnumber.str();
        elkin_outname = rawname + "_elkin.dat" + outnumber.str();
        elborn_outname = rawname + "_elborn.dat" + outnumber.str();
        eltotal_outname = rawname + "_eltotal.dat" + outnumber.str();
        eltotalhooke_outname = rawname + "_eltotalhooke.dat" + outnumber.str();

        // open the main output file
        outfile = fopen(outname.c_str(), "wb" );
        if (this->state.spatatom == mds_spat) {
            Dtype = 1;
        } else if (this->state.spatatom == mds_atom) {
            Dtype = 2;
        }
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

        if (this->state.spatatom == mds_spat) {
            // this is the number of grids in all dimensions, also to all files
            fwrite(&this->state.nxyz[0], sizeof(this->state.nxyz[0]), 1, outfile);
            fwrite(&this->state.nxyz[1], sizeof(this->state.nxyz[1]), 1, outfile);
            fwrite(&this->state.nxyz[2], sizeof(this->state.nxyz[2]), 1, outfile);
            fwrite(&this->state.nxyz[0], sizeof(this->state.nxyz[0]), 1, elcovar_outfile);
            fwrite(&this->state.nxyz[1], sizeof(this->state.nxyz[1]), 1, elcovar_outfile);
            fwrite(&this->state.nxyz[2], sizeof(this->state.nxyz[2]), 1, elcovar_outfile);
            fwrite(&this->state.nxyz[0], sizeof(this->state.nxyz[0]), 1, elborn_outfile);
            fwrite(&this->state.nxyz[1], sizeof(this->state.nxyz[1]), 1, elborn_outfile);
            fwrite(&this->state.nxyz[2], sizeof(this->state.nxyz[2]), 1, elborn_outfile);
            fwrite(&this->state.nxyz[0], sizeof(this->state.nxyz[0]), 1, elkin_outfile);
            fwrite(&this->state.nxyz[1], sizeof(this->state.nxyz[1]), 1, elkin_outfile);
            fwrite(&this->state.nxyz[2], sizeof(this->state.nxyz[2]), 1, elkin_outfile);
            fwrite(&this->state.nxyz[0], sizeof(this->state.nxyz[0]), 1, eltotal_outfile);
            fwrite(&this->state.nxyz[1], sizeof(this->state.nxyz[1]), 1, eltotal_outfile);
            fwrite(&this->state.nxyz[2], sizeof(this->state.nxyz[2]), 1, eltotal_outfile);
            fwrite(&this->state.nxyz[0], sizeof(this->state.nxyz[0]), 1, eltotalhooke_outfile);
            fwrite(&this->state.nxyz[1], sizeof(this->state.nxyz[1]), 1, eltotalhooke_outfile);
            fwrite(&this->state.nxyz[2], sizeof(this->state.nxyz[2]), 1, eltotalhooke_outfile);
        } else {
            // if we are using mds_atom, then there is only ncells == natoms
            fwrite(&this->state.ncells, sizeof(int), 1, outfile);
            fwrite(&this->state.ncells, sizeof(int), 1, elcovar_outfile);
            fwrite(&this->state.ncells, sizeof(int), 1, elborn_outfile);
            fwrite(&this->state.ncells, sizeof(int), 1, elkin_outfile);
            fwrite(&this->state.ncells, sizeof(int), 1, eltotal_outfile);
            fwrite(&this->state.ncells, sizeof(int), 1, eltotalhooke_outfile);
        }

        // calculate stress factors
        real_mds stressfac, covfac, covfac2;
        //stressfac = real_mds(mds_units)/this->state.nframes;
        stressfac = real_mds(mds_units);
        covfac = -real_mds(mds_units)*real_mds(mds_units)*this->state.avg_boxvol/(this->state.temperature*real_mds(KBfac)*this->state.nframes);
        //covfac /= this->state.ncells; // uncomment this for for local-vs-local fluctuations
        covfac2 = realval_mds(0.0);
        if (this->state.pcoupl == true)
            covfac2 = real_mds(mds_units)*real_mds(mds_units)*this->state.avg_boxvol/(this->state.temperature*real_mds(KBfac)*this->state.var_boxvol*this->state.nframes);

        // need to store matrices in double precision
        matrix3_out sum_grid;
        matrix6_out sum_grid_elcovar;
        matrix6_out sum_grid_elborn;
        matrix6_out sum_grid_elkin;
        matrix3_out s;
        matrix6_out elast;

        // zero them
        zeromatrix3(sum_grid);
        zeromatrix3(s);
        zeromatrix6(sum_grid_elcovar);
        zeromatrix6(sum_grid_elborn);
        zeromatrix6(sum_grid_elkin);
        zeromatrix6(elast);

        // need this for corrections below
        matrix6_mds npt_covar_corr[1];
        for ( int i = 0; i < this->state.ncells; i++ )
        {
            //scalematrix3(this->alloc.sum_grid[i], stressfac, this->alloc.sum_grid[i]);
            scalematrix3(this->alloc.avg_grid[i], stressfac, this->alloc.avg_grid[i]); // Use the online average stress instead of the cummulative stress divided by the number of frames
            scalematrix6(this->alloc.sum_grid_elcovar[i], covfac, this->alloc.sum_grid_elcovar[i]);
            scalematrix6(this->alloc.sum_grid_elborn[i], stressfac/this->state.nframes, this->alloc.sum_grid_elborn[i]);
            scalematrix6(this->alloc.sum_grid_elkin[i], stressfac/this->state.nframes, this->alloc.sum_grid_elkin[i]);

            if (this->state.pcoupl == true)
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

        // append the volume data
        if (this->state.spatatom == mds_atom)
        {
            for ( int i = 0; i < this->state.ncells; i++ )
            {
                real_out sum_volume = (real_out)(this->alloc.sum_volume[i] / this->state.nframes);
                fwrite(&sum_volume, sizeof(real_out), 1, outfile);
            }
        }

        fclose(outfile);
        fclose(elcovar_outfile);
        fclose(elborn_outfile);
        fclose(elkin_outfile);
        for ( int i = 0; i < this->state.ncells; i++ )
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

//----------------------------------------------------------------------------------------
// SetVoronoiRadius
//
// Adds a particle position to the voronoi container
// Requires:
// radius  -> radius of the atom
// atomID  -> label of the atom
void StressGrid::SetVoronoiRadius(real_ext radius, int atomID)
{
    if (true == this->state.disable) return;
    std::lock_guard<std::mutex> lock(this->m_mutex_state);

    // quick check if we have the correct number of atoms
    this->state.ierr |= checkError(atomID > this->state.ncells,
            "ERROR:: atomID is greater than number of cells");

    if (MDS_OK == this->state.ierr)
    {
        // setup the molecule id
        this->alloc.radii[atomID] = std::max(radius,realval_ext(0.001));
    }
}

//----------------------------------------------------------------------------------------
// AddVoronoiAtom
//
// Adds a particle position to the voronoi container
// Requires:
// px      -> position in the x dimension
// py      -> position in the y dimension
// pz      -> position in the z dimension
// radius  -> radius of the atom
// atomID  -> label of the atom
// moleID  -> label of the molecule this atom belongs to
void StressGrid::AddVoronoiAtom(real_ext px, real_ext py, real_ext pz, int atomID, int moleID)
{
    if (true == this->state.disable) return;
    std::lock_guard<std::mutex> lock(this->m_mutex_state);

    // quick check if we have the correct number of atoms
    this->state.ierr |= checkError(atomID > this->state.ncells,
            "ERROR:: atomID is greater than number of cells");

    if (MDS_OK == this->state.ierr)
    {
        // set the positions
        this->alloc.positions[3*atomID] = px;
        this->alloc.positions[3*atomID+1] = py;
        this->alloc.positions[3*atomID+2] = pz;

        // set the molecular id
        this->alloc.molecule_id[atomID] = moleID;
    }
}

//----------------------------------------------------------------------------------------
// DistributeInteraction
//
// ROOT OF ALL EVIL
// This function reads the number of atoms, their
// respective positions and forces, and the atom IDs, and calls the functions in charge of distributing the stress
// on the grid depending on the local stress flags and the kind of interaction
// Requires:
// nAtoms  -> number of atoms of the contribution
// R       -> positions of the atoms
// F       -> forces on the atoms
// atomIDs -> labels of the atoms
void StressGrid::DistributeInteraction(int nAtoms, array3_ext *R, array3_ext *F, int *atomIDs = nullptr)
{
    if (true == this->state.disable)
        return;

    int    n;
    int    i,j;
    real_mds temp;
    matrix3_mds stress;

    int batch_id = this->m_thread_map[std::this_thread::get_id()];
    this->state.ierr |= checkError( nAtoms > this->state.maxClust,
        "Distribute Interaction has been called with a number of atoms larger than the maximum cluster");
    
    if (MDS_OK == this->state.ierr)
    {
        // If spatatom==mds_spat distribute the stress spatially following Noll's procedure
        if (this->state.spatatom == mds_spat)
        {
            // Depending on the number of atoms, call a different function
            if (2 == nAtoms) {
                const array3_mds Ra = {(real_mds)R[0][0], (real_mds)R[0][1], (real_mds)R[0][2]};
                const array3_mds Rb = {(real_mds)R[1][0], (real_mds)R[1][1], (real_mds)R[1][2]};
                const array3_mds Fa = {(real_mds)F[0][0], (real_mds)F[0][1], (real_mds)F[0][2]};
                this->DistributePairInteraction(Ra, Rb, Fa, batch_id );
            } else if (3 == nAtoms) {
                const array3_mds Ra = {(real_mds)R[0][0], (real_mds)R[0][1], (real_mds)R[0][2]};
                const array3_mds Rb = {(real_mds)R[1][0], (real_mds)R[1][1], (real_mds)R[1][2]};
                const array3_mds Rc = {(real_mds)R[2][0], (real_mds)R[2][1], (real_mds)R[2][2]};
                const array3_mds Fa  = {(real_mds)F[0][0], (real_mds)F[0][1], (real_mds)F[0][2]};
                const array3_mds Fb  = {(real_mds)F[1][0], (real_mds)F[1][1], (real_mds)F[1][2]};
                const array3_mds Fc  = {(real_mds)F[2][0], (real_mds)F[2][1], (real_mds)F[2][2]};
                this->DistributeN3(Ra, Rb, Rc, Fa, Fb, Fc, batch_id);
            } else if (-3 == nAtoms) {
                const array3_mds Ra = {(real_mds)R[0][0], (real_mds)R[0][1], (real_mds)R[0][2]};
                const array3_mds Rb = {(real_mds)R[1][0], (real_mds)R[1][1], (real_mds)R[1][2]};
                const array3_mds Rc = {(real_mds)R[2][0], (real_mds)R[2][1], (real_mds)R[2][2]};
                const array3_mds Fa  = {(real_mds)F[0][0], (real_mds)F[0][1], (real_mds)F[0][2]};
                const array3_mds Fb  = {(real_mds)F[1][0], (real_mds)F[1][1], (real_mds)F[1][2]};
                const array3_mds Fc  = {(real_mds)F[2][0], (real_mds)F[2][1], (real_mds)F[2][2]};
                this->DistributeSettle(Ra, Rb, Rc, Fa, Fb, Fc, batch_id);
            } else if (4 == nAtoms) {
                const array3_mds Ra = {(real_mds)R[0][0], (real_mds)R[0][1], (real_mds)R[0][2]};
                const array3_mds Rb = {(real_mds)R[1][0], (real_mds)R[1][1], (real_mds)R[1][2]};
                const array3_mds Rc = {(real_mds)R[2][0], (real_mds)R[2][1], (real_mds)R[2][2]};
                const array3_mds Rd = {(real_mds)R[3][0], (real_mds)R[3][1], (real_mds)R[3][2]};
                const array3_mds Fa  = {(real_mds)F[0][0], (real_mds)F[0][1], (real_mds)F[0][2]};
                const array3_mds Fb  = {(real_mds)F[1][0], (real_mds)F[1][1], (real_mds)F[1][2]};
                const array3_mds Fc  = {(real_mds)F[2][0], (real_mds)F[2][1], (real_mds)F[2][2]};
                const array3_mds Fd  = {(real_mds)F[3][0], (real_mds)F[3][1], (real_mds)F[3][2]};
                this->DistributeN4(Ra, Rb, Rc, Rd, Fa, Fb, Fc, Fd, batch_id);
            } else if (5 == nAtoms) {
                const array3_mds Ra = {(real_mds)R[0][0], (real_mds)R[0][1], (real_mds)R[0][2]};
                const array3_mds Rb = {(real_mds)R[1][0], (real_mds)R[1][1], (real_mds)R[1][2]};
                const array3_mds Rc = {(real_mds)R[2][0], (real_mds)R[2][1], (real_mds)R[2][2]};
                const array3_mds Rd = {(real_mds)R[3][0], (real_mds)R[3][1], (real_mds)R[3][2]};
                const array3_mds Rf = {(real_mds)R[4][0], (real_mds)R[4][1], (real_mds)R[4][2]};
                const array3_mds Fa  = {(real_mds)F[0][0], (real_mds)F[0][1], (real_mds)F[0][2]};
                const array3_mds Fb  = {(real_mds)F[1][0], (real_mds)F[1][1], (real_mds)F[1][2]};
                const array3_mds Fc  = {(real_mds)F[2][0], (real_mds)F[2][1], (real_mds)F[2][2]};
                const array3_mds Fd  = {(real_mds)F[3][0], (real_mds)F[3][1], (real_mds)F[3][2]};
                const array3_mds Ff  = {(real_mds)F[4][0], (real_mds)F[4][1], (real_mds)F[4][2]};
                this->DistributeN5(Ra, Rb, Rc, Rd, Rf, Fa, Fb, Fc, Fd, Ff, batch_id );
            } else {
                this->DistributeNBody( nAtoms, R, F, true, batch_id );
            }
        } else if (this->state.spatatom == mds_atom) {
            // This is because SETTLE calls the function with nAtoms=-3
            if (nAtoms < 0) nAtoms = -nAtoms;

            if (atomIDs == nullptr) {
                std::cout << "ERROR:: the atomIDs array is nullptr. Cannot calculate the stress/atom.";
                return;
            }

            for (n = 0; n < nAtoms; n++) {
                if (atomIDs[n] >= this->state.nAtoms) {
                    std::cout << "ERROR:: the atom label" << atomIDs[n] << "is equal or larger than the total number of atoms" << this->state.nAtoms;
                    return;
                }
            }

            //Initialize the value of the (local) stress to 0
            for(i = 0; i< mds_ndim; i++) {
                stress[i][i] = 0.0;
                for(j=i+1; j< mds_ndim; j++) {
                    stress[i][j] = 0.0;
                    stress[j][i] = 0.0;
                }
            }


            //Calculate the stress
            for(n = 0; n < nAtoms; n++) {
                for(i = 0; i< mds_ndim; i++) {
                    temp = -(F[n][i] * R[n][i])/nAtoms;
                    stress[i][i] += temp;
                    for(j=i+1; j< mds_ndim; j++) {
                        temp = -(F[n][i] * R[n][j])/nAtoms;
                        stress[i][j] += temp;
                        stress[j][i] += temp;
                    }
                }
            }

            for (n = 0; n < nAtoms; n++) {
                summatrix3(this->alloc.current_grid[atomIDs[n]],stress,this->alloc.current_grid[atomIDs[n]]);
            }
        }
    }

    return;
}

//----------------------------------------------------------------------------------------
// ComputeNbodyPairForces
//
// This function reads the number of atoms, the atoms' labels and their
// respective positions and forces, and computes the N-body pairwise forces and vectors
// without distributing the stress
// nAtoms  -> number of atoms of the contribution
// R       -> positions of the atoms
// F       -> forces on the atoms
void StressGrid::ComputeNbodyPairForces(int nAtoms, array3_ext *R, array3_ext *F, int *atomIDs = nullptr)
{
    if (true == this->state.disable) return;
    this->DistributeNBody( nAtoms, R, F, false, 0);
}

void StressGrid::DistributeElasticity(
        const array3_ext & xi_ext,
        const array3_ext & xj_ext,
        const array3_ext & xk_ext,
        const array3_ext & xl_ext,
        real_ext phi_ext,
        real_ext kappa_ext)
{
    if (true == this->state.disable)
        return;
    
    int batch_id = this->m_thread_map[std::this_thread::get_id()];
    matrix6_mds * grid = this->alloc.current_grid_elborn+batch_id*this->state.ncells;

    const array3_mds xi = {(real_mds)xi_ext[0], (real_mds)xi_ext[1], (real_mds)xi_ext[2]};
    const array3_mds xj = {(real_mds)xj_ext[0], (real_mds)xj_ext[1], (real_mds)xj_ext[2]};
    const array3_mds xk = {(real_mds)xk_ext[0], (real_mds)xk_ext[1], (real_mds)xk_ext[2]};
    const array3_mds xl = {(real_mds)xl_ext[0], (real_mds)xl_ext[1], (real_mds)xl_ext[2]};
    const real_mds phi = (real_mds)phi_ext;
    const real_mds kappa = (real_mds)kappa_ext;
    
    //------------------------------------------------------------------------------------
    // Calculate the elasticity tensor
    array3_mds diff, diff2;
    diffarray3( xj, xi, diff, this->state.box, this->state.periodic);
    diffarray3( xl, xk, diff2, this->state.box, this->state.periodic);
    const real_mds rinv = realval_mds(1.0)/normarray3(diff);
    const real_mds rinv2 = kappa*rinv/normarray3(diff2);
    const real_mds rinv3 = phi*rinv*rinv*rinv;
    const real_mds diff00 = diff[0]*diff[0];
    const real_mds diff11 = diff[1]*diff[1];
    const real_mds diff22 = diff[2]*diff[2];
    const real_mds diff01 = diff[0]*diff[1];
    const real_mds diff02 = diff[0]*diff[2];
    const real_mds diff12 = diff[1]*diff[2];
    const real_mds diff200 = diff2[0]*diff2[0];
    const real_mds diff211 = diff2[1]*diff2[1];
    const real_mds diff222 = diff2[2]*diff2[2];
    const real_mds diff201 = diff2[0]*diff2[1];
    const real_mds diff202 = diff2[0]*diff2[2];
    const real_mds diff212 = diff2[1]*diff2[2];

    // Stiffness matrix in Voigt notation
    // 0 = xx; 1 = yy; 2 = zz; 3 = yz or zy; 4 = xz or zx; 5 = xy or yx
    // All indices                         Voigt indices           Stress indices
    // ( xxxx xxyy xxzz xxyz xxxz xxxy ) = ( 00 01 02 03 04 05 ) = [ 0000 0011 0022 0012 0002 0001 ]
    // (      yyyy yyzz yyyz yyxz yyxy ) = (    11 12 13 14 15 ) = [      1111 1122 1112 1102 1101 ]
    // (           zzzz zzyz zzxz zzxy ) = (       22 23 24 25 ) = [           2222 2212 2202 2201 ]
    // (                yzyz yzxz yzxy ) = (          33 34 35 ) = [                1212 1202 1201 ]
    // (                     xzxz xzxy ) = (             44 45 ) = [                     0202 0201 ]
    // (                          xyxy ) = (                55 ) = [                          0101 ]

    matrix6_mds elast = {
        { /*00*/diff00*diff200*rinv2 - diff00*diff00*rinv3, /*01*/diff00*diff211*rinv2 - diff00*diff11*rinv3, /*02*/diff00*diff222*rinv2 - diff00*diff22*rinv3, /*03*/diff00*diff212*rinv2 - diff00*diff12*rinv3, /*04*/diff00*diff202*rinv2 - diff00*diff02*rinv3, /*05*/diff00*diff201*rinv2 - diff00*diff01*rinv3 },
        {                                              0.0, /*11*/diff11*diff211*rinv2 - diff11*diff11*rinv3, /*12*/diff11*diff222*rinv2 - diff11*diff22*rinv3, /*13*/diff11*diff212*rinv2 - diff11*diff12*rinv3, /*14*/diff11*diff202*rinv2 - diff11*diff02*rinv3, /*15*/diff11*diff201*rinv2 - diff11*diff01*rinv3 },
        {                                              0.0,                                              0.0, /*22*/diff22*diff222*rinv2 - diff22*diff22*rinv3, /*23*/diff22*diff212*rinv2 - diff22*diff12*rinv3, /*24*/diff22*diff202*rinv2 - diff22*diff02*rinv3, /*25*/diff22*diff201*rinv2 - diff22*diff01*rinv3 },
        {                                              0.0,                                              0.0,                                              0.0, /*33*/diff12*diff212*rinv2 - diff12*diff12*rinv3, /*34*/diff12*diff202*rinv2 - diff12*diff02*rinv3, /*35*/diff12*diff201*rinv2 - diff12*diff01*rinv3 },
        {                                              0.0,                                              0.0,                                              0.0,                                              0.0, /*44*/diff02*diff202*rinv2 - diff02*diff02*rinv3, /*45*/diff02*diff201*rinv2 - diff02*diff01*rinv3 },
        {                                              0.0,                                              0.0,                                              0.0,                                              0.0,                                              0.0, /*55*/diff01*diff201*rinv2 - diff01*diff01*rinv3 }
    };
    
    if (this->state.griddim == mds_griddim_xyz) {
        distribute_observable_3d(this->state, xi, xj, diff, elast, grid);
    } else {
        distribute_observable_1d(this->state, xi, xj, diff, elast, grid);
    }
}


//----------------------------------------------------------------------------------------
// DistributeKinetic
//
// Distributes kinetic contributions onto the grid
// Requires:
// mass       -> mass of the particle
// x          -> position of the atom
// va         -> velocity of the particle at time t for Vel-verlet, or t-dt/2 for leapfrog integrators
// vb         -> velocity of the particle at t+dt/2 (for leapfrog integrators only)
// atomID     -> ID of the atom
//
// For leapfrog integrators we know va(t-dt/2) and vb(t+dt/2), but we want the contribution at v(t) which we don't know.
// So we take the average kinetic contribution from the velocities at each half step -m*(va(t-dt/2)^2 + vb(t+dt/2)^2)/2
// Warning! this is not the same as simply taking the average of the half-step velocities, which would be incorrect.
//
// For velocity-verlet integrators we know v at the same time step, t, as the positions so the contribution is -m*va(t)*va(t)
void StressGrid::DistributeKinetic(real_ext mass, array3_ext x, array3_ext va, array3_ext vb = nullptr, int atomID = -1)
{
    if (true == this->state.disable)
        return;

    matrix3_mds stress = {0};

    if (MDS_OK == this->state.ierr)
    {
        // get the batch id
        int batch_id = this->m_thread_map[std::this_thread::get_id()];
            
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
        }

        // If spatatom==mds_spat distribute the stress spatially following Noll's procedure
        if (this->state.spatatom == mds_spat)
        {
            // select grid based on batch index
            matrix3_mds * grid = this->alloc.current_grid+batch_id*this->state.ncells;

            // Get the coordinates of the point in the grid
            iarray i1 = {
                int(this->state.nxyz[0] * x[0] * this->state.invbox[0][0] - (x[0] < 0.0) ),
                int(this->state.nxyz[1] * x[1] * this->state.invbox[1][1] - (x[1] < 0.0) ),
                int(this->state.nxyz[2] * x[2] * this->state.invbox[2][2] - (x[2] < 0.0) )
            };
            
            // and the index constants
            const int iip1 = ((i1[0] + 1 + this->state.nxyz[0]) % this->state.nxyz[0])*this->state.nxyz[1]*this->state.nxyz[2];
            const int jjp1 = ((i1[1] + 1 + this->state.nxyz[1]) % this->state.nxyz[1])*this->state.nxyz[2];
            const int kkp1 = ((i1[2] + 1 + this->state.nxyz[2]) % this->state.nxyz[2]);
            const int iim1 = ((i1[0] + this->state.nxyz[0]) % this->state.nxyz[0])*this->state.nxyz[1]*this->state.nxyz[2];
            const int jjm1 = ((i1[1] + this->state.nxyz[1]) % this->state.nxyz[1])*this->state.nxyz[2];
            const int kkm1 = ((i1[2] + this->state.nxyz[2]) % this->state.nxyz[2]);
            
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
            scalesummatrix3( C*xc[0]*xc[1]*xc[2],stress,grid[iip1+jjp1+kkp1]);
            scalesummatrix3(-C*xc[0]*xc[1]*xd[2],stress,grid[iip1+jjp1+kkm1]);
            scalesummatrix3(-C*xc[0]*xd[1]*xc[2],stress,grid[iip1+jjm1+kkp1]);
            scalesummatrix3( C*xc[0]*xd[1]*xd[2],stress,grid[iip1+jjm1+kkm1]);
            scalesummatrix3(-C*xd[0]*xc[1]*xc[2],stress,grid[iim1+jjp1+kkp1]);
            scalesummatrix3( C*xd[0]*xc[1]*xd[2],stress,grid[iim1+jjp1+kkm1]);
            scalesummatrix3( C*xd[0]*xd[1]*xc[2],stress,grid[iim1+jjm1+kkp1]);
            scalesummatrix3(-C*xd[0]*xd[1]*xd[2],stress,grid[iim1+jjm1+kkm1]);
        }
        else if (this->state.spatatom == mds_atom)
        {
            if (atomID == -1)
            {
                std::cout << "ERROR:: Unknown atomID for kinetic contribution. Cannot calculate the stress/atom.";
                return;
            }
            else
            {
                summatrix3(this->alloc.current_grid[atomID],stress,this->alloc.current_grid[atomID]);
            }
        }
    }
}

//----------------------------------------------------------------------------------------
// DistributeKineticElast
//
// Distributes kinetic contributions onto the elasticity tensor grid
// Requires:
// mass       -> mass of the particle
// x          -> position of the atom
// va         -> velocity of the particle at time t for Vel-verlet, or t-dt/2 for leapfrog integrators
// vb         -> velocity of the particle at t+dt/2 (for leapfrog integrators only)
//
// For leapfrog integrators we know va(t-dt/2) and vb(t+dt/2), but we want the contribution at v(t) which we don't know.
// So we take the average kinetic contribution from the velocities at each half step -m*(va(t-dt/2)^2 + vb(t+dt/2)^2)/2
// Warning! this is not the same as simply taking the average of the half-step velocities, which would be incorrect.
//
// For velocity-verlet integrators we know v at the same time step, t, as the positions so the contribution is -m*va(t)*va(t)
void StressGrid::DistributeKineticElast(real_ext mass, array3_ext x, array3_ext va, array3_ext vb = nullptr)
{
    if (true == this->state.disable)
        return;

    matrix6_mds elast = {0};

    // Spreads the velocity in one point
    if (MDS_OK == this->state.ierr) {
        // get the batch id
        int batch_id = this->m_thread_map[std::this_thread::get_id()];
        // select grid based on batch index
        matrix6_mds * grid = this->alloc.current_grid_elkin+batch_id*this->state.ncells;

        // Stiffness matrix in Voigt notation
        // 0 = xx; 1 = yy; 2 = zz; 3 = yz or zy; 4 = xz or zx; 5 = xy or yx
        // All indices                         Voigt indices           Stress indices
        // ( xxxx xxyy xxzz xxyz xxxz xxxy ) = ( 00 01 02 03 04 05 ) = [ 0000 0011 0022 0012 0002 0001 ]
        // (      yyyy yyzz yyyz yyxz yyxy ) = (    11 12 13 14 15 ) = [      1111 1122 1112 1102 1101 ]
        // (           zzzz zzyz zzxz zzxy ) = (       22 23 24 25 ) = [           2222 2212 2202 2201 ]
        // (                yzyz yzxz yzxy ) = (          33 34 35 ) = [                1212 1202 1201 ]
        // (                     xzxz xzxy ) = (             44 45 ) = [                     0202 0201 ]
        // (                          xyxy ) = (                55 ) = [                          0101 ]
        //
        // kd = kronecker delta, p is the particle's momenta
        // ijkl   -> kd(i,l)*pj*pk + kd(j,k)*pi*pl + kd(j,l)*pi*pk + kd(i,k)*pj*pl
        if (vb == nullptr)
        {
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
        const iarray i1 = {
            int(this->state.nxyz[0] * x[0] * this->state.invbox[0][0] - (x[0] < realval_mds(0.0)) ),
            int(this->state.nxyz[1] * x[1] * this->state.invbox[1][1] - (x[1] < realval_mds(0.0)) ),
            int(this->state.nxyz[2] * x[2] * this->state.invbox[2][2] - (x[2] < realval_mds(0.0)) )
        };

        // and the index constants
        const int iip1 = ((i1[0] + 1 + this->state.nxyz[0]) % this->state.nxyz[0])*this->state.nxyz[1]*this->state.nxyz[2];
        const int jjp1 = ((i1[1] + 1 + this->state.nxyz[1]) % this->state.nxyz[1])*this->state.nxyz[2];
        const int kkp1 = ((i1[2] + 1 + this->state.nxyz[2]) % this->state.nxyz[2]);
        const int iim1 = ((i1[0] + this->state.nxyz[0]) % this->state.nxyz[0])*this->state.nxyz[1]*this->state.nxyz[2];
        const int jjm1 = ((i1[1] + this->state.nxyz[1]) % this->state.nxyz[1])*this->state.nxyz[2];
        const int kkm1 = ((i1[2] + this->state.nxyz[2]) % this->state.nxyz[2]);

        // xc = vector from the corner of the point to the corner of the cell
        const real_mds C = this->state.invgridsp * this->state.invgridsp;
        const array3_mds xc = {
            x[0]-this->state.gridsp[0]*i1[0],
            x[1]-this->state.gridsp[1]*i1[1],
            x[2]-this->state.gridsp[2]*i1[2]
        };
        const array3_mds xd = {
            xc[0]-this->state.gridsp[0],
            xc[1]-this->state.gridsp[1],
            xc[2]-this->state.gridsp[2]
        };

        // Spread it
        scalesummatrix6( C*xc[0]*xc[1]*xc[2],elast,grid[iip1+jjp1+kkp1]);
        scalesummatrix6(-C*xc[0]*xc[1]*xd[2],elast,grid[iip1+jjp1+kkm1]);
        scalesummatrix6(-C*xc[0]*xd[1]*xc[2],elast,grid[iip1+jjm1+kkp1]);
        scalesummatrix6( C*xc[0]*xd[1]*xd[2],elast,grid[iip1+jjm1+kkm1]);
        scalesummatrix6(-C*xd[0]*xc[1]*xc[2],elast,grid[iim1+jjp1+kkp1]);
        scalesummatrix6( C*xd[0]*xc[1]*xd[2],elast,grid[iim1+jjp1+kkm1]);
        scalesummatrix6( C*xd[0]*xd[1]*xc[2],elast,grid[iim1+jjm1+kkp1]);
        scalesummatrix6(-C*xd[0]*xd[1]*xd[2],elast,grid[iim1+jjm1+kkm1]);
    }
}

/**
 * PRIVATE CLASS FUNCTIONS
 */
// Method to delete the preallocated member variables
void StressGrid::Clear()
{
    // free any allocated memory
    if (this->alloc.Amat                 != nullptr ) delete [] this->alloc.Amat;
    if (this->alloc.AmatT                != nullptr ) delete [] this->alloc.AmatT;
    if (this->alloc.bvec                 != nullptr ) delete [] this->alloc.bvec;
    if (this->alloc.Rij                  != nullptr ) delete [] this->alloc.Rij;
    if (this->alloc.Fij                  != nullptr ) delete [] this->alloc.Fij;
    if (this->alloc.Uij                  != nullptr ) delete [] this->alloc.Uij;
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
    if (this->alloc.sum_volume           != nullptr ) delete [] this->alloc.sum_volume;
    if (this->alloc.molecule_id          != nullptr ) delete [] this->alloc.molecule_id;
    if (this->alloc.radii                != nullptr ) delete [] this->alloc.radii;
    if (this->alloc.positions            != nullptr ) delete [] this->alloc.positions;
    if (this->alloc.lapack               != nullptr )
    {
        for (int i = 0; i < m_max_threads; ++i)
            delete this->alloc.lapack[i];
        delete [] this->alloc.lapack;
    }

    // clear the  pointers
    memset(&this->alloc, 0, sizeof(MDS_ALLOC_STRUCT) );

    // clear portions of the state
    this->state.avg_boxvol = real_mds(0.0);
    this->state.var_boxvol = real_mds(0.0);
    this->state.maxpart    = 0;

    if (this->state.initialized)
        this->state.initialized = false;
    
#ifdef CUSTRESS_ENABLE
    if (this->state.cuda)
        custress_clear();
#endif//CUSTRESS_ENABLE
}

//----------------------------------------------------------------------------------------
// DistributePairInteraction
//
// Distributes interactions onto locals_grid (from the initial grid point to the last grid point)
// Requires:
// xi   -> position of particle I (A)
// xj   -> position of particle J (B)
// F    -> pairwise force
void StressGrid::DistributePairInteraction( const array3_mds & xi, const array3_mds & xj, const array3_mds & F, int batch_id )
{
    //------------------------------------------------------------------------------------
    matrix3_mds * grid = this->alloc.current_grid+batch_id*this->state.ncells;

    // Calculate the stress tensor
    array3_mds diff;
    diffarray3( xj, xi, diff, this->state.box, this->state.periodic);
    const matrix3_mds stress = {
        (real_mds)(F[0])*diff[0], (real_mds)(F[0])*diff[1], (real_mds)(F[0])*diff[2],
        (real_mds)(F[1])*diff[0], (real_mds)(F[1])*diff[1], (real_mds)(F[1])*diff[2],
        (real_mds)(F[2])*diff[0], (real_mds)(F[2])*diff[1], (real_mds)(F[2])*diff[2]
    };

#ifdef CUSTRESS_ENABLE
    if (this->state.cuda && custress_distribute_pair_mdseraction(xi,xj,F,batch_id))
        return;
#endif
    if (this->state.griddim == mds_griddim_xyz) {
        distribute_observable_3d(this->state, xi, xj, diff, stress, grid);
    } else {
        distribute_observable_1d(this->state, xi, xj, diff, stress, grid);
    }
}

void StressGrid::DistributePairInteraction3D(const array3_mds & xi, const array3_mds & xj, const array3_mds & F, int batch_id, const matrix3_mds stress, const array3_mds diff)
{
    // this is the 3D case
    matrix3_mds * grid = this->alloc.current_grid+batch_id*this->state.ncells;
    
    //------------------------------------------------------------------------------------
    // calculate the grid coordinates (no pbc) for the extreme points
    iarray x = {
        int(this->state.nxyz[0] * xi[0] * this->state.invbox[0][0] - (xi[0] < 0.0) ),
        int(this->state.nxyz[1] * xi[1] * this->state.invbox[1][1] - (xi[1] < 0.0) ),
        int(this->state.nxyz[2] * xi[2] * this->state.invbox[2][2] - (xi[2] < 0.0) )
    };

    const iarray i2 = {
        int(this->state.nxyz[0] * xj[0] * this->state.invbox[0][0] - (xj[0] < 0.0) ),
        int(this->state.nxyz[1] * xj[1] * this->state.invbox[1][1] - (xj[1] < 0.0) ),
        int(this->state.nxyz[2] * xj[2] * this->state.invbox[2][2] - (xj[2] < 0.0) ),
    };
    const array3_mds t_c1 = {
        xi[0] / (xi[0]-xj[0]),
        xi[1] / (xi[1]-xj[1]),
        xi[2] / (xi[2]-xj[2])
    };
    const array3_mds t_c2 = {
        this->state.gridsp[0] / (xi[0]-xj[0]),
        this->state.gridsp[1] / (xi[1]-xj[1]),
        this->state.gridsp[2] / (xi[2]-xj[2])
    };

    const iarray c = {
        int((i2[0]>x[0])-(x[0]>i2[0]) ),
        int((i2[1]>x[1])-(x[1]>i2[1]) ),
        int((i2[2]>x[2])-(x[2]>i2[2]) )
    };

    iarray xn = {
        int(x[0]+(c[0]+1)/2 ),
        int(x[1]+(c[1]+1)/2 ),
        int(x[2]+(c[2]+1)/2 )
    };

    array3_mds d_cgrid = {
        xi[0]-(x[0]+0.5)*this->state.gridsp[0],
        xi[1]-(x[1]+0.5)*this->state.gridsp[1],
        xi[2]-(x[2]+0.5)*this->state.gridsp[2]
    };

    // calculate parametric time in each dimension, and related constants
    array3_mds t = {
        (c[0] == 0) ? realval_mds(1.1) : t_c1[0]-xn[0]*t_c2[0],
        (c[1] == 0) ? realval_mds(1.1) : t_c1[1]-xn[1]*t_c2[1],
        (c[2] == 0) ? realval_mds(1.1) : t_c1[2]-xn[2]*t_c2[2]
    };

    //------------------------------------------------------------------------------------
    // Distribute the stress

    // now the position/spatial constants
    const real_mds C = realval_mds(0.125)*this->state.invgridsp*this->state.invgridsp;
    const real_mds axy = diff[0]*diff[1];
    const real_mds axz = diff[0]*diff[2];
    const real_mds ayz = diff[1]*diff[2];
    const real_mds axyz = diff[0]*ayz; 
    
    // track previous time of crossing
    real_mds oldt = 0.0; 
    
    const int iterations = c[0]*(i2[0]-x[0]) + c[1]*(i2[1]-x[1]) + c[2]*(i2[2]-x[2]);
    for (int count = 0; count <= iterations; ++count)
    {
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
    
        const int iip1 = ((x[0] + 1 + this->state.nxyz[0]) % this->state.nxyz[0])*this->state.nxyz[1]*this->state.nxyz[2];
        const int jjp1 = ((x[1] + 1 + this->state.nxyz[1]) % this->state.nxyz[1])*this->state.nxyz[2];
        const int kkp1 = ((x[2] + 1 + this->state.nxyz[2]) % this->state.nxyz[2]);
        const int iim1 = ((x[0] + this->state.nxyz[0]) % this->state.nxyz[0])*this->state.nxyz[1]*this->state.nxyz[2];
        const int jjm1 = ((x[1] + this->state.nxyz[1]) % this->state.nxyz[1])*this->state.nxyz[2];
        const int kkm1 = ((x[2] + this->state.nxyz[2]) % this->state.nxyz[2]);

        // the composite constants in terms of i, j, k
        const real_mds D[8] = {
            realval_mds(8.0)*bxyz*dt1 + realval_mds(4.0)*(diff[0]*byz+diff[1]*bxz+diff[2]*bxy)*dt2
                + realval_mds(2.0)*(d_cgrid[0]*ayz+d_cgrid[1]*axz+d_cgrid[2]*axy)*dt3 + realval_mds(2.0)*axyz*dt4,
            this->state.gridsp[0]*(realval_mds(4.0)*byz*dt1 + realval_mds(2.0)*(diff[1]*d_cgrid[2]+diff[2]*d_cgrid[1])*dt2 + ayz*dt3),
            this->state.gridsp[1]*(realval_mds(4.0)*bxz*dt1 + realval_mds(2.0)*(diff[0]*d_cgrid[2]+diff[2]*d_cgrid[0])*dt2 + axz*dt3),
            this->state.gridsp[2]*(realval_mds(4.0)*bxy*dt1 + realval_mds(2.0)*(diff[0]*d_cgrid[1]+diff[1]*d_cgrid[0])*dt2 + axy*dt3),
            this->state.gridsp[3]*(realval_mds(2.0)*d_cgrid[2]*dt1+diff[2]*dt2),
            this->state.gridsp[4]*(realval_mds(2.0)*d_cgrid[1]*dt1+diff[1]*dt2),
            this->state.gridsp[5]*(realval_mds(2.0)*d_cgrid[0]*dt1+diff[0]*dt2),
            this->state.gridsp[6]*dt1,
        };

        // perform the sums into the grid
        scalesummatrix3(C*( D[0] + D[1] + D[2] + D[3] + D[4] + D[5] + D[6] + D[7]), stress, grid[iip1 + jjp1 + kkp1]);
        scalesummatrix3(C*(-D[0] - D[1] - D[2] + D[3] - D[4] + D[5] + D[6] + D[7]), stress, grid[iip1 + jjp1 + kkm1]);
        scalesummatrix3(C*(-D[0] - D[1] + D[2] - D[3] + D[4] - D[5] + D[6] + D[7]), stress, grid[iip1 + jjm1 + kkp1]);
        scalesummatrix3(C*( D[0] + D[1] - D[2] - D[3] - D[4] - D[5] + D[6] + D[7]), stress, grid[iip1 + jjm1 + kkm1]);
        scalesummatrix3(C*(-D[0] + D[1] - D[2] - D[3] + D[4] + D[5] - D[6] + D[7]), stress, grid[iim1 + jjp1 + kkp1]);
        scalesummatrix3(C*( D[0] - D[1] + D[2] - D[3] - D[4] + D[5] - D[6] + D[7]), stress, grid[iim1 + jjp1 + kkm1]);
        scalesummatrix3(C*( D[0] - D[1] - D[2] + D[3] + D[4] - D[5] - D[6] + D[7]), stress, grid[iim1 + jjm1 + kkp1]);
        scalesummatrix3(C*(-D[0] + D[1] + D[2] + D[3] - D[4] - D[5] - D[6] + D[7]), stress, grid[iim1 + jjm1 + kkm1]);

        d_cgrid[iX] -= c[iX] * this->state.gridsp[iX];
        oldt = t[iX];
        
        x[iX] += c[iX];
        xn[iX] += c[iX];

        // Next cross point:
        t[iX] = t_c1[iX]-xn[iX]*t_c2[iX];
    }
}

// Decompose 3-body potentials (angles)
void StressGrid::DistributeN3(const array3_mds & Ra, const array3_mds & Rb, const array3_mds & Rc, const array3_mds & Fa, const array3_mds & Fb, const array3_mds & Fc, int batch_id )
{
    // If the force decomposition is cCFD or CFD
    if (this->state.fdecomp == mds_ccfd || this->state.fdecomp == mds_ncfd) {
        // Vectors between particles
        array3_mds AB, AC, BC;
        diffarray3(Rb, Ra, AB, this->state.box, this->state.periodic);
        diffarray3(Rc, Ra, AC, this->state.box, this->state.periodic);
        diffarray3(Rc, Rb, BC, this->state.box, this->state.periodic);

        /// We want to solve M*x = b
        // Matrix of the system (9 equations x 3 unknowns)
        double M[mds_nrow3*mds_ncol3] = {
            /* Vectors for particle 1                 */  /* Vectors for particle 2                  */   /* Vectors for particle 3                  */
            (double)AB[0], (double)AB[1], (double)AB[2], -(double)AB[0], -(double)AB[1], -(double)AB[2],  0.0          ,  0.0          ,  0.0          ,
            (double)AC[0], (double)AC[1], (double)AC[2],  0.0          ,  0.0          ,  0.0          , -(double)AC[0], -(double)AC[1], -(double)AC[2],
            0.0          , 0.0          , 0.0          ,  (double)BC[0],  (double)BC[1],  (double)BC[2], -(double)BC[0], -(double)BC[1], -(double)BC[2],
        };
    
        // Force commponents for each particle
        double b[mds_nrow3] = {
            /* Force on particle 1 */
            (double)Fa[0], (double)Fa[1], (double)Fa[2],

            /* Force on particle 2 */
            (double)Fb[0], (double)Fb[1], (double)Fb[2],

            /* Force on particle 3 */
            (double)Fc[0], (double)Fc[1], (double)Fc[2],
        };
       
        this->state.ierr |= checkError(
                this->alloc.lapack[batch_id]->SolveMinNorm(mds_nrow3, mds_ncol3, M, b),
                "LAPACK solver failed");

        // (Covariant) Central Force decomposition
        real_mds lab = (real_mds)b[0];
        real_mds lac = (real_mds)b[1];
        real_mds lbc = (real_mds)b[2];

        array3_mds Fij1 = {lab * AB[0], lab * AB[1], lab * AB[2]};
        this->DistributePairInteraction(Ra, Rb, Fij1, batch_id);

        array3_mds Fij2 = {lac * AC[0], lac * AC[1], lac * AC[2]};
        this->DistributePairInteraction(Ra, Rc, Fij2, batch_id);

        array3_mds Fij3 = {lbc * BC[0], lbc * BC[1], lbc * BC[2]};
        this->DistributePairInteraction(Rb, Rc, Fij3, batch_id);

    } else if (this->state.fdecomp == mds_gld) {
        array3_mds Fij1 = {
            (Fa[0]-Fb[0])/realval_mds(3.0), (Fa[1]-Fb[1])/realval_mds(3.0), (Fa[2]-Fb[2])/realval_mds(3.0)};
        this->DistributePairInteraction(Ra, Rb, Fij1, batch_id);

        array3_mds Fij2 = {
            (Fa[0]-Fc[0])/realval_mds(3.0), (Fa[1]-Fc[1])/realval_mds(3.0), (Fa[2]-Fc[2])/realval_mds(3.0)};
        this->DistributePairInteraction(Ra, Rc, Fij2, batch_id);

        array3_mds Fij3 = {
            (Fb[0]-Fc[0])/realval_mds(3.0), (Fb[1]-Fc[1])/realval_mds(3.0), (Fb[2]-Fc[2])/realval_mds(3.0)};
        this->DistributePairInteraction(Rb, Rc, Fij3, batch_id);
    }
}

// Decompose Settle
void StressGrid::DistributeSettle( const array3_mds & Ra, const array3_mds & Rb, const array3_mds & Rc, const array3_mds & Fa, const array3_mds & Fb, const array3_mds & Fc, int batch_id )
{
    if (this->state.fdecomp == mds_ccfd || this->state.fdecomp == mds_ncfd || this->state.fdecomp == mds_gld ) {
        array3_mds AB, AC, BC;
        diffarray3(Rb, Ra, AB, this->state.box, this->state.periodic);
        diffarray3(Rc, Ra, AC, this->state.box, this->state.periodic);
        diffarray3(Rc, Rb, BC, this->state.box, this->state.periodic);

        real_mds normAB,normAC,normBC;
        normAB=normarray3(AB);
        normAC=normarray3(AC);
        normBC=normarray3(BC);

        /// We want to solve M*x = b
        // Matrix of the system (9 equations x 3 unknowns)
        double M[mds_nrow3*mds_ncol3] = {
            /* Vectors for particle 1                */   /* Vectors for particle 2                  */   /* Vectors for particle 3                  */
            (double)AB[0], (double)AB[1], (double)AB[2], -(double)AB[0], -(double)AB[1], -(double)AB[2],  0.0          ,  0.0          ,  0.0          ,
            (double)AC[0], (double)AC[1], (double)AC[2],  0.0          ,  0.0          ,  0.0          , -(double)AC[0], -(double)AC[1], -(double)AC[2],
            0.0          , 0.0          , 0.0          ,  (double)BC[0],  (double)BC[1],  (double)BC[2], -(double)BC[0], -(double)BC[1], -(double)BC[2],
        };

        // Force commponents for each particle
        double b[mds_nrow3] = {
            /* Force on particle 1 */
            (double)Fa[0],
            (double)Fa[1],
            (double)Fa[2],
            /* Force on particle 2 */
            (double)Fb[0],
            (double)Fb[1],
            (double)Fb[2],
            /* Force on particle 3 */
            (double)Fc[0],
            (double)Fc[1],
            (double)Fc[2],
        };

        this->state.ierr |= checkError(
                this->alloc.lapack[batch_id]->SolveMinNorm(mds_nrow3, mds_ncol3, M, b),
                "LAPACK solver failed");

        // (Covariant) Central Force decomposition
        real_mds lab = (real_mds)b[0];
        real_mds lac = (real_mds)b[1];
        real_mds lbc = (real_mds)b[2];        

        array3_mds Fij1 = {lab * AB[0], lab * AB[1], lab * AB[2]};
        this->DistributePairInteraction( Ra, Rb, Fij1, batch_id);
        
        array3_mds Fij2 = {lac * AC[0], lac * AC[1], lac * AC[2]};
        this->DistributePairInteraction( Ra, Rc, Fij2, batch_id );

        array3_mds Fij3 = {lbc * BC[0], lbc * BC[1], lbc * BC[2]};
        this->DistributePairInteraction( Rb, Rc, Fij3, batch_id );
        
        // Calculate scalar force and bond stiffness
        real_mds phi_ab = lab*normAB;
        real_mds phi_ac = lac*normAC;
        real_mds phi_bc = lbc*normBC;
        
        //Calculate Elasticity (if we are using mixed precision, will have to convert to ext floats)
        if constexpr (sizeof(real_mds) == sizeof(real_ext) ) {
            this->DistributeElasticity(Ra, Rb, Ra, Rb, phi_ab, realval_mds(0.0));
            this->DistributeElasticity(Ra, Rc, Ra, Rc, phi_ac, realval_mds(0.0));
            this->DistributeElasticity(Rb, Rc, Rb, Rc, phi_bc, realval_mds(0.0));
        } else {
            const array3_ext Ra_ext = {(real_ext)Ra[0], (real_ext)Ra[1], (real_ext)Ra[2]};
            const array3_ext Rb_ext = {(real_ext)Rb[0], (real_ext)Rb[1], (real_ext)Rb[2]};
            const array3_ext Rc_ext = {(real_ext)Rc[0], (real_ext)Rc[1], (real_ext)Rc[2]};
            this->DistributeElasticity(Ra_ext, Rb_ext, Ra_ext, Rb_ext, (real_ext)phi_ab, realval_ext(0.0));
            this->DistributeElasticity(Ra_ext, Rc_ext, Ra_ext, Rc_ext, (real_ext)phi_ac, realval_ext(0.0));
            this->DistributeElasticity(Rb_ext, Rc_ext, Rb_ext, Rc_ext, (real_ext)phi_bc, realval_ext(0.0));
        }
    }
}

// Decompose 4-body potentials (dihedrals)
void StressGrid::DistributeN4( const array3_mds & Ra, const array3_mds & Rb, const array3_mds & Rc, const array3_mds & Rd, const array3_mds & Fa, const array3_mds & Fb, const array3_mds & Fc, const array3_mds & Fd, int batch_id )
{
    if(this->state.fdecomp == mds_ccfd || this->state.fdecomp == mds_ncfd) {
        array3_mds AB, AC, AD, BC, BD, CD;
        diffarray3(Rb, Ra, AB, this->state.box, this->state.periodic);
        diffarray3(Rc, Ra, AC, this->state.box, this->state.periodic);
        diffarray3(Rd, Ra, AD, this->state.box, this->state.periodic);
        diffarray3(Rc, Rb, BC, this->state.box, this->state.periodic);
        diffarray3(Rd, Rb, BD, this->state.box, this->state.periodic);
        diffarray3(Rd, Rc, CD, this->state.box, this->state.periodic);

        double M[mds_nrow4*mds_ncol4] = {
            /* Vectors for particle 1                  */   /* Vectors for particle 2                  */   /* Vectors for particle 3                  */   /* Vectors for particle 4                  */
            (double)AB[0],  (double)AB[1],  (double)AB[2], -(double)AB[0], -(double)AB[1], -(double)AB[2],  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,
            (double)AC[0],  (double)AC[1],  (double)AC[2],  0.0          ,  0.0          ,  0.0          , -(double)AC[0], -(double)AC[1], -(double)AC[2],  0.0          ,  0.0          ,  0.0          ,
            (double)AD[0],  (double)AD[1],  (double)AD[2],  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          , -(double)AD[0], -(double)AD[1], -(double)AD[2],
            0.0          ,  0.0          ,  0.0          ,  (double)BC[0],  (double)BC[1],  (double)BC[2], -(double)BC[0], -(double)BC[1], -(double)BC[2],  0.0          ,  0.0          ,  0.0          ,
            0.0          ,  0.0          ,  0.0          ,  (double)BD[0],  (double)BD[1],  (double)BD[2],  0.0          ,  0.0          ,  0.0          , -(double)BD[0], -(double)BD[1], -(double)BD[2],
            0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  (double)CD[0],  (double)CD[1],  (double)CD[2], -(double)CD[0], -(double)CD[1], -(double)CD[2],
        };

        double b[mds_nrow4] = {
            /* Force on particle 1 */
            (double)Fa[0],
            (double)Fa[1],
            (double)Fa[2],
            /* Force on particle 2 */
            (double)Fb[0],
            (double)Fb[1],
            (double)Fb[2],
            /* Force on particle 3 */
            (double)Fc[0],
            (double)Fc[1],
            (double)Fc[2],
            /* Force on particle 4 */
            (double)Fd[0],
            (double)Fd[1],
            (double)Fd[2],
        };
        
        this->state.ierr |= checkError(
                this->alloc.lapack[batch_id]->SolveMinNorm(mds_nrow4, mds_ncol4, M, b),
                "LAPACK solver failed");

        // Sum the 6 contributions to the stress
        real_mds lab = (real_mds)b[0];
        real_mds lac = (real_mds)b[1];
        real_mds lad = (real_mds)b[2];
        real_mds lbc = (real_mds)b[3];
        real_mds lbd = (real_mds)b[4];
        real_mds lcd = (real_mds)b[5];

        array3_mds Fij1 = {lab * AB[0], lab * AB[1], lab * AB[2]};
        this->DistributePairInteraction(Ra, Rb, Fij1, batch_id);

        array3_mds Fij2 = {lac * AC[0], lac * AC[1], lac * AC[2]};
        this->DistributePairInteraction(Ra, Rc, Fij2, batch_id);

        array3_mds Fij3 = {lad * AD[0], lad * AD[1], lad * AD[2]};
        this->DistributePairInteraction(Ra, Rd, Fij3, batch_id);

        array3_mds Fij4 = {lbc * BC[0], lbc * BC[1], lbc * BC[2]};
        this->DistributePairInteraction(Rb, Rc, Fij4, batch_id);

        array3_mds Fij5 = {lbd * BD[0], lbd * BD[1], lbd * BD[2]};
        this->DistributePairInteraction(Rb, Rd, Fij5, batch_id);

        array3_mds Fij6 = {lcd * CD[0], lcd * CD[1], lcd * CD[2]};
        this->DistributePairInteraction(Rc, Rd, Fij6, batch_id);
    } else if (this->state.fdecomp == mds_gld) {
        array3_mds Fij1 = {
            (Fa[0]-Fb[0])/realval_mds(4.0), (Fa[1]-Fb[1])/realval_mds(4.0), (Fa[2]-Fb[2])/realval_mds(4.0)};
        this->DistributePairInteraction(Ra, Rb, Fij1, batch_id);

        array3_mds Fij2 = {
            (Fa[0]-Fc[0])/realval_mds(4.0), (Fa[1]-Fc[1])/realval_mds(4.0), (Fa[2]-Fc[2])/realval_mds(4.0)};
        this->DistributePairInteraction(Ra, Rc, Fij2, batch_id);

        array3_mds Fij3 = {
            (Fa[0]-Fd[0])/realval_mds(4.0), (Fa[1]-Fd[1])/realval_mds(4.0), (Fa[2]-Fd[2])/realval_mds(4.0)};
        this->DistributePairInteraction(Ra, Rd, Fij3, batch_id);

        array3_mds Fij4 = {
            (Fb[0]-Fc[0])/realval_mds(4.0), (Fb[1]-Fc[1])/realval_mds(4.0), (Fb[2]-Fc[2])/realval_mds(4.0)};
        this->DistributePairInteraction(Rb, Rc, Fij4, batch_id);

        array3_mds Fij5 = {
            (Fb[0]-Fd[0])/realval_mds(4.0), (Fb[1]-Fd[1])/realval_mds(4.0), (Fb[2]-Fd[2])/realval_mds(4.0)};
        this->DistributePairInteraction(Rb, Rd, Fij5, batch_id);

        array3_mds Fij6 = {
            (Fc[0]-Fd[0])/realval_mds(4.0), (Fc[1]-Fd[1])/realval_mds(4.0), (Fc[2]-Fd[2])/realval_mds(4.0)};
        this->DistributePairInteraction(Rc, Rd, Fij6, batch_id);
    }
}


// Decompose 5-body potentials (CMAP)
void StressGrid::DistributeN5(
        const array3_mds & Ra,
        const array3_mds & Rb,
        const array3_mds & Rc,
        const array3_mds & Rd,
        const array3_mds & Re,
        const array3_mds & Fa,
        const array3_mds & Fb,
        const array3_mds & Fc,
        const array3_mds & Fd,
        const array3_mds & Fe,
        int batch_id)
{
    //************************************************************************************

    // Matrix of the system (15 equations x 10 unknowns)
    // Vector, we want to solve M*x = b
    // Scalar product of the Normal and the initial CFD

    // If the force decomposition is cCFD or CFD
    if(this->state.fdecomp == mds_ccfd || this->state.fdecomp == mds_ncfd) {
        array3_mds AB, AC, AD, AE, BC, BD, BE, CD, CE, DE;
        diffarray3(Rb, Ra, AB, this->state.box, this->state.periodic);
        diffarray3(Rc, Ra, AC, this->state.box, this->state.periodic);
        diffarray3(Rd, Ra, AD, this->state.box, this->state.periodic);
        diffarray3(Re, Ra, AE, this->state.box, this->state.periodic);
        diffarray3(Rc, Rb, BC, this->state.box, this->state.periodic);
        diffarray3(Rd, Rb, BD, this->state.box, this->state.periodic);
        diffarray3(Re, Rb, BE, this->state.box, this->state.periodic);
        diffarray3(Rd, Rc, CD, this->state.box, this->state.periodic);
        diffarray3(Re, Rc, CE, this->state.box, this->state.periodic);
        diffarray3(Re, Rd, DE, this->state.box, this->state.periodic);

        real_mds normAB,normAC,normAD,normAE,normBC,normBD,normBE,normCD,normCE,normDE;
        normAB=normarray3(AB);
        normAC=normarray3(AC);
        normAD=normarray3(AD);
        normAE=normarray3(AE);
        normBC=normarray3(BC);
        normBD=normarray3(BD);
        normBE=normarray3(BE);
        normCD=normarray3(CD);
        normCE=normarray3(CE);
        normDE=normarray3(DE);

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

        double M[mds_nrow5*mds_ncol5] = {
             (double)AB[0], (double)AB[1], (double)AB[2], -(double)AB[0], -(double)AB[1], -(double)AB[2],  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,
             (double)AC[0], (double)AC[1], (double)AC[2],  0.0          ,  0.0          ,  0.0          , -(double)AC[0], -(double)AC[1], -(double)AC[2],  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,
             (double)AD[0], (double)AD[1], (double)AD[2],  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          , -(double)AD[0], -(double)AD[1], -(double)AD[2],  0.0          ,  0.0          ,  0.0          ,
             (double)AE[0], (double)AE[1], (double)AE[2],  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          , -(double)AE[0], -(double)AE[1], -(double)AE[2],
             0.0          , 0.0          , 0.0          ,  (double)BC[0],  (double)BC[1],  (double)BC[2], -(double)BC[0], -(double)BC[1], -(double)BC[2],  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,
             0.0          , 0.0          , 0.0          ,  (double)BD[0],  (double)BD[1],  (double)BD[2],  0.0          ,  0.0          ,  0.0          , -(double)BD[0], -(double)BD[1], -(double)BD[2],  0.0          ,  0.0          ,  0.0          ,
             0.0          , 0.0          , 0.0          ,  (double)BE[0],  (double)BE[1],  (double)BE[2],  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          , -(double)BE[0], -(double)BE[1], -(double)BE[2],
             0.0          , 0.0          , 0.0          ,  0.0          ,  0.0          ,  0.0          ,  (double)CD[0],  (double)CD[1],  (double)CD[2], -(double)CD[0], -(double)CD[1], -(double)CD[2],  0.0          ,  0.0          ,  0.0          ,
             0.0          , 0.0          , 0.0          ,  0.0          ,  0.0          ,  0.0          ,  (double)CE[0],  (double)CE[1],  (double)CE[2],  0.0          ,  0.0          ,  0.0          , -(double)CE[0], -(double)CE[1], -(double)CE[2],
             0.0          , 0.0          , 0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  0.0          ,  (double)DE[0],  (double)DE[1],  (double)DE[2], -(double)DE[0], -(double)DE[1], -(double)DE[2],
        };

        double b[mds_nrow5] = {
            (double)Fa[0],
            (double)Fa[1],
            (double)Fa[2],
            (double)Fb[0],
            (double)Fb[1],
            (double)Fb[2],
            (double)Fc[0],
            (double)Fc[1],
            (double)Fc[2],
            (double)Fd[0],
            (double)Fd[1],
            (double)Fd[2],
            (double)Fe[0],
            (double)Fe[1],
            (double)Fe[2],
        };

        this->state.ierr |= checkError(
                this->alloc.lapack[batch_id]->SolveMinNorm(mds_nrow5, mds_ncol5, M, b),
               "LAPACK solver failed");

        // If cCFD project the least squares CFD to the shape space
        if(this->state.fdecomp == mds_ccfd) {
            // Calculate the normal to the Shape Space
            real_mds CaleyMengerNormal[mds_ncol5] = {0};
            ShapeSpace5Normal(normAB,normAC,normAD,normAE,normBC,normBD,normBE,normCD,normCE,normDE,CaleyMengerNormal);
            
            // Covariant derivative:
            real_mds prod = realval_mds(0.0);
            for (int i = 0; i < mds_ncol5; i++ ) {
                prod +=(real_mds)b[i]*CaleyMengerNormal[i];
            }

            for (int i = 0; i < mds_ncol5; i++ ) {
                b[i] = (real_mds)b[i] - prod * CaleyMengerNormal[i];
            }
        }

        real_mds lab = (real_mds)b[0];
        real_mds lac = (real_mds)b[1];
        real_mds lad = (real_mds)b[2];
        real_mds lae = (real_mds)b[3];
        real_mds lbc = (real_mds)b[4];
        real_mds lbd = (real_mds)b[5];
        real_mds lbe = (real_mds)b[6];
        real_mds lcd = (real_mds)b[7];
        real_mds lce = (real_mds)b[8];
        real_mds lde = (real_mds)b[9];

        array3_mds Fij1 = {lab * AB[0], lab * AB[1], lab * AB[2]};
        this->DistributePairInteraction(Ra, Rb, Fij1, batch_id);

        array3_mds Fij2 = {lac * AC[0], lac * AC[1], lac * AC[2]};
        this->DistributePairInteraction(Ra, Rc, Fij2, batch_id);

        array3_mds Fij3 = {lad * AD[0], lad * AD[1], lad * AD[2]};
        this->DistributePairInteraction(Ra, Rd, Fij3, batch_id);

        array3_mds Fij4 = {lae * AE[0], lae * AE[1], lae * AE[2]};
        this->DistributePairInteraction(Ra, Re, Fij4, batch_id);

        array3_mds Fij5 = {lbc * BC[0], lbc * BC[1], lbc * BC[2]};
        this->DistributePairInteraction(Rb, Rc, Fij5, batch_id);

        array3_mds Fij6 = {lbd * BD[0], lbd * BD[1], lbd * BD[2]};
        this->DistributePairInteraction(Rb, Rd, Fij6, batch_id);

        array3_mds Fij7 = {lbe * BE[0], lbe * BE[1], lbe * BE[2]};
        this->DistributePairInteraction(Rb, Re, Fij7, batch_id);

        array3_mds Fij8 = {lcd * CD[0], lcd * CD[1], lcd * CD[2]};
        this->DistributePairInteraction(Rc, Rd, Fij8, batch_id);

        array3_mds Fij9 = {lce * CE[0], lce * CE[1], lce * CE[2]};
        this->DistributePairInteraction(Rc, Re, Fij9, batch_id);

        array3_mds Fij10= {lde * DE[0], lde * DE[1], lde * DE[2]};
        this->DistributePairInteraction(Rd, Re, Fij10, batch_id);
    } else if (this->state.fdecomp == mds_gld) {
        array3_mds Fij1 = {
            (Fa[0]-Fb[0])/realval_mds(5.0), (Fa[1]-Fb[1])/realval_mds(5.0), (Fa[2]-Fb[2])/realval_mds(5.0)};
        this->DistributePairInteraction(Ra, Rb, Fij1, batch_id);

        array3_mds Fij2 = {
            (Fa[0]-Fc[0])/realval_mds(5.0), (Fa[1]-Fc[1])/realval_mds(5.0), (Fa[2]-Fc[2])/realval_mds(5.0)};
        this->DistributePairInteraction(Ra, Rc, Fij2, batch_id);

        array3_mds Fij3 = {
            (Fa[0]-Fd[0])/realval_mds(5.0), (Fa[1]-Fd[1])/realval_mds(5.0), (Fa[2]-Fd[2])/realval_mds(5.0)};
        this->DistributePairInteraction(Ra, Rd, Fij3, batch_id);

        array3_mds Fij4 = {
            (Fa[0]-Fe[0])/realval_mds(5.0), (Fa[1]-Fe[1])/realval_mds(5.0), (Fa[2]-Fe[2])/realval_mds(5.0)};
        this->DistributePairInteraction(Ra, Re, Fij4, batch_id);

        array3_mds Fij5 = {
            (Fb[0]-Fc[0])/realval_mds(5.0), (Fb[1]-Fc[1])/realval_mds(5.0), (Fb[2]-Fc[2])/realval_mds(5.0)};
        this->DistributePairInteraction(Rb, Rc, Fij5, batch_id);

        array3_mds Fij6 = {
            (Fb[0]-Fd[0])/realval_mds(5.0), (Fb[1]-Fd[1])/realval_mds(5.0), (Fb[2]-Fd[2])/realval_mds(5.0)};
        this->DistributePairInteraction(Rb, Rd, Fij6, batch_id);

        array3_mds Fij7 = {
            (Fb[0]-Fe[0])/realval_mds(5.0), (Fb[1]-Fe[1])/realval_mds(5.0), (Fb[2]-Fe[2])/realval_mds(5.0)};
        this->DistributePairInteraction(Rb, Re, Fij7, batch_id);

        array3_mds Fij8 = {
            (Fc[0]-Fd[0])/realval_mds(5.0), (Fc[1]-Fd[1])/realval_mds(5.0), (Fc[2]-Fd[2])/realval_mds(5.0)};
        this->DistributePairInteraction(Rc, Rd, Fij8, batch_id);

        array3_mds Fij9 = {
            (Fc[0]-Fe[0])/realval_mds(5.0), (Fc[1]-Fe[1])/realval_mds(5.0), (Fc[2]-Fe[2])/realval_mds(5.0)};
        this->DistributePairInteraction(Rc, Re, Fij9, batch_id);

        array3_mds Fij10 = {
            (Fd[0]-Fe[0])/realval_mds(5.0), (Fd[1]-Fe[1])/realval_mds(5.0), (Fd[2]-Fe[2])/realval_mds(5.0)};
        this->DistributePairInteraction(Rd, Re, Fij10, batch_id);
    }
}

// General function to decompose N-body potentials (it can be used to compute higher order terms coming from EAM for instance)
void StressGrid::DistributeNBody ( int nPart, array3_ext *R, array3_ext *F, bool distribute_stress, int batch_id)
{
    // this forces NBody to be a single threaded call
    std::lock_guard<std::mutex> lock(this->m_mutex_state);

    int i,j,k, iD, jD, n;
    array3_mds F_ij_temp, Ri_temp, Rj_temp;

    // grow the temp state as needed (only allocating for a single thread here!)
    if (nPart > this->state.maxpart) {
        printf("batch_id(%i) is deleting p_bvec\n", batch_id);
        if (this->alloc.Amat  != nullptr) delete [] this->alloc.Amat;
        if (this->alloc.AmatT != nullptr) delete [] this->alloc.AmatT;
        if (this->alloc.bvec  != nullptr) delete [] this->alloc.bvec;
        if (this->alloc.Rij   != nullptr) delete [] this->alloc.Rij;
        if (this->alloc.Uij   != nullptr) delete [] this->alloc.Uij;
        if (this->alloc.Fij   != nullptr) delete [] this->alloc.Fij;
    
        int maxrows = mds_ndim*nPart;
        int maxcols = (nPart*(nPart-1))/2;

        this->alloc.Amat  = new double [maxrows*maxcols];
        this->alloc.AmatT = new double [maxrows*maxcols];
        this->alloc.bvec  = new double [maxrows];
        this->alloc.Rij  = new array3_mds [nPart*nPart];
        this->alloc.Fij  = new array3_mds [nPart*nPart];
        this->alloc.Uij  = new array3_mds [maxcols];
        
        int maxcols_prev = (this->state.maxpart*(this->state.maxpart-1))/2;
        printf("batch_id(%i) grew p_bvec from %i to %i \n", batch_id, maxcols_prev, maxcols);
        
        this->state.maxpart = nPart;
    }

    // zero the pairwise force and pairwise position arrays
    for ( i = 0; i < nPart*nPart; ++i)
    {
        this->alloc.Rij[i][0] = this->alloc.Rij[i][1] = this->alloc.Rij[i][2] = realval_mds(0.0);
        this->alloc.Fij[i][0] = this->alloc.Fij[i][1] = this->alloc.Fij[i][2] = realval_mds(0.0);
    }

    n = 0;
    for ( i = 0; i < nPart; i++ )
    {
        for ( j = i+1; j < nPart; j++ )
        {
            diffarray3(R[j], R[i], this->alloc.Uij[n], this->state.box, this->state.periodic);
            copyarray3(this->alloc.Uij[n], this->alloc.Rij[i*nPart+j]);
            scalearray3(this->alloc.Uij[n], realval_mds(-1.0), this->alloc.Rij[j*nPart+i]);
            scalearray3(this->alloc.Uij[n], realval_mds(1.0)/normarray3(this->alloc.Uij[n]),this->alloc.Uij[n]);
            n++;
        }
    }

    // If the force decomposition is cCFD or CFD
    if(this->state.fdecomp == mds_ccfd || this->state.fdecomp == mds_ncfd)
    {
        //Number of rows and columns
        int nRow;
        int nCol;

        nRow = mds_ndim * nPart;
        nCol = (nPart * (nPart - 1)) / 2;

        for ( i = 0; i < nCol*nRow; i++ )
        {
            this->alloc.Amat [i] = 0.0;
            this->alloc.AmatT[i] = 0.0;
        }

        n = 0;
        for ( i = 0; i < nPart; i++ )
        {
            iD = mds_ndim * i;
            for ( j = i+1; j < nPart; j++ )
            {
                jD = mds_ndim * j;
                for ( k = 0; k < mds_ndim; k++ )
                {
                    this->alloc.Amat [nRow*n+(iD+k)] =  this->alloc.Uij[n][k];
                    this->alloc.Amat [nRow*n+(jD+k)] = -this->alloc.Uij[n][k];
                    this->alloc.AmatT[(iD+k)*nCol+n] =  this->alloc.Uij[n][k];
                    this->alloc.AmatT[(jD+k)*nCol+n] = -this->alloc.Uij[n][k];
                }
                n++;
            }

            copyarray3(F[i], &this->alloc.bvec[iD]);
        }

        this->state.ierr |= checkError(
                this->alloc.lapack[batch_id]->SolveMinNorm(nRow, nCol, this->alloc.Amat, this->alloc.bvec),
                "LAPACK solver failed");

        if(this->state.fdecomp == mds_ccfd)
            this->alloc.lapack[batch_id]->QQTb( nCol, nRow, nCol, nRow-6, this->alloc.AmatT, this->alloc.bvec );

        n = 0;
        for ( i = 0; i < nPart; i++ )
        {
            for ( j = i+1; j < nPart; j++ )
            {
                scalearray3(this->alloc.Uij[n], (real_mds)this->alloc.bvec[n], F_ij_temp);
                copyarray3(F_ij_temp, this->alloc.Fij[i*nPart+j]);
                scalearray3(F_ij_temp, realval_mds(-1.0), this->alloc.Fij[j*nPart+i]);

                if (distribute_stress)
                {
                    copyarray3(R[i], Ri_temp);
                    copyarray3(R[j], Rj_temp);
                    this->DistributePairInteraction(Ri_temp, Rj_temp, F_ij_temp, batch_id);
                }

                n++;
            }
        }
    }
    else if(this->state.fdecomp == mds_gld)
    {
        n = 0;
        for ( i = 0; i < nPart; i++ )
        {
            for ( j = i+1; j < nPart; j++ )
            {
                diffarray3(F[i], F[j], F_ij_temp );
                scalearray3(F_ij_temp, realval_mds(1.0)/static_cast<real_mds>(nPart), F_ij_temp);
                copyarray3(F_ij_temp, this->alloc.Fij[i*nPart+j]);
                scalearray3(F_ij_temp, realval_mds(-1.0), this->alloc.Fij[j*nPart+i]);

                if (distribute_stress)
                {
                    copyarray3(R[i], Ri_temp);
                    copyarray3(R[j], Rj_temp);
                    this->DistributePairInteraction(Ri_temp, Rj_temp, F_ij_temp, batch_id);
                }
                n++;
            }
        }
    }
}

/*=========================================================================
=========================================================================*/
