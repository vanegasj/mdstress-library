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
#define PI 3.1415926536
#define KB 1.38064852E-23

//KBfac is equal to KB*10E22 to convert V/(KB*T) from J/m^3 to bar^-1
#define KBfac 1.38064852E-1
#define KBN 8.31446261815324

#include "mds_stressgrid.h"
#include "mds_barrier.h"
#include "voro++.hh"

#define iab 0
#define ibg 1
#define iag 2

//#define CUSTRESS_ENABLE
//#ifdef CUSTRESS_ENABLE
//#include "mds_custress.h"
//#endif//CUSTRESS_ENABLE

using namespace mds;

//Constructor
StressGrid::StressGrid()
{
    this->m_nxyz[0] = 0;
    this->m_nxyz[1] = 0;
    this->m_nxyz[2] = 0;
    this->m_nxyzc[0] = 0;
    this->m_nxyzc[1] = 0;
    this->m_nxyzc[2] = 0;
    this->m_griddim = mds_griddim_xyz;
    this->m_maxClust = mds_maxpart;
    this->m_gridsp[0] = 0.0;
    this->m_gridsp[1] = 0.0;
    this->m_gridsp[2] = 0.0;
    this->m_gridsp[3] = 0.0;
    this->m_gridsp[4] = 0.0;
    this->m_gridsp[5] = 0.0;
    this->m_gridsp[6] = 0.0;
    this->m_gridspc[0] = 0.0;
    this->m_gridspc[1] = 0.0;
    this->m_gridspc[2] = 0.0;
    this->m_gridspc[3] = 0.0;
    this->m_gridspc[4] = 0.0;
    this->m_gridspc[5] = 0.0;
    this->m_gridspc[6] = 0.0;
    this->m_invgridsp = 0.0;
    this->m_invgridspc = 0.0;
    this->m_spacing = 0.0;
    this->m_spacingc = 0.0;
    
    this->m_spatatom = 0;
    this->m_fdecomp  = 0;
    this->m_contrib  = 0;
    
    this->m_ierr     = 0;

    this->m_nframes  = 0;
    this->m_nreset   = 0;
    
    for ( int i = 0; i < mds_ndim; i++ )
    {
        for ( int j = 0; j < mds_ndim; j++ )
        {
            this->m_box   [i][j]  = 0.0;
            this->m_sumbox[i][j]  = 0.0;
            this->m_invbox[i][j]  = 0.0;
        }
    }

    this->m_avg_boxvol    = 0.0;
    this->m_var_boxvol    = 0.0;
    this->m_maxpart       = 0;
    this->h_lapack        = nullptr;
    this->p_Amat          = nullptr;
    this->p_AmatT         = nullptr;
    this->p_bvec          = nullptr;
    this->p_Rij           = nullptr;
    this->p_Fij           = nullptr;
    this->p_Uij           = nullptr;
    this->p_current_grid  = nullptr;
    this->p_current_gridtot = nullptr;
    this->p_current_grid_elborn  = nullptr;
    this->p_current_grid_elkin  = nullptr;
    this->p_current_gridc = nullptr;
    this->p_sum_grid      = nullptr;
    this->p_avg_grid      = nullptr;
    this->p_avg_gridtot   = nullptr;
    this->p_sum_gridc     = nullptr;
    this->p_sum_grid_elcovar = nullptr;
    this->p_sum_grid_elkin = nullptr;
    this->p_sum_grid_elborn = nullptr;
    this->p_sum_grid_volcovar = nullptr;
    this->p_sum_gridtot_volcovar = nullptr;
    this->p_sum_volume    = nullptr;
    this->p_molecule_id   = nullptr;
    this->p_radii         = nullptr;
    this->p_positions     = nullptr;
    this->p_pos_gridc     = nullptr;
    this->m_nodispcor   = false;
    this->m_cuda        = false;
    this->m_initialized = false;
    this->m_disable     = false;
    this->m_periodic[0] = false;
    this->m_periodic[1] = false;
    this->m_periodic[2] = false;
    this->m_pcoupl      = false;
    this->m_mindihangle = 0.0;
    this->m_rcoulomb    = 0.0;
    this->m_epsfac      = 0.0;
    this->m_gridctype   = mds_gridc_off;

    this->m_max_threads = 0;
}

//Destructor
StressGrid::~StressGrid()
{
    this->Clear();
}

// Method to delete the preallocated member variables
void StressGrid::Clear()
{
    if (this->p_Amat                 != nullptr ) delete [] this->p_Amat;
    if (this->p_AmatT                != nullptr ) delete [] this->p_AmatT;
    if (this->p_bvec                 != nullptr ) delete [] this->p_bvec;
    if (this->p_Rij                  != nullptr ) delete [] this->p_Rij;
    if (this->p_Fij                  != nullptr ) delete [] this->p_Fij;
    if (this->p_Uij                  != nullptr ) delete [] this->p_Uij;
    if (this->p_current_grid         != nullptr ) delete [] this->p_current_grid;
    if (this->p_current_gridtot      != nullptr ) delete [] this->p_current_gridtot;
    if (this->p_current_grid_elborn  != nullptr ) delete [] this->p_current_grid_elborn;
    if (this->p_current_grid_elkin   != nullptr ) delete [] this->p_current_grid_elkin;
    if (this->p_current_gridc        != nullptr ) delete [] this->p_current_gridc;
    if (this->p_sum_grid             != nullptr ) delete [] this->p_sum_grid;
    if (this->p_avg_grid             != nullptr ) delete [] this->p_avg_grid;
    if (this->p_avg_gridtot          != nullptr ) delete [] this->p_avg_gridtot;
    if (this->p_sum_grid_elcovar     != nullptr ) delete [] this->p_sum_grid_elcovar;
    if (this->p_sum_grid_elkin       != nullptr ) delete [] this->p_sum_grid_elkin;
    if (this->p_sum_grid_elborn      != nullptr ) delete [] this->p_sum_grid_elborn;
    if (this->p_sum_grid_volcovar    != nullptr ) delete [] this->p_sum_grid_volcovar;
    if (this->p_sum_gridtot_volcovar != nullptr ) delete [] this->p_sum_gridtot_volcovar;
    if (this->p_sum_gridc            != nullptr ) delete [] this->p_sum_gridc;
    if (this->p_sum_volume           != nullptr ) delete [] this->p_sum_volume;
    if (this->p_molecule_id          != nullptr ) delete [] this->p_molecule_id;
    if (this->p_radii                != nullptr ) delete [] this->p_radii;
    if (this->p_positions            != nullptr ) delete [] this->p_positions;
    if (this->p_pos_gridc            != nullptr ) delete [] this->p_pos_gridc;
    if (this->h_lapack               != nullptr )
    {
        for (int i = 0; i < m_max_threads; ++i)
            delete this->h_lapack[i];
        delete [] this->h_lapack;
    }
    
    this->m_avg_boxvol    = 0.0;
    this->m_var_boxvol    = 0.0;
    this->m_maxpart       = 0;
    this->p_Amat          = nullptr;
    this->p_AmatT         = nullptr;
    this->p_bvec          = nullptr;
    this->p_Rij           = nullptr;
    this->p_Fij           = nullptr;
    this->p_Uij           = nullptr;
    this->p_current_grid  = nullptr;
    this->p_current_gridtot  = nullptr;
    this->p_current_grid_elborn  = nullptr;
    this->p_current_grid_elkin  = nullptr;
    this->p_current_gridc = nullptr;
    this->p_sum_grid      = nullptr;
    this->p_avg_grid      = nullptr;
    this->p_avg_gridtot   = nullptr;
    this->p_sum_grid_elcovar      = nullptr;
    this->p_sum_grid_elkin      = nullptr;
    this->p_sum_grid_elborn      = nullptr;
    this->p_sum_grid_volcovar      = nullptr;
    this->p_sum_gridtot_volcovar      = nullptr;
    this->p_sum_gridc     = nullptr;
    this->p_sum_volume    = nullptr;
    this->p_molecule_id   = nullptr;
    this->p_radii         = nullptr;
    this->p_positions     = nullptr;
    this->p_pos_gridc     = nullptr;
    this->h_lapack        = nullptr;

    if (this->m_initialized)
        this->m_initialized = false;

#ifdef CUSTRESS_ENABLE
    if (this->m_cuda)
        custress_clear();
#endif//CUSTRESS_ENABLE
}

// This function is provided to identify bad settings
int StressGrid::CheckSettings()
{
    if ( this->m_spatatom != mds_spat && this->m_spatatom != mds_atom )
    {
        std::cout << "ERROR::StressGrid: The stress type (SetStressType) should be either spatial (0) or atomic (1)\n";
        return 1;
    }
    if ( this->m_spatatom == mds_spat )
    {
        if ( this->m_nxyz[0] < 0 || this->m_nxyz[1] < 0 || this->m_nxyz[2] < 0)
        {
            std::cout << "ERROR::StressGrid: The number of grid cells are negative: (nx,ny,nz)=" << " ( " <<this->m_nxyz[0] << ", " << this->m_nxyz[1] << ", " <<this->m_nxyz[2] << " )\n"; 
            return 2;
        }
        
        if ( this->m_nxyzc[0] < 0 || this->m_nxyzc[1] < 0 || this->m_nxyzc[2] < 0)
        {
            std::cout << "ERROR::StressGrid: The number of charge grid cells are negative: (nxc,nyc,nzc)=" << " ( " <<this->m_nxyzc[0] << ", " << this->m_nxyzc[1] << ", " <<this->m_nxyzc[2] << " )\n"; 
            return 2;
        }
        
        if ( this->m_nxyz[0] == 0 && this->m_nxyz[1] == 0 && this->m_nxyz[2] == 0)
        {
            if ( this->m_spacing < mds_eps )
            {
                std::cout << "ERROR::StressGrid: The local spacing is too small: " << this->m_spacing << "( < " << mds_eps << " )\n"; 
                return 3;
            }
            
            if ( iszeromatrix3(this->m_box) )
            {
                std::cout << "ERROR::StressGrid: The initial MD box haven't been set or it's too small\n"; 
                return 4;
            }
        }
        
        if ( this->m_nxyzc[0] == 0 && this->m_nxyzc[1] == 0 && this->m_nxyzc[2] == 0)
        {
            if ( this->m_spacingc < mds_eps )
            {
                std::cout << "ERROR::StressGrid: The local charge spacing is too small: " << this->m_spacingc << "( < " << mds_eps << " )\n"; 
                return 3;
            }
            
            if ( iszeromatrix3(this->m_box) )
            {
                std::cout << "ERROR::StressGrid: The initial MD box haven't been set or it's too small\n"; 
                return 4;
            }
        }
        if ( this->m_fdecomp < mds_ccfd || this->m_fdecomp > mds_gld )
        {
            std::cout << "ERROR::StressGrid: The stress type (SetForceDecomposition) should be: 0 (cCFD), 1 (nCFD), 2 (GLD) or 3 (GMC)\n";
            return 5;
        }
    }
    else
    {
        if ( this->m_nAtoms <= 0 )
        {
            std::cout << "ERROR::StressGrid: Number of atoms must > 0\n";
            return 6;
        }
    }

    if ( this->m_filename.size() == 0 )
    {
        std::cout << "ERROR::StressGrid: Filename for output is not set (SetFileName)" << mds_sl << " and " << mds_cmp << "\n";
        return 8;
    }
    
    if ( this->m_maxClust <= 1 )
    {
        std::cout << "ERROR:StressGrid: The maximum number of particles in a cluster is not valid:" << this->m_maxClust << "\n";
    }
    this->Clear();
    
    return 0;
}

//This function initialize the grid depending on the settings. If the settings are incorrect, 
//it throws an error
void StressGrid::Init()
{
    if (true == this->m_disable) return;
    std::lock_guard<std::mutex> lock(this->m_mutex_state);
    
    //First call checksettings to check if all parameters are OK
    this->m_ierr = this->CheckSettings();

    if ( !this->m_ierr )
    {
        //Sizes
        if (this->m_spatatom == mds_spat)
        {
            if(this->m_nxyz[0] == 0)   this->m_nxyz[0] = static_cast<int>(this->m_box[0][0]/this->m_spacing);
            if(this->m_nxyz[1] == 0)   this->m_nxyz[1] = static_cast<int>(this->m_box[1][1]/this->m_spacing);
            if(this->m_nxyz[2] == 0)   this->m_nxyz[2] = static_cast<int>(this->m_box[2][2]/this->m_spacing);
            
            if(this->m_nxyz[0]==0)  this->m_nxyz[0]=1;
            if(this->m_nxyz[1]==0)  this->m_nxyz[1]=1;
            if(this->m_nxyz[2]==0)  this->m_nxyz[2]=1;
            
            this->m_ncells = this->m_nxyz[0]*this->m_nxyz[1]*this->m_nxyz[2];

            if (this->m_ncells == this->m_nxyz[0])
                this->m_griddim = mds_griddim_xxx;
            else
            if (this->m_ncells == this->m_nxyz[1])
                this->m_griddim = mds_griddim_yyy;
            else
            if (this->m_ncells == this->m_nxyz[2])
                this->m_griddim = mds_griddim_zzz;
            else
                this->m_griddim = mds_griddim_xyz;
            
            // charge distribution grid
            if (this->m_gridctype != mds_gridc_off)
            {
                if(this->m_nxyzc[0] == 0)   this->m_nxyzc[0] = static_cast<int>(this->m_box[0][0]/this->m_spacingc);
                if(this->m_nxyzc[1] == 0)   this->m_nxyzc[1] = static_cast<int>(this->m_box[1][1]/this->m_spacingc);
                if(this->m_nxyzc[2] == 0)   this->m_nxyzc[2] = static_cast<int>(this->m_box[2][2]/this->m_spacingc);
                
                if(this->m_nxyzc[0]==0)  this->m_nxyzc[0]=1;
                if(this->m_nxyzc[1]==0)  this->m_nxyzc[1]=1;
                if(this->m_nxyzc[2]==0)  this->m_nxyzc[2]=1;
                
                this->m_ncellsc = this->m_nxyzc[0]*this->m_nxyzc[1]*this->m_nxyzc[2];
                this->p_pos_gridc     = new array3_int [this->m_ncellsc];
                this->p_sum_gridc     = new real_int [this->m_ncellsc];
                this->p_current_gridc = new real_int [this->m_ncellsc*this->m_max_threads];
                
                for (int i=0; i < this->m_ncellsc; i++)
                {
                    this->p_sum_gridc[i] = 0.0;
                }

                for (int i=0; i < this->m_ncellsc*this->m_max_threads; i++)
                {
                    this->p_current_gridc[i] = 0.0;
                }
            }
        }
        else
        {
            this->m_ncells = this->m_nAtoms;

            // create the molecule_id array
            this->p_molecule_id = new int[this->m_ncells];
            
            // create the radii array
            this->p_radii = new real_int[this->m_ncells];
            this->p_positions = new real_int[3*this->m_ncells];
            this->p_sum_volume = new real_int[this->m_ncells];

            // set them to defaults
            for ( int i=0; i < this->m_ncells; ++i )
            {
                this->p_molecule_id[i] = 0;
                this->p_radii[i] = realval_int(0.001);
                this->p_sum_volume[i] = realval_int(0.0);
        
                this->p_positions[3*i] = realval_int(0.0);
                this->p_positions[3*i+1] = realval_int(0.0);
                this->p_positions[3*i+2] = realval_int(0.0);
            }
        }

        //Give size to current and sum grid
        this->p_sum_grid            = new matrix3_int [this->m_ncells];
        this->p_avg_grid            = new matrix3_int [this->m_ncells];
        this->p_avg_gridtot         = new matrix3_int [1];
        this->p_sum_grid_elcovar    = new matrix6_int [this->m_ncells];
        this->p_sum_grid_elkin      = new matrix6_int [this->m_ncells];
        this->p_sum_grid_elborn     = new matrix6_int [this->m_ncells];
        this->p_sum_grid_volcovar   = new matrix3_int [this->m_ncells];
        this->p_sum_gridtot_volcovar   = new matrix3_int [1];
        this->p_current_grid_elkin  = new matrix6_int [this->m_ncells*this->m_max_threads];
        this->p_current_grid_elborn = new matrix6_int [this->m_ncells*this->m_max_threads];
        this->p_current_grid        = new matrix3_int [this->m_ncells*this->m_max_threads];
        this->p_current_gridtot     = new matrix3_int [1];

        //Set all to zero
        this->m_nframes = 0;
        this->m_avg_boxvol = realval_int(0.0);
        this->m_var_boxvol = realval_int(0.0);
        zeromatrix3(p_current_gridtot[0]);
        zeromatrix3(p_avg_gridtot[0]);
        zeromatrix3(p_sum_gridtot_volcovar[0]);
        for (int i=0; i < this->m_ncells; i++)
        {
            zeromatrix3(this->p_sum_grid[i]);
            zeromatrix3(this->p_avg_grid[i]);
            zeromatrix3(this->p_sum_grid_volcovar[i]);
            zeromatrix6(this->p_sum_grid_elcovar[i]);
            zeromatrix6(this->p_sum_grid_elborn[i]);
            zeromatrix6(this->p_sum_grid_elkin[i]);
        }
        for (int i=0; i < this->m_ncells*this->m_max_threads; i++)
        {
            zeromatrix3(this->p_current_grid[i]);
            zeromatrix6(this->p_current_grid_elborn[i]);
            zeromatrix6(this->p_current_grid_elkin[i]);
        }

        // Finally, create the lapack objects to deal with linear solvers and projections
        this->h_lapack = new Lapack*[m_max_threads];
        for (int i=0; i <this->m_max_threads; ++i)
            this->h_lapack[i] = new Lapack (mds_ndim*this->m_maxClust,(this->m_maxClust*(this->m_maxClust-1))/2);
#ifdef CUSTRESS_ENABLE
        if (this->m_cuda)
            custress_init(this->m_max_threads, this->m_ncells, this->m_nxyz[0], this->m_nxyz[1], this->m_nxyz[2]);
#endif//CUSTRESS_ENABLE

        // Let us know what precision is being used
        std::cout << "STRESSLIB: Internal Floating Points Precision set to " << sizeof(real_int) << " Bytes" << std::endl;
        std::cout << "STRESSLIB: External Floating Points Precision set to " << sizeof(real_ext) << " Bytes" << std::endl;
        std::cout << "STRESSLIB: Output Floating Points Precision set to " << sizeof(real_out) << " Bytes" << std::endl;

        // we have successfully initialized
        this->m_initialized = true;
    }
}

void StressGrid::SetPeriodicBoundaries(bool x, bool y, bool z, bool enforce)
{ 
    if (true == this->m_disable) return;
    std::lock_guard<std::mutex> lock(m_mutex_state);

#ifdef CUSTRESS_ENABLE
    if (this->m_cuda)
        custress_set_periodic(x, y, z, enforce);
#endif//CUSTRESS_ENABLE

    this->m_periodic[0] = x;
    this->m_periodic[1] = y;
    this->m_periodic[2] = z;
    this->m_periodic[3] = enforce;
}

// This function updates the box, invbox and computes the new spacings.
void StressGrid::UpdateBoxSpacings ( matrix3_ext box )
{
    if (true == this->m_disable)
        return;

    if ( !this->m_ierr )
    {
        // every thread must process this latch before proceeding
        static barrier ubs_entry(this->m_max_threads);
        ubs_entry.count_down_and_wait();

        // only thread 0 performs update
        if (this->m_thread_map[std::this_thread::get_id()] == 0)
        {
            copymatrix3( box, this->m_box);
            inversematrix3( this->m_box, this->m_invbox );

            this->m_gridsp[0] = this->m_box[0][0]/static_cast<real_int>(this->m_nxyz[0]);
            this->m_gridsp[1] = this->m_box[1][1]/static_cast<real_int>(this->m_nxyz[1]);
            this->m_gridsp[2] = this->m_box[2][2]/static_cast<real_int>(this->m_nxyz[2]);
            this->m_gridsp[3] = this->m_gridsp[0]*this->m_gridsp[1];
            this->m_gridsp[4] = this->m_gridsp[0]*this->m_gridsp[2];
            this->m_gridsp[5] = this->m_gridsp[1]*this->m_gridsp[2];
            this->m_gridsp[6] = this->m_gridsp[0]*this->m_gridsp[1]*this->m_gridsp[2];
            this->m_invgridsp =  realval_int(1.0)/(this->m_gridsp[0]*this->m_gridsp[1]*this->m_gridsp[2]);

            if (this->m_gridctype != mds_gridc_off)
            {
                this->m_gridspc[0] = this->m_box[0][0]/static_cast<real_int>(this->m_nxyzc[0]);
                this->m_gridspc[1] = this->m_box[1][1]/static_cast<real_int>(this->m_nxyzc[1]);
                this->m_gridspc[2] = this->m_box[2][2]/static_cast<real_int>(this->m_nxyzc[2]);
                this->m_gridspc[3] = this->m_gridspc[0]*this->m_gridspc[1];
                this->m_gridspc[4] = this->m_gridspc[0]*this->m_gridspc[2];
                this->m_gridspc[5] = this->m_gridspc[1]*this->m_gridspc[2];
                this->m_gridspc[6] = this->m_gridspc[0]*this->m_gridspc[1]*this->m_gridspc[2];
                this->m_invgridspc = realval_int(1.0)/(this->m_gridspc[0]*this->m_gridspc[1]*this->m_gridspc[2]);

                for (int i=0; i < this->m_ncellsc; i++)
                {
                    iarray xi;
                    //xi[0] = i/(this->m_nxyzc[1]*this->m_nxyzc[2]);
                    //xi[1] = (i-xi[0]*this->m_nxyzc[1]*this->m_nxyzc[2])/this->m_nxyzc[2];
                    //xi[2] = (i-xi[0]*this->m_nxyzc[1]*this->m_nxyzc[2]-xi[1]*this->m_nxyzc[2]);
                    xi[0] = i/(this->m_nxyzc[2]*this->m_nxyzc[1]);
                    xi[1] = (i/this->m_nxyzc[2])%this->m_nxyzc[1];
                    xi[2] = i%this->m_nxyzc[2];

                    // calculate the position
                    this->p_pos_gridc[i][0] = this->m_gridspc[0]*xi[0];
                    this->p_pos_gridc[i][1] = this->m_gridspc[1]*xi[1];
                    this->p_pos_gridc[i][2] = this->m_gridspc[2]*xi[2];
                }

            }

            summatrix3( this->m_box, this->m_sumbox, this->m_sumbox );
            //this->m_sum_boxvol = this->m_sum_boxvol + box[0][0]*box[1][1]*box[2][2];
#ifdef CUSTRESS_ENABLE
            if (this->m_cuda)
                custress_update_box_spacings(this->m_box, this->m_invbox, this->m_gridsp);
#endif//CUSTRESS_ENABLE
        }

        // every thread must process this latch before exiting
        static barrier ubs_exit(this->m_max_threads);
        ubs_exit.count_down_and_wait();
    }
}

// This function sums the current grid to sum_grid and sets current_grid
// to zero.
void StressGrid::SumGrid ( )
{
    if (true == this->m_disable)
        return;

    if ( !this->m_ierr )
    {
        // get the thread id
        int thread_id = this->m_thread_map[std::this_thread::get_id()];

        if (this->m_gridctype != mds_gridc_off)
        {
            // every thread must process this latch before proceeding
            static barrier sumgrid_enter_charge_reduction(this->m_max_threads);
            sumgrid_enter_charge_reduction.count_down_and_wait();

            // reduce all charge current grids
            for (int i = thread_id; i < this->m_ncellsc; i+=this->m_max_threads)
            {
                if (i < this->m_ncellsc)
                {
                    for (int j = 1; j < this->m_max_threads; ++j)
                    {
                        this->p_current_gridc[i] = this->p_current_gridc[i] + this->p_current_gridc[i+j*this->m_ncellsc];
                        this->p_current_gridc[i+j*this->m_ncellsc] = 0.0;
                    }
                }
            }

            // every thread must process this latch before proceeding
            static barrier sumgrid_enter_coulomb(this->m_max_threads);
            sumgrid_enter_coulomb.count_down_and_wait();
            real_int beta = this->m_ewaldcoeff_q;
            // do coulomb distribute stress here
            for (int i = thread_id; i < this->m_ncellsc; i+=this->m_max_threads)
            {
                // calculate the charge
                real_int qi = this->p_current_gridc[i];
                if (abs(qi) > realval_int(1E-16) )
                {
                    qi /= this->m_invgridspc;
                    // calculate the indices
                    for (int j = i+1; j < this->m_ncellsc; j+=1)
                    {
                        // calculate the charge
                        real_int qj = this->p_current_gridc[j];
                        if (abs(qj) > realval_int(1E-16) )
                        {
                            qj /= this->m_invgridspc;

                            // calculate r
                            array3_int diff;
                            diffarray3(this->p_pos_gridc[i], this->p_pos_gridc[j], diff, this->m_box, this->m_periodic);

                            // calculate r and rinv
                            real_int r2 = (real_int)(diff[0]*diff[0])+(real_int)(diff[1]*diff[1])+(real_int)(diff[2]*diff[2]);
                            real_int r = sqrt(r2);
                            real_int rinv = realval_int(1.0)/r;

                            // calculate the force
                            //double F = -this->m_epsfac*(qi*qj)*(rinv*rinv*rinv);
                            real_int F = -this->m_epsfac*qi*qj*(realval_int(2.0)*beta*exp(-beta*beta*r2)/sqrt(PI) - erf(beta*r)*rinv)*rinv*rinv;

                            // calculate the force vectors
                            array3_int Fij = {
                                F*diff[0],
                                F*diff[1],
                                F*diff[2],
                                };
                            
                            // correct Ri
                            array3_int Ri;
                            sumarray3(this->p_pos_gridc[j], diff, Ri);

                            // distribute stress
                            this->DistributePairInteraction(Ri, this->p_pos_gridc[j], Fij, thread_id);
                        }
                    }
                }
            }
            // end coulomb distribute stress
        }
        
        // every thread must process this latch before proceeding
        static barrier sumgrid_enter_stress_reduction(this->m_max_threads);
        sumgrid_enter_stress_reduction.count_down_and_wait();
        
        // reduce all stress current grids
        for (int i = thread_id; i < this->m_ncells; i+=this->m_max_threads)
        {
            if (i < this->m_ncells)
            {
                for (int j = 1; j < this->m_max_threads; ++j)
                {
                    summatrix3(this->p_current_grid[i], this->p_current_grid[i+j*this->m_ncells], this->p_current_grid[i] );
                    zeromatrix3(this->p_current_grid[i+j*this->m_ncells]);
                    summatrix6(this->p_current_grid_elborn[i], this->p_current_grid_elborn[i+j*this->m_ncells], this->p_current_grid_elborn[i] );
                    zeromatrix6(this->p_current_grid_elborn[i+j*this->m_ncells]);
                    summatrix6(this->p_current_grid_elkin[i], this->p_current_grid_elkin[i+j*this->m_ncells], this->p_current_grid_elkin[i] );
                    zeromatrix6(this->p_current_grid_elkin[i+j*this->m_ncells]);
                }
            }
        }

        // every thread must process this latch before proceeding
        static barrier sumgrid_continue(this->m_max_threads);
        sumgrid_continue.count_down_and_wait();

        if (thread_id == 0)
        {
#ifdef CUSTRESS_ENABLE
            if (this->m_cuda)
                custress_sum_grid(this->p_current_grid);
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

            this->m_nframes++;
            real_int Vnew, deltaV, deltaV2;
            Vnew = this->m_box[0][0]*this->m_box[1][1]*this->m_box[2][2];
            deltaV = Vnew - this->m_avg_boxvol;
            this->m_avg_boxvol += deltaV/this->m_nframes;
            deltaV2 = Vnew - this->m_avg_boxvol;
            this->m_var_boxvol += deltaV*deltaV2;

            if (this->m_spatatom == mds_spat)
            {
                for (int i = 0; i < this->m_ncells; i++)
                    summatrix3(this->p_current_gridtot[0], this->p_current_grid[i], this->p_current_gridtot[0]); // compute y = sigma_total_kl = sum(sigma_local_kl)/this->m_ncells
                scalematrix3(this->p_current_gridtot[0], realval_int(1.0)/this->m_ncells, this->p_current_gridtot[0]); // scale by this->m_ncells
                scalesummatrix3(realval_int(-1.0), this->p_avg_gridtot[0], this->p_current_gridtot[0]); // subtract meany from y and store back into y (which now becomes dy)
                scalesummatrix3(realval_int(1.0)/this->m_nframes, this->p_current_gridtot[0], this->p_avg_gridtot[0]); //compute meany
                scalesummatrix3(deltaV, this->p_current_gridtot[0], this->p_sum_gridtot_volcovar[0]); // accumulate the covar(sigma_total_kl, Vol)
                matrix6_int tmp_covar[1];
                matrix3_int dx[1];
                for (int i = 0; i < this->m_ncells; i++)
                {
                    summatrix3( this->p_sum_grid[i], this->p_current_grid[i], this->p_sum_grid[i] );
                    summatrix6( this->p_sum_grid_elborn[i], this->p_current_grid_elborn[i], this->p_sum_grid_elborn[i] );
                    summatrix6( this->p_sum_grid_elkin[i], this->p_current_grid_elkin[i], this->p_sum_grid_elkin[i] );
                    scalematrix3(this->p_avg_grid[i], realval_int(-1.0), dx[0]); // dx = -meanx
                    summatrix3(dx[0], this->p_current_grid[i], dx[0]); // dx += x
                    scalesummatrix3(realval_int(1.0)/this->m_nframes, dx[0], this->p_avg_grid[i]); //compute meanx
                    scalesummatrix3(deltaV, dx[0], this->p_sum_grid_volcovar[i]); // accumulate covar(sigma_local_ij, Vol)
                    matrixouterprod6( dx[0], this->p_current_gridtot[0], tmp_covar[0]); // dx*dy
                    summatrix6(this->p_sum_grid_elcovar[i], tmp_covar[0], this->p_sum_grid_elcovar[i]); //this accumulates the covar of sigma_local_ij*sigma_total_kl
                }

                for (int i = 0; i < this->m_ncellsc; i++)
                {
                    this->p_sum_gridc[i] = this->p_sum_gridc[i] + this->p_current_gridc[i];
                }
            }
            else
            {
                // initialize the voro container
                array3_int vor_box;
                vor_box[0] = (real_int)this->m_box[0][0];
                vor_box[1] = (real_int)this->m_box[1][1];
                vor_box[2] = (real_int)this->m_box[2][2];

                real_int gfxy,gfxz;
                gfxy = vor_box[1]/vor_box[0];
                gfxz = vor_box[2]/vor_box[0];

                // need to count the number of sites without radius 0.0
                int vcells = 0;
                for (int i = 0; i < this->m_ncells; ++i)
                {
                    if (this->p_radii[i] > 0.0)
                        vcells += 1;
                }

                int gridn[3];
                gridn[0] = pow(this->m_ncells/(3*gfxy*gfxz), 1/3.0);
                gridn[1] = gridn[0]*gfxy;
                gridn[2] = gridn[0]*gfxz;

                // create the voronoi objects
                voro::particle_order vorpo = voro::particle_order(this->m_ncells);
                voro::container_poly vorcon = voro::container_poly(0.0, vor_box[0], 0.0, vor_box[1], 0.0, vor_box[2], gridn[0], gridn[1], gridn[2], this->m_periodic[0], this->m_periodic[1], this->m_periodic[2], 8);
                // fill the container
                int voro_cells = 0;
                for (int i = 0; i < this->m_ncells; ++i)
                {
                    if (this->p_radii[i] > 0.0)
                    {
                        voro_cells += 1;
                        double px = (double)this->p_positions[3*i];
                        double py = (double)this->p_positions[3*i+1];
                        double pz = (double)this->p_positions[3*i+2];

                        // we are scaling the radii down to 1/100th the size here so that the number of voronoi cells is the same as
                        // the number of atoms with non-zero radii. This is because voro++ does a check and if two particles are within
                        // a distance less than the sum of their radii, they are put in the same voronoi cell, which does not allow
                        // the calculation of volumes of every particle in the system
                        vorcon.put(vorpo, i, px, py, pz, 0.01*this->p_radii[i]);
                    }
                }

                voro::voronoicell c;
                voro::c_loop_order vl(vorcon, vorpo);

                // the particle/cell id
                int pid = 0;

                // track the particle/cell id of atom with largest volume in this molecule
                int this_molecule = this->p_molecule_id[pid];
                int last_pid = 0;
                real_int last_volume = realval_int(0.0);

                int cells_computed = 0;
                if (vl.start())
                {
                    do{
                        if (this->p_radii[pid] > 0.0 && vorcon.compute_cell(c,vl))
                        {
                            // count the cells
                            cells_computed += 1;

                            // get the volume of this cell
                            last_volume = (real_int)(c.volume());
                            last_pid = pid;

                            // mark the last volume encountered
                            this_molecule = this->p_molecule_id[pid];

                            scalematrix3( this->p_current_grid[pid], realval_int(1.0)/last_volume, this->p_current_grid[pid] );
                            summatrix3( this->p_sum_grid[pid], this->p_current_grid[pid], this->p_sum_grid[pid] );

                            // add the volume
                            this->p_sum_volume[pid] += last_volume;
                        }
                        else
                        {
                            // check that we are working on the same molecule
                            if (this_molecule != this->p_molecule_id[pid])
                            {
                                this->m_ierr = 13;
                                std::cout << "ERROR:: radius of zero encountered, but last atom molecule was not this molecule" << std::endl;
                            }
                            else
                            {
                                scalematrix3( this->p_current_grid[pid], realval_int(1.0)/last_volume, this->p_current_grid[pid] );
                                summatrix3( this->p_sum_grid[last_pid], this->p_current_grid[pid], this->p_sum_grid[last_pid] );
                            }
                        }

                        // always increment the particle count
                        pid += 1;
                    }while(vl.inc());

                    // should do an error check here
                    if (pid != this->m_nAtoms)
                    {
                        this->m_ierr = 11;
                        std::cout << "ERROR:: number of atoms processed does not match number of sites" << std::endl;
                    }
                }

            }

            zeromatrix3(this->p_current_gridtot[0]);
            for(int i = 0; i < this->m_ncells; ++i)
            {
                zeromatrix3(this->p_current_grid[i]);
                zeromatrix6(this->p_current_grid_elborn[i]);
                zeromatrix6(this->p_current_grid_elkin[i]);
            }
            for(int i = 0; i < this->m_ncellsc; ++i)
            {
                this->p_current_gridc[i] = 0.0;
            }
        }

        // every thread must process this latch before exiting
        static barrier sumgrid_exit(this->m_max_threads);
        sumgrid_exit.count_down_and_wait();
    }
}

void StressGrid::DispersionCorrection (real_ext shift)
{
    if (true == this->m_disable)
        return;

    if (this->m_nodispcor == false)
    {
        // select the correct grid
        int batch_id = this->m_thread_map[std::this_thread::get_id()];
        matrix3_int * grid = this->p_current_grid+batch_id*this->m_ncells;

        // add shift to grid
        for(int i = 0; i < this->m_ncells; i++)
        {
            for(int m = 0; m < 3; m++)
                grid[i][m][m] += shift;
        }
    }
}

//Set both sum_grid and current_grid to zero. Sum the number of resets (this is used for 
//printing files) and set the number of frames to zero
void StressGrid::Reset ( )
{
    if (true == this->m_disable)
        return;

    if ( !this->m_ierr)
    {
        // get the thread id
        int thread_id = this->m_thread_map[std::this_thread::get_id()];

        // every thread must process this latch before proceeding
        static barrier reset_entry(this->m_max_threads);
        reset_entry.count_down_and_wait();

        // every thread zeros its own current grid
        for( int i=0; i<this->m_ncells; i++ )
        {
            zeromatrix3(this->p_current_grid[i+thread_id*this->m_max_threads]);
            zeromatrix6(this->p_current_grid_elborn[i+thread_id*this->m_max_threads]);
            zeromatrix6(this->p_current_grid_elkin[i+thread_id*this->m_max_threads]);
        }
        for( int i=0; i<this->m_ncellsc; i++ )
        {
            this->p_current_gridc[i+thread_id*this->m_max_threads] = 0.0;
        }

        if (thread_id == 0)
        {
            // thread 0 deals with the sum grid
            zeromatrix3(this->p_current_gridtot[0]);
            zeromatrix3(this->p_avg_gridtot[0]);
            for( int i=0; i<this->m_ncells; i++ )
            {
                zeromatrix3( this->p_sum_grid[i] );
                zeromatrix3( this->p_avg_grid[i] );
                zeromatrix3( this->p_sum_grid_volcovar[i] );
                zeromatrix6( this->p_sum_grid_elborn[i] );
                zeromatrix6( this->p_sum_grid_elcovar[i] );
                zeromatrix6( this->p_sum_grid_elkin[i] );
            }
            for( int i=0; i<this->m_ncellsc; i++ )
            {
                this->p_sum_gridc[i] = 0.0;
            }

            if (this->m_spatatom == mds_atom)
            {
                for( int i=0; i<this->m_ncells; i++ )
                    this->p_sum_volume[i] = 0.0;
            }
            this->m_nframes = 0;
            this->m_nreset ++;
            this->m_avg_boxvol = 0.0;
            this->m_var_boxvol = 0.0;
        }

        // every thread must process this latch before exiting
        static barrier reset_exit(this->m_max_threads);
        reset_exit.count_down_and_wait();
    }
}


//Writes file with average stress to grid using the filename set by the user
void StressGrid::Write ( )
{
    if (true == this->m_disable)
        return;

    if ( !this->m_ierr)
    {
            int                Dtype=1;
            std::string        outname, charge_outname,  elborn_outname, elcovar_outname, elkin_outname, eltotal_outname;
            std::ostringstream outnumber;
            FILE *outfile = nullptr;
            FILE *charge_outfile = nullptr;
            FILE *elborn_outfile = nullptr;
            FILE *elcovar_outfile = nullptr;
            FILE *elkin_outfile = nullptr;
            FILE *eltotal_outfile = nullptr;;

            outnumber << this->m_nreset;
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

            // open the main output file
            outfile = fopen(outname.c_str(), "wb" );
            if (this->m_spatatom == mds_spat)
                Dtype = 1;
            else if (this->m_spatatom == mds_atom)
                Dtype = 2;
            fwrite(&Dtype, sizeof(int), 1, outfile);

            // use different dtype for elasticity file
            elcovar_outfile = fopen(elcovar_outname.c_str(), "wb" );
            elborn_outfile = fopen(elborn_outname.c_str(), "wb" );
            elkin_outfile = fopen(elkin_outname.c_str(), "wb" );
            eltotal_outfile = fopen(eltotal_outname.c_str(), "wb" );

            Dtype = 6;
            fwrite(&Dtype, sizeof(int), 1, elcovar_outfile);
            fwrite(&Dtype, sizeof(int), 1, elborn_outfile);
            fwrite(&Dtype, sizeof(int), 1, elkin_outfile);
            fwrite(&Dtype, sizeof(int), 1, eltotal_outfile);

            //Divide sumbox with respect to the number of frames to get the avg
            matrix3_int        avgbox;
            scalematrix3( this->m_sumbox, realval_int(1.0)/this->m_nframes, avgbox);
            
            //Need to copy to an array of output precision
            matrix3_out        avgbox_out;
            copymatrix3(avgbox, avgbox_out);

            // writing average box to each outfile
            fwrite(avgbox_out, sizeof(matrix3_out), 1, outfile);
            fwrite(avgbox_out, sizeof(matrix3_out), 1, elcovar_outfile);
            fwrite(avgbox_out, sizeof(matrix3_out), 1, elborn_outfile);
            fwrite(avgbox_out, sizeof(matrix3_out), 1, elkin_outfile);
            fwrite(avgbox_out, sizeof(matrix3_out), 1, eltotal_outfile);

            if (this->m_spatatom == mds_spat)
            {
                // this is the number of grids in all dimensions, also to all files
                fwrite(&this->m_nxyz[0], sizeof(this->m_nxyz[0]), 1, outfile);
                fwrite(&this->m_nxyz[1], sizeof(this->m_nxyz[1]), 1, outfile);
                fwrite(&this->m_nxyz[2], sizeof(this->m_nxyz[2]), 1, outfile);
                fwrite(&this->m_nxyz[0], sizeof(this->m_nxyz[0]), 1, elcovar_outfile);
                fwrite(&this->m_nxyz[1], sizeof(this->m_nxyz[1]), 1, elcovar_outfile);
                fwrite(&this->m_nxyz[2], sizeof(this->m_nxyz[2]), 1, elcovar_outfile);
                fwrite(&this->m_nxyz[0], sizeof(this->m_nxyz[0]), 1, elborn_outfile);
                fwrite(&this->m_nxyz[1], sizeof(this->m_nxyz[1]), 1, elborn_outfile);
                fwrite(&this->m_nxyz[2], sizeof(this->m_nxyz[2]), 1, elborn_outfile);
                fwrite(&this->m_nxyz[0], sizeof(this->m_nxyz[0]), 1, elkin_outfile);
                fwrite(&this->m_nxyz[1], sizeof(this->m_nxyz[1]), 1, elkin_outfile);
                fwrite(&this->m_nxyz[2], sizeof(this->m_nxyz[2]), 1, elkin_outfile);
                fwrite(&this->m_nxyz[0], sizeof(this->m_nxyz[0]), 1, eltotal_outfile);
                fwrite(&this->m_nxyz[1], sizeof(this->m_nxyz[1]), 1, eltotal_outfile);
                fwrite(&this->m_nxyz[2], sizeof(this->m_nxyz[2]), 1, eltotal_outfile);
            }
            else
            {
                // if we are using mds_atom, then there is only ncells == natoms
                fwrite(&this->m_ncells, sizeof(int), 1, outfile);
                fwrite(&this->m_ncells, sizeof(int), 1, elcovar_outfile);
                fwrite(&this->m_ncells, sizeof(int), 1, elborn_outfile);
                fwrite(&this->m_ncells, sizeof(int), 1, elkin_outfile);
                fwrite(&this->m_ncells, sizeof(int), 1, eltotal_outfile);
            }

            // calculate stress factors
            real_int stressfac, stress2fac, covfac, covfac2;
            //stressfac = real_int(mds_units)/this->m_nframes;
            stressfac = real_int(mds_units);
            stress2fac = real_int(mds_units)*real_int(mds_units)/this->m_nframes;
            covfac = -real_int(mds_units)*real_int(mds_units)*this->m_avg_boxvol/(this->m_temperature*real_int(KBfac)*this->m_nframes);
            covfac2 = realval_int(0.0);
            if (this->m_pcoupl == true)
                covfac2 = real_int(mds_units)*real_int(mds_units)*this->m_avg_boxvol/(this->m_temperature*real_int(KBfac)*this->m_var_boxvol*this->m_nframes);

            // need to store matrices in double precision
            matrix3_out sum_grid;
            matrix6_out sum_grid_elcovar;
            matrix6_out sum_grid_elborn;
            matrix6_out sum_grid_elkin;

            // zero them
            zeromatrix3(sum_grid);
            zeromatrix6(sum_grid_elcovar);
            zeromatrix6(sum_grid_elborn);
            zeromatrix6(sum_grid_elkin);

            // need this for corrections below
            matrix6_int npt_covar_corr[1];
            for ( int i = 0; i < this->m_ncells; i++ )
            {
                //scalematrix3(this->p_sum_grid[i], stressfac, this->p_sum_grid[i]);
                scalematrix3(this->p_avg_grid[i], stressfac, this->p_avg_grid[i]); // Use the online average stress instead of the cummulative stress divided by the number of frames
                scalematrix6(this->p_sum_grid_elcovar[i], covfac, this->p_sum_grid_elcovar[i]);
                scalematrix6(this->p_sum_grid_elborn[i], stressfac, this->p_sum_grid_elborn[i]);
                scalematrix6(this->p_sum_grid_elkin[i], stressfac, this->p_sum_grid_elkin[i]);

                matrixouterprod6(this->p_sum_grid_volcovar[i], this->p_sum_gridtot_volcovar[0], npt_covar_corr[0]); // Cov(sigma_local_ij,V)*Cov(sigma_total_kl,V)
                scalematrix6(npt_covar_corr[0], covfac2, npt_covar_corr[0]); // scale the term above by <V>/kT*Var(V)
                summatrix6(this->p_sum_grid_elcovar[i], npt_covar_corr[0], this->p_sum_grid_elcovar[i]);
                
                // copying to dmatrix3/6 here for m_ncell writes to respective files
                //copymatrix3(this->p_sum_grid[i], sum_grid);
                copymatrix3(this->p_avg_grid[i], sum_grid);
                copymatrix6(this->p_sum_grid_elcovar[i], sum_grid_elcovar);
                copymatrix6(this->p_sum_grid_elborn[i], sum_grid_elborn);
                copymatrix6(this->p_sum_grid_elkin[i], sum_grid_elkin);

                fwrite(&sum_grid[0], sizeof(matrix3_out), 1, outfile);
                fwrite(&sum_grid_elcovar[0], sizeof(matrix6_out), 1, elcovar_outfile);
                fwrite(&sum_grid_elborn[0], sizeof(matrix6_out), 1, elborn_outfile);
                fwrite(&sum_grid_elkin[0], sizeof(matrix6_out), 1, elkin_outfile);
            }

            // append the volume data
            if (this->m_spatatom == mds_atom)
            {
                for ( int i = 0; i < this->m_ncells; i++ )
                {
                    real_out sum_volume = (real_out)(this->p_sum_volume[i] / this->m_nframes);
                    fwrite(&sum_volume, sizeof(real_out), 1, outfile);
                }
            }

            fclose(outfile);
            fclose(elcovar_outfile);
            fclose(elborn_outfile);
            fclose(elkin_outfile);
            for ( int i = 0; i < this->m_ncells; i++ )
            {
                summatrix6(this->p_sum_grid_elcovar[i], this->p_sum_grid_elborn[i], this->p_sum_grid_elcovar[i]); // add up all the contributions to the total elasticity tensor
                summatrix6(this->p_sum_grid_elcovar[i], this->p_sum_grid_elkin[i], this->p_sum_grid_elcovar[i]);
                
                // need to store matrices in double precision
                copymatrix6(this->p_sum_grid_elcovar[i], sum_grid_elcovar);

                fwrite(&sum_grid_elcovar[0], sizeof(matrix6_out), 1, eltotal_outfile);
            }

            fclose(eltotal_outfile);
            if (this->m_gridctype != mds_gridc_off)
            {
                charge_outname = "charge_" + this->m_filename + outnumber.str();
                if (charge_outname.find(".dat") == std::string::npos)
                    charge_outname = charge_outname + "." + mds_fileext;
                charge_outfile = fopen(charge_outname.c_str(), "wb" );

                // use different dtype for charge file
                Dtype = 3;
                fwrite(&Dtype, sizeof(int), 1, charge_outfile);
                fwrite(avgbox, sizeof(matrix3_out), 1, charge_outfile);

                if (this->m_spatatom == mds_spat)
                {
                    fwrite(&this->m_nxyzc[0], sizeof(this->m_nxyzc[0]), 1, charge_outfile);
                    fwrite(&this->m_nxyzc[1], sizeof(this->m_nxyzc[1]), 1, charge_outfile);
                    fwrite(&this->m_nxyzc[2], sizeof(this->m_nxyzc[2]), 1, charge_outfile);
                }
                else
                {
                    fwrite(&this->m_ncellsc, sizeof(int), 1, charge_outfile);
                }

                for ( int i = 0; i < this->m_ncellsc; i++ )
                {
                    this->p_sum_gridc[i] = this->p_sum_gridc[i]/this->m_nframes;
                    real_out sum_gridc = (real_out)this->p_sum_gridc[i];
                    fwrite(&sum_gridc, sizeof(real_out), 1, charge_outfile);
                }

                fclose(charge_outfile);
            }
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
    if (true == this->m_disable) return;
    std::lock_guard<std::mutex> lock(this->m_mutex_state);

    // quick check if we have the correct number of atoms
    if (atomID > this->m_ncells)
    {
        this->m_ierr = 12;
        std::cout << "ERROR:: atomID is greater than number of cells" << std::endl;
    }

    if (!this->m_ierr)
    {
        // setup the molecule id
        this->p_radii[atomID] = std::max(radius,realval_ext(0.001));
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
    if (true == this->m_disable) return;
    std::lock_guard<std::mutex> lock(this->m_mutex_state);

    // quick check if we have the correct number of atoms
    if (atomID > this->m_ncells)
    {
        this->m_ierr = 12;
        std::cout << "ERROR:: atomID is greater than number of cells" << std::endl;
    }

    if (!this->m_ierr)
    {
        // set the positions
        this->p_positions[3*atomID] = px;
        this->p_positions[3*atomID+1] = py;
        this->p_positions[3*atomID+2] = pz;

        // set the molecular id
        this->p_molecule_id[atomID] = moleID;
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
    if (true == this->m_disable)
        return;

    int    n;
    int    i,j;
    real_int temp;
    array3_int Ra,Rb,Rc,Rd,Rf;
    array3_int Fa,Fb,Fc,Fd,Ff;
    matrix3_int stress;

    int batch_id = this->m_thread_map[std::this_thread::get_id()];
    if ( nAtoms > this->m_maxClust )
    {
        std::cout << "ERROR::StressGrid: Distribute Interaction has been called with a number of atoms larger than the maximum cluster size previously set, nAtoms=" << nAtoms << " and maxClust=" << this->m_maxClust << "\n";
        this->m_ierr = 8;
    }
    
    if ( !this->m_ierr )
    {
        // If spatatom==mds_spat distribute the stress spatially following Noll's procedure
        if (this->m_spatatom == mds_spat)
        {
            // Depending on the number of atoms, call a different function
            switch (nAtoms)
            {
                case 2:
                    copyarray3(R[0],Ra);
                    copyarray3(R[1],Rb);
                    copyarray3(F[0],Fa);
                    this->DistributePairInteraction( Ra, Rb, Fa, batch_id );
                    break;

                case 3:
                    copyarray3(R[0],Ra);
                    copyarray3(R[1],Rb);
                    copyarray3(R[2],Rc);
                    copyarray3(F[0],Fa);
                    copyarray3(F[1],Fb);
                    copyarray3(F[2],Fc);
                    this->DistributeN3( Ra, Rb, Rc, Fa, Fb, Fc, batch_id );
                    break;

                case -3:
                    copyarray3(R[0],Ra);
                    copyarray3(R[1],Rb);
                    copyarray3(R[2],Rc);
                    copyarray3(F[0],Fa);
                    copyarray3(F[1],Fb);
                    copyarray3(F[2],Fc);
                    this->DistributeSettle( Ra, Rb, Rc, Fa, Fb, Fc, batch_id );
                    break;

                case 4:
                    copyarray3(R[0],Ra);
                    copyarray3(R[1],Rb);
                    copyarray3(R[2],Rc);
                    copyarray3(R[3],Rd);
                    copyarray3(F[0],Fa);
                    copyarray3(F[1],Fb);
                    copyarray3(F[2],Fc);
                    copyarray3(F[3],Fd);
                    this->DistributeN4( Ra, Rb, Rc, Rd, Fa, Fb, Fc, Fd, batch_id );
                    break;

                case 5:
                    copyarray3(R[0],Ra);
                    copyarray3(R[1],Rb);
                    copyarray3(R[2],Rc);
                    copyarray3(R[3],Rd);
                    copyarray3(R[4],Rf);
                    copyarray3(F[0],Fa);
                    copyarray3(F[1],Fb);
                    copyarray3(F[2],Fc);
                    copyarray3(F[3],Fd);
                    copyarray3(F[4],Ff);
                    this->DistributeN5( Ra, Rb, Rc, Rd, Rf, Fa, Fb, Fc, Fd, Rf, batch_id );
                    break;

                default:
                    this->DistributeNBody( nAtoms, R, F, true, batch_id );
                    break;
            }

        }
        // If spatatom==mds_atom, distributes the stress per atom using the atomic stress definition
        else if (this->m_spatatom == mds_atom)
        {
            // This is because SETTLE calls the function with nAtoms=-3
            if (nAtoms < 0) nAtoms = -nAtoms;

            if (atomIDs == nullptr)
            {
                std::cout << "ERROR:: the atomIDs array is nullptr. Cannot calculate the stress/atom.";
                return;
            }

            for (n = 0; n < nAtoms; n++ )
            {
                if ( atomIDs[n] >= this->m_nAtoms )
                {
                    std::cout << "ERROR:: the atom label" << atomIDs[n] << "is equal or larger than the total number of atoms" << this->m_nAtoms;
                    return;
                }
            }

            //Initialize the value of the (local) stress to 0
            for( i = 0; i< mds_ndim; i++ )
            {
                stress[i][i] = 0.0;
                for( j=i+1; j< mds_ndim; j++ )
                {
                    stress[i][j] = 0.0;
                    stress[j][i] = 0.0;
                }
            }


            //Calculate the stress
            for( n = 0; n < nAtoms; n++ )
            {
                for( i = 0; i< mds_ndim; i++ )
                {
                    temp = -(F[n][i] * R[n][i])/nAtoms;
                    stress[i][i] += temp;
                    for( j=i+1; j< mds_ndim; j++ )
                    {
                        temp = -(F[n][i] * R[n][j])/nAtoms;
                        stress[i][j] += temp;
                        stress[j][i] += temp;
                    }
                }
            }

            for ( n = 0; n < nAtoms; n++ )
                summatrix3(this->p_current_grid[atomIDs[n]],stress,this->p_current_grid[atomIDs[n]]);
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
    if (true == this->m_disable) return;
    this->DistributeNBody( nAtoms, R, F, false, 0);
}

//----------------------------------------------------------------------------------------
// DistributePairInteraction
//
// Distributes interactions onto locals_grid (from the initial grid point to the last grid point)
// Requires:
// xi   -> position of particle I (A)
// xj   -> position of particle J (B)
// F    -> pairwise force
void StressGrid::DistributePairInteraction( array3_int xi, array3_int xj, array3_int F, int batch_id )
{
#ifdef CUSTRESS_ENABLE
    if (this->m_cuda && custress_distribute_pair_interaction(xi,xj,F,batch_id))
        return;
#endif
    if (this->m_griddim == mds_griddim_xyz)
        DistributePairInteraction3D(xi,xj,F,batch_id);
    else
        DistributePairInteraction1D(xi,xj,F,batch_id);
}

void StressGrid::DistributePairInteraction1D(array3_int xi, array3_int xj, array3_int F, int batch_id )
{
    // select grid based on batch index
    matrix3_int * grid = this->p_current_grid+batch_id*this->m_ncells;
    
    //------------------------------------------------------------------------------------
    // Calculate the stress tensor
    array3_int diff;
    diffarray3( xj, xi, diff, this->m_box, this->m_periodic);
    matrix3_int stress;
    stress[0][0] = (real_int)(F[0])*diff[0];
    stress[1][0] = (real_int)(F[1])*diff[0];
    stress[2][0] = (real_int)(F[2])*diff[0];
    stress[0][1] = (real_int)(F[0])*diff[1];
    stress[1][1] = (real_int)(F[1])*diff[1];
    stress[2][1] = (real_int)(F[2])*diff[1];
    stress[0][2] = (real_int)(F[0])*diff[2];
    stress[1][2] = (real_int)(F[1])*diff[2];
    stress[2][2] = (real_int)(F[2])*diff[2];

    // calculate the grid coordinates (no pbc) for the extreme points
    int x = this->m_nxyz[this->m_griddim] * xi[this->m_griddim] * this->m_invbox[this->m_griddim][this->m_griddim] - (xi[this->m_griddim] < 0.0);
    const int i2 = this->m_nxyz[this->m_griddim] * xj[this->m_griddim] * this->m_invbox[this->m_griddim][this->m_griddim] - (xj[this->m_griddim] < 0.0);
    const int c = (i2>x)-(x>i2);
    const real_int t_c1 = xi[this->m_griddim] / (xi[this->m_griddim]-xj[this->m_griddim]);
    const real_int t_c2 = this->m_gridsp[this->m_griddim] / (xi[this->m_griddim]-xj[this->m_griddim]);
    const real_int C = realval_int(0.5)*this->m_invgridsp*this->m_invgridsp;
    real_int d_cgrid = xi[this->m_griddim]-(x+realval_int(0.5))*this->m_gridsp[this->m_griddim];
    int xn = x+(c+1)/2;

    //------------------------------------------------------------------------------------
    // Distribute the stress

    // track previous time of crossing and check that sum is complete (?)
    real_int oldt = 0.0; 

    // fix the number of iterations
    const int iterations = c*(i2-x);
    for (int count = 0; count <= iterations; ++count)
    {
        // there is always iterations+1, where the last iteration deals
        // with any residual
        real_int newt = (iterations == count) ? realval_int(1.0) : t_c1-xn*t_c2;

        // work out the parametric time constants
        const real_int t12 = oldt*oldt;
        const real_int t22 = newt*newt;
        const real_int dt1 = newt - oldt;
        const real_int dt2 = t22 - t12;

        const int p1 = ((x + 1 + this->m_nxyz[this->m_griddim]) % this->m_nxyz[this->m_griddim]);
        const int m1 = ((x + this->m_nxyz[this->m_griddim]) % this->m_nxyz[this->m_griddim]);

        // the composite constants in terms of i, j, k
        const real_int D1 = this->m_gridsp[5-this->m_griddim]*(realval_int(2.0)*d_cgrid*dt1+diff[this->m_griddim]*dt2);
        const real_int D2 = this->m_gridsp[6]*dt1;
        scalesummatrix3(C*( D1 + D2), stress, grid[p1]);
        scalesummatrix3(C*(-D1 + D2), stress, grid[m1]);
        
        d_cgrid -= c * m_gridsp[this->m_griddim];
        oldt = newt;
        
        x += c;
        xn += c;
    }
}

void StressGrid::DistributePairInteraction3D(array3_int xi, array3_int xj, array3_int F, int batch_id )
{
    // this is the 3D case
    matrix3_int * grid = this->p_current_grid+batch_id*this->m_ncells;
    
    //------------------------------------------------------------------------------------
    // Calculate the stress tensor
    array3_int diff;
    diffarray3( xj, xi, diff, this->m_box, this->m_periodic);

    matrix3_int stress;
    stress[0][0] = F[0]*diff[0];
    stress[1][0] = F[1]*diff[0];
    stress[2][0] = F[2]*diff[0];
    stress[0][1] = F[0]*diff[1];
    stress[1][1] = F[1]*diff[1];
    stress[2][1] = F[2]*diff[1];
    stress[0][2] = F[0]*diff[2];
    stress[1][2] = F[1]*diff[2];
    stress[2][2] = F[2]*diff[2];

    // calculate the grid coordinates (no pbc) for the extreme points
    iarray x, xn, i2, c;
    array3_int t,t_c1,t_c2,d_cgrid;
    x[0] = this->m_nxyz[0] * xi[0] * this->m_invbox[0][0] - (xi[0] < 0.0);
    x[1] = this->m_nxyz[1] * xi[1] * this->m_invbox[1][1] - (xi[1] < 0.0);
    x[2] = this->m_nxyz[2] * xi[2] * this->m_invbox[2][2] - (xi[2] < 0.0);
    i2[0] = this->m_nxyz[0] * xj[0] * this->m_invbox[0][0] - (xj[0] < 0.0);
    i2[1] = this->m_nxyz[1] * xj[1] * this->m_invbox[1][1] - (xj[1] < 0.0);
    i2[2] = this->m_nxyz[2] * xj[2] * this->m_invbox[2][2] - (xj[2] < 0.0);
    c[0] = (i2[0]>x[0])-(x[0]>i2[0]);
    c[1] = (i2[1]>x[1])-(x[1]>i2[1]);
    c[2] = (i2[2]>x[2])-(x[2]>i2[2]);
    d_cgrid[0] = xi[0]-(x[0]+0.5)*this->m_gridsp[0];
    d_cgrid[1] = xi[1]-(x[1]+0.5)*this->m_gridsp[1];
    d_cgrid[2] = xi[2]-(x[2]+0.5)*this->m_gridsp[2];
    xn[0] = x[0]+(c[0]+1)/2;
    xn[1] = x[1]+(c[1]+1)/2;
    xn[2] = x[2]+(c[2]+1)/2;

    // calculate parametric time in each dimension, and related constants
    t_c1[0] = xi[0] / (xi[0]-xj[0]);
    t_c1[1] = xi[1] / (xi[1]-xj[1]);
    t_c1[2] = xi[2] / (xi[2]-xj[2]);
    t_c2[0] = this->m_gridsp[0] / (xi[0]-xj[0]);
    t_c2[1] = this->m_gridsp[1] / (xi[1]-xj[1]);
    t_c2[2] = this->m_gridsp[2] / (xi[2]-xj[2]);
    t[0] = (c[0] == 0) ? realval_int(1.1) : t_c1[0]-xn[0]*t_c2[0];
    t[1] = (c[1] == 0) ? realval_int(1.1) : t_c1[1]-xn[1]*t_c2[1];
    t[2] = (c[2] == 0) ? realval_int(1.1) : t_c1[2]-xn[2]*t_c2[2];

    //------------------------------------------------------------------------------------
    // Distribute the stress

    // now the position/spatial constants
    const real_int C = realval_int(0.125)*this->m_invgridsp*this->m_invgridsp;
    const real_int axy = diff[0]*diff[1];
    const real_int axz = diff[0]*diff[2];
    const real_int ayz = diff[1]*diff[2];
    const real_int axyz = diff[0]*ayz; 
    
    // track previous time of crossing
    real_int oldt = 0.0; 
    
    const int iterations = c[0]*(i2[0]-x[0]) + c[1]*(i2[1]-x[1]) + c[2]*(i2[2]-x[2]);
    for (int count = 0; count <= iterations; ++count)
    {
        // figure out index
        const int cmp0x = ((t[0]<t[1]+mds_eps) + (t[0]<t[2]+mds_eps))/2;
        const int cmp1x = ((t[1]<t[0]+mds_eps) + (t[1]<t[2]+mds_eps))/2;
        const int cmp2x = ((t[2]<t[0]+mds_eps) + (t[2]<t[1]+mds_eps))/2;
        const int iX = (1-cmp0x)*(cmp1x+2*(1-cmp1x)*cmp2x);
        const real_int newt = (iterations == count) ? 1.0 : t[iX];

        // work out the parametric time constants
        const real_int t12 = oldt*oldt;
        const real_int t22 = newt*newt;
        const real_int dt1 = newt - oldt;
        const real_int dt2 = t22 - t12;
        const real_int dt3 = realval_int(4.0)*(t22*newt - t12*oldt)/realval_int(3.0);
        const real_int dt4 = t22*t22 - t12*t12;

        // additional constants
        const real_int bxy = d_cgrid[0]*d_cgrid[1];
        const real_int bxz = d_cgrid[0]*d_cgrid[2];
        const real_int byz = d_cgrid[1]*d_cgrid[2];
        const real_int bxyz = d_cgrid[0]*byz;
    
        const int iip1 = ((x[0] + 1 + this->m_nxyz[0]) % this->m_nxyz[0])*this->m_nxyz[1]*this->m_nxyz[2];
        const int jjp1 = ((x[1] + 1 + this->m_nxyz[1]) % this->m_nxyz[1])*this->m_nxyz[2];
        const int kkp1 = ((x[2] + 1 + this->m_nxyz[2]) % this->m_nxyz[2]);
        const int iim1 = ((x[0] + this->m_nxyz[0]) % this->m_nxyz[0])*this->m_nxyz[1]*this->m_nxyz[2];
        const int jjm1 = ((x[1] + this->m_nxyz[1]) % this->m_nxyz[1])*this->m_nxyz[2];
        const int kkm1 = ((x[2] + this->m_nxyz[2]) % this->m_nxyz[2]);

        // the composite constants in terms of i, j, k
        const real_int D[8] = {
            realval_int(8.0)*bxyz*dt1 + realval_int(4.0)*(diff[0]*byz+diff[1]*bxz+diff[2]*bxy)*dt2
                + realval_int(2.0)*(d_cgrid[0]*ayz+d_cgrid[1]*axz+d_cgrid[2]*axy)*dt3 + realval_int(2.0)*axyz*dt4,
            this->m_gridsp[0]*(realval_int(4.0)*byz*dt1 + realval_int(2.0)*(diff[1]*d_cgrid[2]+diff[2]*d_cgrid[1])*dt2 + ayz*dt3),
            this->m_gridsp[1]*(realval_int(4.0)*bxz*dt1 + realval_int(2.0)*(diff[0]*d_cgrid[2]+diff[2]*d_cgrid[0])*dt2 + axz*dt3),
            this->m_gridsp[2]*(realval_int(4.0)*bxy*dt1 + realval_int(2.0)*(diff[0]*d_cgrid[1]+diff[1]*d_cgrid[0])*dt2 + axy*dt3),
            this->m_gridsp[3]*(realval_int(2.0)*d_cgrid[2]*dt1+diff[2]*dt2),
            this->m_gridsp[4]*(realval_int(2.0)*d_cgrid[1]*dt1+diff[1]*dt2),
            this->m_gridsp[5]*(realval_int(2.0)*d_cgrid[0]*dt1+diff[0]*dt2),
            this->m_gridsp[6]*dt1,
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

        d_cgrid[iX] -= c[iX] * m_gridsp[iX];
        oldt = t[iX];
        
        x[iX] += c[iX];
        xn[iX] += c[iX];

        // Next cross point:
        t[iX] = t_c1[iX]-xn[iX]*t_c2[iX];
    }
}

void StressGrid::DistributeElasticity_internal1D(array3_int xi, array3_int xj, array3_int xk, array3_int xl, real_int phi, real_int kappa)
{
    // this is the 1D case
    int batch_id = this->m_thread_map[std::this_thread::get_id()];
    matrix6_int * gridElast = this->p_current_grid_elborn+batch_id*this->m_ncells;

    //------------------------------------------------------------------------------------
    // Calculate the elasticity tensor
    array3_int diff, diff2;
    real_int rinv, rinv2, rinv3;
    diffarray3( xj, xi, diff, this->m_box, this->m_periodic);
    diffarray3( xl, xk, diff2, this->m_box, this->m_periodic);
    rinv = realval_int(1.0)/normarray3(diff);
    rinv2 = (real_int)(kappa)*rinv/normarray3(diff2);
    rinv3 = (real_int)(phi)*rinv*rinv*rinv;

    // Stiffness matrix in Voigt notation
    // 0 = xx; 1 = yy; 2 = zz; 3 = yz or zy; 4 = xz or zx; 5 = xy or yx
    // All indices                         Voigt indices           Stress indices
    // ( xxxx xxyy xxzz xxyz xxxz xxxy ) = ( 00 01 02 03 04 05 ) = [ 0000 0011 0022 0012 0002 0001 ]
    // (      yyyy yyzz yyyz yyxz yyxy ) = (    11 12 13 14 15 ) = [      1111 1122 1112 1102 1101 ]
    // (           zzzz zzyz zzxz zzxy ) = (       22 23 24 25 ) = [           2222 2212 2202 2201 ]
    // (                yzyz yzxz yzxy ) = (          33 34 35 ) = [                1212 1202 1201 ]
    // (                     xzxz xzxy ) = (             44 45 ) = [                     0202 0201 ]
    // (                          xyxy ) = (                55 ) = [                          0101 ]

    matrix6_int elast;
    elast[0][0] = diff[0]*diff[0]*diff2[0]*diff2[0]*rinv2 - diff[0]*diff[0]*diff[0]*diff[0]*rinv3;
    elast[0][1] = diff[0]*diff[0]*diff2[1]*diff2[1]*rinv2 - diff[0]*diff[0]*diff[1]*diff[1]*rinv3;
    elast[0][2] = diff[0]*diff[0]*diff2[2]*diff2[2]*rinv2 - diff[0]*diff[0]*diff[2]*diff[2]*rinv3;
    elast[0][3] = diff[0]*diff[0]*diff2[1]*diff2[2]*rinv2 - diff[0]*diff[0]*diff[1]*diff[2]*rinv3;
    elast[0][4] = diff[0]*diff[0]*diff2[0]*diff2[2]*rinv2 - diff[0]*diff[0]*diff[0]*diff[2]*rinv3;
    elast[0][5] = diff[0]*diff[0]*diff2[0]*diff2[1]*rinv2 - diff[0]*diff[0]*diff[0]*diff[1]*rinv3;
    elast[1][1] = diff[1]*diff[1]*diff2[1]*diff2[1]*rinv2 - diff[1]*diff[1]*diff[1]*diff[1]*rinv3;
    elast[1][2] = diff[1]*diff[1]*diff2[2]*diff2[2]*rinv2 - diff[1]*diff[1]*diff[2]*diff[2]*rinv3;
    elast[1][3] = diff[1]*diff[1]*diff2[1]*diff2[2]*rinv2 - diff[1]*diff[1]*diff[1]*diff[2]*rinv3;
    elast[1][4] = diff[1]*diff[1]*diff2[0]*diff2[2]*rinv2 - diff[1]*diff[1]*diff[0]*diff[2]*rinv3;
    elast[1][5] = diff[1]*diff[1]*diff2[0]*diff2[1]*rinv2 - diff[1]*diff[1]*diff[0]*diff[1]*rinv3;
    elast[2][2] = diff[2]*diff[2]*diff2[2]*diff2[2]*rinv2 - diff[2]*diff[2]*diff[2]*diff[2]*rinv3;
    elast[2][3] = diff[2]*diff[2]*diff2[1]*diff2[2]*rinv2 - diff[2]*diff[2]*diff[1]*diff[2]*rinv3;
    elast[2][4] = diff[2]*diff[2]*diff2[0]*diff2[2]*rinv2 - diff[2]*diff[2]*diff[0]*diff[2]*rinv3;
    elast[2][5] = diff[2]*diff[2]*diff2[0]*diff2[1]*rinv2 - diff[2]*diff[2]*diff[0]*diff[1]*rinv3;
    elast[3][3] = diff[1]*diff[2]*diff2[1]*diff2[2]*rinv2 - diff[1]*diff[2]*diff[1]*diff[2]*rinv3;
    elast[3][4] = diff[1]*diff[2]*diff2[0]*diff2[2]*rinv2 - diff[1]*diff[2]*diff[0]*diff[2]*rinv3;
    elast[3][5] = diff[1]*diff[2]*diff2[0]*diff2[1]*rinv2 - diff[1]*diff[2]*diff[0]*diff[1]*rinv3;
    elast[4][4] = diff[0]*diff[2]*diff2[0]*diff2[2]*rinv2 - diff[0]*diff[2]*diff[0]*diff[2]*rinv3;
    elast[4][5] = diff[0]*diff[2]*diff2[0]*diff2[1]*rinv2 - diff[0]*diff[2]*diff[0]*diff[1]*rinv3;
    elast[5][5] = diff[0]*diff[1]*diff2[0]*diff2[1]*rinv2 - diff[0]*diff[1]*diff[0]*diff[1]*rinv3;

    // calculate the grid coordinates (no pbc) for the extreme points
    int x = this->m_nxyz[this->m_griddim] * xi[this->m_griddim] * this->m_invbox[this->m_griddim][this->m_griddim] - (xi[this->m_griddim] < 0.0);
    const int i2 = this->m_nxyz[this->m_griddim] * xj[this->m_griddim] * this->m_invbox[this->m_griddim][this->m_griddim] - (xj[this->m_griddim] < 0.0);
    const int c = (i2>x)-(x>i2);
    const real_int t_c1 = xi[this->m_griddim] / (xi[this->m_griddim]-xj[this->m_griddim]);
    const real_int t_c2 = this->m_gridsp[this->m_griddim] / (xi[this->m_griddim]-xj[this->m_griddim]);
    const real_int C = realval_int(0.5)*this->m_invgridsp*this->m_invgridsp;
    real_int d_cgrid = xi[this->m_griddim]-(x+realval_int(0.5))*this->m_gridsp[this->m_griddim];
    int xn = x+(c+1)/2;

    //------------------------------------------------------------------------------------
    // Distribute the stress

    // track previous time of crossing and check that sum is complete (?)
    real_int oldt = realval_int(0.0); 

    // fix the number of iterations
    const int iterations = c*(i2-x);
    for (int count = 0; count <= iterations; ++count)
    {
        // there is always iterations+1, where the last iteration deals
        // with any residual
        const real_int newt = (iterations == count) ? realval_int(1.0) : t_c1-xn*t_c2;

        // work out the parametric time constants
        const real_int t12 = oldt*oldt;
        const real_int t22 = newt*newt;
        const real_int dt1 = newt - oldt;
        const real_int dt2 = t22 - t12;

        const int p1 = ((x + 1 + this->m_nxyz[this->m_griddim]) % this->m_nxyz[this->m_griddim]);
        const int m1 = ((x + this->m_nxyz[this->m_griddim]) % this->m_nxyz[this->m_griddim]);

        // the composite constants in terms of i, j, k
        const real_int D1 = this->m_gridsp[5-this->m_griddim]*(realval_int(2.0)*d_cgrid*dt1+diff[this->m_griddim]*dt2);
        const real_int D2 = this->m_gridsp[6]*dt1;
        scalesummatrix6(C*( D1 + D2), elast, gridElast[p1]);
        scalesummatrix6(C*(-D1 + D2), elast, gridElast[m1]);
        
        d_cgrid -= c * m_gridsp[this->m_griddim];
        oldt = newt;
        
        x += c;
        xn += c;
    }
}

void StressGrid::DistributeElasticity_internal3D(array3_int xi, array3_int xj, array3_int xk, array3_int xl, real_int phi, real_int kappa)
{
    // this is the 3D case
    int batch_id = this->m_thread_map[std::this_thread::get_id()];
    matrix6_int * gridElast = this->p_current_grid_elborn+batch_id*this->m_ncells;

    //------------------------------------------------------------------------------------
    // Calculate the stress tensor
    array3_int diff, diff2;
    real_int rinv, rinv2, rinv3;
    diffarray3( xj, xi, diff, this->m_box, this->m_periodic);
    diffarray3( xl, xk, diff2, this->m_box, this->m_periodic);
    rinv = realval_int(1.0)/normarray3(diff);
    rinv2 = (real_int)(kappa)*rinv/normarray3(diff2);
    rinv3 = (real_int)(phi)*rinv*rinv*rinv;

    // Stiffness matrix in Voigt notation
    // 0 = xx; 1 = yy; 2 = zz; 3 = yz or zy; 4 = xz or zx; 5 = xy or yx
    // All indices                         Voigt indices           Stress indices
    // ( xxxx xxyy xxzz xxyz xxxz xxxy ) = ( 00 01 02 03 04 05 ) = [ 0000 0011 0022 0012 0002 0001 ]
    // (      yyyy yyzz yyyz yyxz yyxy ) = (    11 12 13 14 15 ) = [      1111 1122 1112 1102 1101 ]
    // (           zzzz zzyz zzxz zzxy ) = (       22 23 24 25 ) = [           2222 2212 2202 2201 ]
    // (                yzyz yzxz yzxy ) = (          33 34 35 ) = [                1212 1202 1201 ]
    // (                     xzxz xzxy ) = (             44 45 ) = [                     0202 0201 ]
    // (                          xyxy ) = (                55 ) = [                          0101 ]

    matrix6_int elast;
    elast[0][0] = diff[0]*diff[0]*diff2[0]*diff2[0]*rinv2 - diff[0]*diff[0]*diff[0]*diff[0]*rinv3;
    elast[0][1] = diff[0]*diff[0]*diff2[1]*diff2[1]*rinv2 - diff[0]*diff[0]*diff[1]*diff[1]*rinv3;
    elast[0][2] = diff[0]*diff[0]*diff2[2]*diff2[2]*rinv2 - diff[0]*diff[0]*diff[2]*diff[2]*rinv3;
    elast[0][3] = diff[0]*diff[0]*diff2[1]*diff2[2]*rinv2 - diff[0]*diff[0]*diff[1]*diff[2]*rinv3;
    elast[0][4] = diff[0]*diff[0]*diff2[0]*diff2[2]*rinv2 - diff[0]*diff[0]*diff[0]*diff[2]*rinv3;
    elast[0][5] = diff[0]*diff[0]*diff2[0]*diff2[1]*rinv2 - diff[0]*diff[0]*diff[0]*diff[1]*rinv3;
    elast[1][1] = diff[1]*diff[1]*diff2[1]*diff2[1]*rinv2 - diff[1]*diff[1]*diff[1]*diff[1]*rinv3;
    elast[1][2] = diff[1]*diff[1]*diff2[2]*diff2[2]*rinv2 - diff[1]*diff[1]*diff[2]*diff[2]*rinv3;
    elast[1][3] = diff[1]*diff[1]*diff2[1]*diff2[2]*rinv2 - diff[1]*diff[1]*diff[1]*diff[2]*rinv3;
    elast[1][4] = diff[1]*diff[1]*diff2[0]*diff2[2]*rinv2 - diff[1]*diff[1]*diff[0]*diff[2]*rinv3;
    elast[1][5] = diff[1]*diff[1]*diff2[0]*diff2[1]*rinv2 - diff[1]*diff[1]*diff[0]*diff[1]*rinv3;
    elast[2][2] = diff[2]*diff[2]*diff2[2]*diff2[2]*rinv2 - diff[2]*diff[2]*diff[2]*diff[2]*rinv3;
    elast[2][3] = diff[2]*diff[2]*diff2[1]*diff2[2]*rinv2 - diff[2]*diff[2]*diff[1]*diff[2]*rinv3;
    elast[2][4] = diff[2]*diff[2]*diff2[0]*diff2[2]*rinv2 - diff[2]*diff[2]*diff[0]*diff[2]*rinv3;
    elast[2][5] = diff[2]*diff[2]*diff2[0]*diff2[1]*rinv2 - diff[2]*diff[2]*diff[0]*diff[1]*rinv3;
    elast[3][3] = diff[1]*diff[2]*diff2[1]*diff2[2]*rinv2 - diff[1]*diff[2]*diff[1]*diff[2]*rinv3;
    elast[3][4] = diff[1]*diff[2]*diff2[0]*diff2[2]*rinv2 - diff[1]*diff[2]*diff[0]*diff[2]*rinv3;
    elast[3][5] = diff[1]*diff[2]*diff2[0]*diff2[1]*rinv2 - diff[1]*diff[2]*diff[0]*diff[1]*rinv3;
    elast[4][4] = diff[0]*diff[2]*diff2[0]*diff2[2]*rinv2 - diff[0]*diff[2]*diff[0]*diff[2]*rinv3;
    elast[4][5] = diff[0]*diff[2]*diff2[0]*diff2[1]*rinv2 - diff[0]*diff[2]*diff[0]*diff[1]*rinv3;
    elast[5][5] = diff[0]*diff[1]*diff2[0]*diff2[1]*rinv2 - diff[0]*diff[1]*diff[0]*diff[1]*rinv3;

    // calculate the grid coordinates (no pbc) for the extreme points
    iarray x, xn, i2, c;
    array3_int t,t_c1,t_c2,d_cgrid;
    x[0] = this->m_nxyz[0] * xi[0] * this->m_invbox[0][0] - (xi[0] < realval_int(0.0) );
    x[1] = this->m_nxyz[1] * xi[1] * this->m_invbox[1][1] - (xi[1] < realval_int(0.0) );
    x[2] = this->m_nxyz[2] * xi[2] * this->m_invbox[2][2] - (xi[2] < realval_int(0.0) );
    i2[0] = this->m_nxyz[0] * xj[0] * this->m_invbox[0][0] - (xj[0] < realval_int(0.0) );
    i2[1] = this->m_nxyz[1] * xj[1] * this->m_invbox[1][1] - (xj[1] < realval_int(0.0) );
    i2[2] = this->m_nxyz[2] * xj[2] * this->m_invbox[2][2] - (xj[2] < realval_int(0.0) );
    c[0] = (i2[0]>x[0])-(x[0]>i2[0]);
    c[1] = (i2[1]>x[1])-(x[1]>i2[1]);
    c[2] = (i2[2]>x[2])-(x[2]>i2[2]);
    d_cgrid[0] = xi[0]-(x[0]+realval_int(0.5))*this->m_gridsp[0];
    d_cgrid[1] = xi[1]-(x[1]+realval_int(0.5))*this->m_gridsp[1];
    d_cgrid[2] = xi[2]-(x[2]+realval_int(0.5))*this->m_gridsp[2];
    xn[0] = x[0]+(c[0]+1)/2;
    xn[1] = x[1]+(c[1]+1)/2;
    xn[2] = x[2]+(c[2]+1)/2;

    // calculate parametric time in each dimension, and related constants
    t_c1[0] = xi[0] / (xi[0]-xj[0]);
    t_c1[1] = xi[1] / (xi[1]-xj[1]);
    t_c1[2] = xi[2] / (xi[2]-xj[2]);
    t_c2[0] = this->m_gridsp[0] / (xi[0]-xj[0]);
    t_c2[1] = this->m_gridsp[1] / (xi[1]-xj[1]);
    t_c2[2] = this->m_gridsp[2] / (xi[2]-xj[2]);
    t[0] = (c[0] == 0) ? realval_int(1.1) : t_c1[0]-xn[0]*t_c2[0];
    t[1] = (c[1] == 0) ? realval_int(1.1) : t_c1[1]-xn[1]*t_c2[1];
    t[2] = (c[2] == 0) ? realval_int(1.1) : t_c1[2]-xn[2]*t_c2[2];

    //------------------------------------------------------------------------------------
    // Distribute the stress

    // now the position/spatial constants
    const real_int C = realval_int(0.125)*this->m_invgridsp*this->m_invgridsp;
    const real_int axy = diff[0]*diff[1];
    const real_int axz = diff[0]*diff[2];
    const real_int ayz = diff[1]*diff[2];
    const real_int axyz = diff[0]*ayz; 
    
    // track previous time of crossing
    real_int oldt = realval_int(0.0); 
    
    const int iterations = c[0]*(i2[0]-x[0]) + c[1]*(i2[1]-x[1]) + c[2]*(i2[2]-x[2]);
    for (int count = 0; count <= iterations; ++count)
    {
        // figure out index
        const int cmp0x = ((t[0]<t[1]+mds_eps) + (t[0]<t[2]+mds_eps))/2;
        const int cmp1x = ((t[1]<t[0]+mds_eps) + (t[1]<t[2]+mds_eps))/2;
        const int cmp2x = ((t[2]<t[0]+mds_eps) + (t[2]<t[1]+mds_eps))/2;
        const int iX = (1-cmp0x)*(cmp1x+2*(1-cmp1x)*cmp2x);
        const real_int newt = (iterations == count) ? realval_int(1.0) : t[iX];

        // work out the parametric time constants
        const real_int t12 = oldt*oldt;
        const real_int t22 = newt*newt;
        const real_int dt1 = newt - oldt;
        const real_int dt2 = t22 - t12;
        const real_int dt3 = realval_int(4.0)*(t22*newt - t12*oldt)/realval_int(3.0);
        const real_int dt4 = t22*t22 - t12*t12;

        // additional constants
        const real_int bxy = d_cgrid[0]*d_cgrid[1];
        const real_int bxz = d_cgrid[0]*d_cgrid[2];
        const real_int byz = d_cgrid[1]*d_cgrid[2];
        const real_int bxyz = d_cgrid[0]*byz;
    
        const int iip1 = ((x[0] + 1 + this->m_nxyz[0]) % this->m_nxyz[0])*this->m_nxyz[1]*this->m_nxyz[2];
        const int jjp1 = ((x[1] + 1 + this->m_nxyz[1]) % this->m_nxyz[1])*this->m_nxyz[2];
        const int kkp1 = ((x[2] + 1 + this->m_nxyz[2]) % this->m_nxyz[2]);
        const int iim1 = ((x[0] + this->m_nxyz[0]) % this->m_nxyz[0])*this->m_nxyz[1]*this->m_nxyz[2];
        const int jjm1 = ((x[1] + this->m_nxyz[1]) % this->m_nxyz[1])*this->m_nxyz[2];
        const int kkm1 = ((x[2] + this->m_nxyz[2]) % this->m_nxyz[2]);

        // the composite constants in terms of i, j, k
        const real_int D[8] = {
            realval_int(8.0)*bxyz*dt1 + realval_int(4.0)*(diff[0]*byz+diff[1]*bxz+diff[2]*bxy)*dt2
                + realval_int(2.0)*(d_cgrid[0]*ayz+d_cgrid[1]*axz+d_cgrid[2]*axy)*dt3 + realval_int(2.0)*axyz*dt4,
            this->m_gridsp[0]*(realval_int(4.0)*byz*dt1 + realval_int(2.0)*(diff[1]*d_cgrid[2]+diff[2]*d_cgrid[1])*dt2 + ayz*dt3),
            this->m_gridsp[1]*(realval_int(4.0)*bxz*dt1 + realval_int(2.0)*(diff[0]*d_cgrid[2]+diff[2]*d_cgrid[0])*dt2 + axz*dt3),
            this->m_gridsp[2]*(realval_int(4.0)*bxy*dt1 + realval_int(2.0)*(diff[0]*d_cgrid[1]+diff[1]*d_cgrid[0])*dt2 + axy*dt3),
            this->m_gridsp[3]*(realval_int(2.0)*d_cgrid[2]*dt1+diff[2]*dt2),
            this->m_gridsp[4]*(realval_int(2.0)*d_cgrid[1]*dt1+diff[1]*dt2),
            this->m_gridsp[5]*(realval_int(2.0)*d_cgrid[0]*dt1+diff[0]*dt2),
            this->m_gridsp[6]*dt1,
        };

        // perform the sums into the grid
        scalesummatrix6(C*( D[0] + D[1] + D[2] + D[3] + D[4] + D[5] + D[6] + D[7]), elast, gridElast[iip1 + jjp1 + kkp1]);
        scalesummatrix6(C*(-D[0] - D[1] - D[2] + D[3] - D[4] + D[5] + D[6] + D[7]), elast, gridElast[iip1 + jjp1 + kkm1]);
        scalesummatrix6(C*(-D[0] - D[1] + D[2] - D[3] + D[4] - D[5] + D[6] + D[7]), elast, gridElast[iip1 + jjm1 + kkp1]);
        scalesummatrix6(C*( D[0] + D[1] - D[2] - D[3] - D[4] - D[5] + D[6] + D[7]), elast, gridElast[iip1 + jjm1 + kkm1]);
        scalesummatrix6(C*(-D[0] + D[1] - D[2] - D[3] + D[4] + D[5] - D[6] + D[7]), elast, gridElast[iim1 + jjp1 + kkp1]);
        scalesummatrix6(C*( D[0] - D[1] + D[2] - D[3] - D[4] + D[5] - D[6] + D[7]), elast, gridElast[iim1 + jjp1 + kkm1]);
        scalesummatrix6(C*( D[0] - D[1] - D[2] + D[3] + D[4] - D[5] - D[6] + D[7]), elast, gridElast[iim1 + jjm1 + kkp1]);
        scalesummatrix6(C*(-D[0] + D[1] + D[2] + D[3] - D[4] - D[5] - D[6] + D[7]), elast, gridElast[iim1 + jjm1 + kkm1]);

        d_cgrid[iX] -= c[iX] * m_gridsp[iX];
        oldt = t[iX];
        
        x[iX] += c[iX];
        xn[iX] += c[iX];

        // Next cross point:
        t[iX] = t_c1[iX]-xn[iX]*t_c2[iX];
    }
}

void StressGrid::DistributeElasticity(array3_ext xi_ext, array3_ext xj_ext, array3_ext xk_ext, array3_ext xl_ext, real_ext phi_ext, real_ext kappa_ext)
{
    if (true == this->m_disable)
        return;

    array3_int xi, xj, xk, xl;
    real_int phi, kappa;

    copyarray3(xi_ext, xi);
    copyarray3(xj_ext, xj);
    copyarray3(xk_ext, xk);
    copyarray3(xl_ext, xl);
    phi = (real_int)phi_ext;
    kappa = (real_int)kappa_ext;
    
    if (this->m_griddim == mds_griddim_xyz)
        DistributeElasticity_internal3D(xi, xj, xk, xl, phi, kappa);
    else
        DistributeElasticity_internal1D(xi, xj, xk, xl, phi, kappa);
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
    if (true == this->m_disable)
        return;

    matrix3_int stress;

    // Spreads the velocity in one point
    iarray i1;
    array3_int xc,xd;
    int index, iip1, iim1, jjp1, jjm1, kkp1, kkm1;
    real_int factor,C;

    if ( !this->m_ierr )
    {
        // get the batch id
        int batch_id = this->m_thread_map[std::this_thread::get_id()];
            
        if (vb == nullptr)
        {
            stress[0][0] = (real_int)(-mass)*(real_int)(va[0])*(real_int)(va[0]);
            stress[0][1] = (real_int)(-mass)*(real_int)(va[0])*(real_int)(va[1]);
            stress[0][2] = (real_int)(-mass)*(real_int)(va[0])*(real_int)(va[2]);
            stress[1][0] = (real_int)(-mass)*(real_int)(va[1])*(real_int)(va[0]);
            stress[1][1] = (real_int)(-mass)*(real_int)(va[1])*(real_int)(va[1]);
            stress[1][2] = (real_int)(-mass)*(real_int)(va[1])*(real_int)(va[2]);
            stress[2][0] = (real_int)(-mass)*(real_int)(va[2])*(real_int)(va[0]);
            stress[2][1] = (real_int)(-mass)*(real_int)(va[2])*(real_int)(va[1]);
            stress[2][2] = (real_int)(-mass)*(real_int)(va[2])*(real_int)(va[2]);
        }
        else
        {
            stress[0][0] = (real_int)(-mass)*((real_int)(va[0])*(real_int)(va[0]) + (real_int)(vb[0])*(real_int)(vb[0]))/realval_int(2.0);
            stress[0][1] = (real_int)(-mass)*((real_int)(va[0])*(real_int)(va[1]) + (real_int)(vb[0])*(real_int)(vb[1]))/realval_int(2.0);
            stress[0][2] = (real_int)(-mass)*((real_int)(va[0])*(real_int)(va[2]) + (real_int)(vb[0])*(real_int)(vb[2]))/realval_int(2.0);
            stress[1][0] = (real_int)(-mass)*((real_int)(va[1])*(real_int)(va[0]) + (real_int)(vb[1])*(real_int)(vb[0]))/realval_int(2.0);
            stress[1][1] = (real_int)(-mass)*((real_int)(va[1])*(real_int)(va[1]) + (real_int)(vb[1])*(real_int)(vb[1]))/realval_int(2.0);
            stress[1][2] = (real_int)(-mass)*((real_int)(va[1])*(real_int)(va[2]) + (real_int)(vb[1])*(real_int)(vb[2]))/realval_int(2.0);
            stress[2][0] = (real_int)(-mass)*((real_int)(va[2])*(real_int)(va[0]) + (real_int)(vb[2])*(real_int)(vb[0]))/realval_int(2.0);
            stress[2][1] = (real_int)(-mass)*((real_int)(va[2])*(real_int)(va[1]) + (real_int)(vb[2])*(real_int)(vb[1]))/realval_int(2.0);
            stress[2][2] = (real_int)(-mass)*((real_int)(va[2])*(real_int)(va[2]) + (real_int)(vb[2])*(real_int)(vb[2]))/realval_int(2.0);
        }

        // If spatatom==mds_spat distribute the stress spatially following Noll's procedure
        if (this->m_spatatom == mds_spat)
        {
            // Get the coordinates of the point in the grid
            i1[0] = this->m_nxyz[0] * x[0] * this->m_invbox[0][0] - (x[0] < 0.0);
            i1[1] = this->m_nxyz[1] * x[1] * this->m_invbox[1][1] - (x[1] < 0.0);
            i1[2] = this->m_nxyz[2] * x[2] * this->m_invbox[2][2] - (x[2] < 0.0);
            
            // and the index constants
            iip1 = ((i1[0] + 1 + this->m_nxyz[0]) % this->m_nxyz[0])*this->m_nxyz[1]*this->m_nxyz[2];
            jjp1 = ((i1[1] + 1 + this->m_nxyz[1]) % this->m_nxyz[1])*this->m_nxyz[2];
            kkp1 = ((i1[2] + 1 + this->m_nxyz[2]) % this->m_nxyz[2]);
            iim1 = ((i1[0] + this->m_nxyz[0]) % this->m_nxyz[0])*this->m_nxyz[1]*this->m_nxyz[2];
            jjm1 = ((i1[1] + this->m_nxyz[1]) % this->m_nxyz[1])*this->m_nxyz[2];
            kkm1 = ((i1[2] + this->m_nxyz[2]) % this->m_nxyz[2]);
            
            // xc = vector from the corner of the point to the corner of the cell
            C = this->m_invgridsp * this->m_invgridsp;
            xc[0] = (real_int)(x[0])-this->m_gridsp[0]*i1[0];
            xc[1] = (real_int)(x[1])-this->m_gridsp[1]*i1[1];
            xc[2] = (real_int)(x[2])-this->m_gridsp[2]*i1[2];
            xd[0] = xc[0]-this->m_gridsp[0];
            xd[1] = xc[1]-this->m_gridsp[1];
            xd[2] = xc[2]-this->m_gridsp[2];
            
            // select grid based on batch index
            matrix3_int * grid = this->p_current_grid+batch_id*this->m_ncells;

            // Spread it
            scalesummatrix3( C*xc[0]*xc[1]*xc[2],stress,grid[iip1+jjp1+kkp1]);
            scalesummatrix3(-C*xc[0]*xc[1]*xd[2],stress,grid[iip1+jjp1+kkm1]);
            scalesummatrix3(-C*xc[0]*xd[1]*xc[2],stress,grid[iip1+jjm1+kkp1]);
            scalesummatrix3( C*xc[0]*xd[1]*xd[2],stress,grid[iip1+jjm1+kkm1]);
            scalesummatrix3(-C*xd[0]*xc[1]*xc[2],stress,grid[iim1+jjp1+kkp1]);
            scalesummatrix3( C*xd[0]*xc[1]*xd[2],stress,grid[iim1+jjp1+kkm1]);
            scalesummatrix3( C*xd[0]*xd[1]*xc[2],stress,grid[iim1+jjm1+kkp1]);
            scalesummatrix3(-C*xd[0]*xd[1]*xd[2],stress,grid[iim1+jjm1+kkm1]);
        }
        else if (this->m_spatatom == mds_atom)
        {
            if (atomID == -1)
            {
                std::cout << "ERROR:: Unknown atomID for kinetic contribution. Cannot calculate the stress/atom.";
                return;
            }
            else
            {
                summatrix3(this->p_current_grid[atomID],stress,this->p_current_grid[atomID]);
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
    if (true == this->m_disable)
        return;

    matrix6_int elast;

    // Spreads the velocity in one point
    iarray i1;
    array3_int xc,xd,pa,pb;
    int index, iip1, iim1, jjp1, jjm1, kkp1, kkm1;
    real_int factor,C;
    if ( !this->m_ierr )
    {
        // get the batch id
        int batch_id = this->m_thread_map[std::this_thread::get_id()];
        // select grid based on batch index
        matrix6_int * grid = this->p_current_grid_elkin+batch_id*this->m_ncells;

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
            /*                                                                                               ijkl
            elast[0][0] = mass*(1*va[0]*va[0] + 1*va[0]*va[0] + 1*va[0]*va[0] + 1*va[0]*va[0]); //c_xxxx = c_0000
            elast[0][1] = mass*(0*va[0]*va[1] + 0*va[0]*va[1] + 0*va[0]*va[1] + 0*va[0]*va[1]); //c_xxyy = c_0011
            elast[0][2] = mass*(0*va[0]*va[2] + 0*va[0]*va[2] + 0*va[0]*va[2] + 0*va[0]*va[2]); //c_xxzz = c_0022
            elast[0][3] = mass*(0*va[0]*va[1] + 0*va[0]*va[2] + 0*va[0]*va[1] + 0*va[0]*va[2]); //c_xxyz = c_0012
            elast[0][4] = mass*(0*va[0]*va[0] + 1*va[0]*va[2] + 0*va[0]*va[0] + 1*va[0]*va[2]); //c_xxxz = c_0002
            elast[0][5] = mass*(0*va[0]*va[0] + 1*va[0]*va[1] + 0*va[0]*va[0] + 1*va[0]*va[1]); //c_xxxy = c_0001
            elast[1][1] = mass*(1*va[1]*va[1] + 1*va[1]*va[1] + 1*va[1]*va[1] + 1*va[1]*va[1]); //c_yyyy = c_1111
            elast[1][2] = mass*(0*va[1]*va[2] + 0*va[1]*va[2] + 0*va[1]*va[2] + 0*va[1]*va[2]); //c_yyzz = c_1122
            elast[1][3] = mass*(0*va[1]*va[1] + 1*va[1]*va[2] + 0*va[1]*va[1] + 1*va[1]*va[2]); //c_yyyz = c_1112
            elast[1][4] = mass*(0*va[1]*va[0] + 0*va[1]*va[2] + 0*va[1]*va[0] + 0*va[1]*va[2]); //c_yyxz = c_1102
            elast[1][5] = mass*(1*va[1]*va[0] + 0*va[1]*va[1] + 1*va[1]*va[0] + 0*va[1]*va[1]); //c_yyxy = c_1101
            elast[2][2] = mass*(1*va[2]*va[2] + 1*va[2]*va[2] + 1*va[2]*va[2] + 1*va[2]*va[2]); //c_zzzz = c_2222
            elast[2][3] = mass*(1*va[2]*va[1] + 0*va[2]*va[2] + 1*va[2]*va[1] + 0*va[2]*va[2]); //c_zzyz = c_2212
            elast[2][4] = mass*(1*va[2]*va[0] + 0*va[2]*va[2] + 1*va[2]*va[0] + 0*va[2]*va[2]); //c_zzxz = c_2202
            elast[2][5] = mass*(0*va[2]*va[0] + 0*va[2]*va[1] + 0*va[2]*va[0] + 0*va[2]*va[1]); //c_zzxy = c_2201
            elast[3][3] = mass*(0*va[2]*va[1] + 0*va[1]*va[2] + 1*va[1]*va[1] + 1*va[2]*va[2]); //c_yzyz = c_1212
            elast[3][4] = mass*(0*va[2]*va[0] + 0*va[1]*va[2] + 1*va[1]*va[0] + 0*va[2]*va[2]); //c_yzxz = c_1202
            elast[3][5] = mass*(1*va[2]*va[0] + 0*va[1]*va[1] + 0*va[1]*va[0] + 0*va[2]*va[1]); //c_yzxy = c_1201
            elast[4][4] = mass*(0*va[2]*va[0] + 0*va[0]*va[2] + 1*va[0]*va[0] + 1*va[2]*va[2]); //c_xzxz = c_0202
            elast[4][5] = mass*(0*va[2]*va[0] + 0*va[0]*va[1] + 0*va[0]*va[0] + 1*va[2]*va[1]); //c_xzxy = c_0201
            elast[5][5] = mass*(0*va[1]*va[0] + 0*va[0]*va[1] + 1*va[0]*va[0] + 1*va[1]*va[1]); //c_xyxy = c_0101
            */
            elast[0][0] = (real_int)(mass)*realval_int(4.0)*(real_int)(va[0])*(real_int)(va[0]); //c_xxxx = c_0000
            elast[1][1] = (real_int)(mass)*realval_int(4.0)*(real_int)(va[1])*(real_int)(va[1]); //c_yyyy = c_1111
            elast[2][2] = (real_int)(mass)*realval_int(4.0)*(real_int)(va[2])*(real_int)(va[2]); //c_zzzz = c_2222

            elast[0][4] = (real_int)(mass)*((real_int)(va[0])*(real_int)(va[2]) + (real_int)(va[0])*(real_int)(va[2])); //c_xxxz = c_0002
            elast[0][5] = (real_int)(mass)*((real_int)(va[0])*(real_int)(va[1]) + (real_int)(va[0])*(real_int)(va[1])); //c_xxxy = c_0001
            elast[1][3] = (real_int)(mass)*((real_int)(va[1])*(real_int)(va[2]) + (real_int)(va[1])*(real_int)(va[2])); //c_yyyz = c_1112
            elast[1][5] = (real_int)(mass)*((real_int)(va[1])*(real_int)(va[0]) + (real_int)(va[1])*(real_int)(va[0])); //c_yyxy = c_1101
            elast[2][3] = (real_int)(mass)*((real_int)(va[2])*(real_int)(va[1]) + (real_int)(va[2])*(real_int)(va[1])); //c_zzyz = c_2212
            elast[2][4] = (real_int)(mass)*((real_int)(va[2])*(real_int)(va[0]) + (real_int)(va[2])*(real_int)(va[0])); //c_zzxz = c_2202
            elast[4][4] = (real_int)(mass)*((real_int)(va[0])*(real_int)(va[0]) + (real_int)(va[2])*(real_int)(va[2])); //c_xzxz = c_0202
            elast[5][5] = (real_int)(mass)*((real_int)(va[0])*(real_int)(va[0]) + (real_int)(va[1])*(real_int)(va[1])); //c_xyxy = c_0101
            elast[3][3] = (real_int)(mass)*((real_int)(va[1])*(real_int)(va[1]) + (real_int)(va[2])*(real_int)(va[2])); //c_yzyz = c_1212

            elast[3][4] = (real_int)(mass)*(real_int)(va[1])*(real_int)(va[0]); //c_yzxz = c_1202
            elast[3][5] = (real_int)(mass)*(real_int)(va[2])*(real_int)(va[0]); //c_yzxy = c_1201
            elast[4][5] = (real_int)(mass)*(real_int)(va[2])*(real_int)(va[1]); //c_xzxy = c_0201

            elast[0][1] = realval_int(0.0); //c_xxyy = c_0011
            elast[0][2] = realval_int(0.0); //c_xxzz = c_0022
            elast[0][3] = realval_int(0.0); //c_xxyz = c_0012
            elast[1][2] = realval_int(0.0); //c_yyzz = c_1122
            elast[1][4] = realval_int(0.0); //c_yyxz = c_1102
            elast[2][5] = realval_int(0.0); //c_zzxy = c_2201
        }
        else
        {
            /*
            elast[0][0] = mass*(1*va[0]*va[0] + 1*va[0]*va[0] + 1*va[0]*va[0] + 1*va[0]*va[0] + 1*vb[0]*vb[0] + 1*vb[0]*vb[0] + 1*vb[0]*vb[0] + 1*vb[0]*vb[0])/2.0; //c_xxxx = c_0000
            elast[0][1] = mass*(0*va[0]*va[1] + 0*va[0]*va[1] + 0*va[0]*va[1] + 0*va[0]*va[1] + 0*vb[0]*vb[1] + 0*vb[0]*vb[1] + 0*vb[0]*vb[1] + 0*vb[0]*vb[1])/2.0; //c_xxyy = c_0011
            elast[0][2] = mass*(0*va[0]*va[2] + 0*va[0]*va[2] + 0*va[0]*va[2] + 0*va[0]*va[2] + 0*vb[0]*vb[2] + 0*vb[0]*vb[2] + 0*vb[0]*vb[2] + 0*vb[0]*vb[2])/2.0; //c_xxzz = c_0022
            elast[0][3] = mass*(0*va[0]*va[1] + 0*va[0]*va[2] + 0*va[0]*va[1] + 0*va[0]*va[2] + 0*vb[0]*vb[1] + 0*vb[0]*vb[2] + 0*vb[0]*vb[1] + 0*vb[0]*vb[2])/2.0; //c_xxyz = c_0012
            elast[0][4] = mass*(0*va[0]*va[0] + 1*va[0]*va[2] + 0*va[0]*va[0] + 1*va[0]*va[2] + 0*vb[0]*vb[0] + 1*vb[0]*vb[2] + 0*vb[0]*vb[0] + 1*vb[0]*vb[2])/2.0; //c_xxxz = c_0002
            elast[0][5] = mass*(0*va[0]*va[0] + 1*va[0]*va[1] + 0*va[0]*va[0] + 1*va[0]*va[1] + 0*vb[0]*vb[0] + 1*vb[0]*vb[1] + 0*vb[0]*vb[0] + 1*vb[0]*vb[1])/2.0; //c_xxxy = c_0001
            elast[1][1] = mass*(1*va[1]*va[1] + 1*va[1]*va[1] + 1*va[1]*va[1] + 1*va[1]*va[1] + 1*vb[1]*vb[1] + 1*vb[1]*vb[1] + 1*vb[1]*vb[1] + 1*vb[1]*vb[1])/2.0; //c_yyyy = c_1111
            elast[1][2] = mass*(0*va[1]*va[2] + 0*va[1]*va[2] + 0*va[1]*va[2] + 0*va[1]*va[2] + 0*vb[1]*vb[2] + 0*vb[1]*vb[2] + 0*vb[1]*vb[2] + 0*vb[1]*vb[2])/2.0; //c_yyzz = c_1122
            elast[1][3] = mass*(0*va[1]*va[1] + 1*va[1]*va[2] + 0*va[1]*va[1] + 1*va[1]*va[2] + 0*vb[1]*vb[1] + 1*vb[1]*vb[2] + 0*vb[1]*vb[1] + 1*vb[1]*vb[2])/2.0; //c_yyyz = c_1112
            elast[1][4] = mass*(0*va[1]*va[0] + 0*va[1]*va[2] + 0*va[1]*va[0] + 0*va[1]*va[2] + 0*vb[1]*vb[0] + 0*vb[1]*vb[2] + 0*vb[1]*vb[0] + 0*vb[1]*vb[2])/2.0; //c_yyxz = c_1102
            elast[1][5] = mass*(1*va[1]*va[0] + 0*va[1]*va[1] + 1*va[1]*va[0] + 0*va[1]*va[1] + 1*vb[1]*vb[0] + 0*vb[1]*vb[1] + 1*vb[1]*vb[0] + 0*vb[1]*vb[1])/2.0; //c_yyxy = c_1101
            elast[2][2] = mass*(1*va[2]*va[2] + 1*va[2]*va[2] + 1*va[2]*va[2] + 1*va[2]*va[2] + 1*vb[2]*vb[2] + 1*vb[2]*vb[2] + 1*vb[2]*vb[2] + 1*vb[2]*vb[2])/2.0; //c_zzzz = c_2222
            elast[2][3] = mass*(1*va[2]*va[1] + 0*va[2]*va[2] + 1*va[2]*va[1] + 0*va[2]*va[2] + 1*vb[2]*vb[1] + 0*vb[2]*vb[2] + 1*vb[2]*vb[1] + 0*vb[2]*vb[2])/2.0; //c_zzyz = c_2212
            elast[2][4] = mass*(1*va[2]*va[0] + 0*va[2]*va[2] + 1*va[2]*va[0] + 0*va[2]*va[2] + 1*vb[2]*vb[0] + 0*vb[2]*vb[2] + 1*vb[2]*vb[0] + 0*vb[2]*vb[2])/2.0; //c_zzxz = c_2202
            elast[2][5] = mass*(0*va[2]*va[0] + 0*va[2]*va[1] + 0*va[2]*va[0] + 0*va[2]*va[1] + 0*vb[2]*vb[0] + 0*vb[2]*vb[1] + 0*vb[2]*vb[0] + 0*vb[2]*vb[1])/2.0; //c_zzxy = c_2201
            elast[3][3] = mass*(0*va[2]*va[1] + 0*va[1]*va[2] + 1*va[1]*va[1] + 1*va[2]*va[2] + 0*vb[2]*vb[1] + 0*vb[1]*vb[2] + 1*vb[1]*vb[1] + 1*vb[2]*vb[2])/2.0; //c_yzyz = c_1212
            elast[3][4] = mass*(0*va[2]*va[0] + 0*va[1]*va[2] + 1*va[1]*va[0] + 0*va[2]*va[2] + 0*vb[2]*vb[0] + 0*vb[1]*vb[2] + 1*vb[1]*vb[0] + 0*vb[2]*vb[2])/2.0; //c_yzxz = c_1202
            elast[3][5] = mass*(1*va[2]*va[0] + 0*va[1]*va[1] + 0*va[1]*va[0] + 0*va[2]*va[1] + 1*vb[2]*vb[0] + 0*vb[1]*vb[1] + 0*vb[1]*vb[0] + 0*vb[2]*vb[1])/2.0; //c_yzxy = c_1201
            elast[4][4] = mass*(0*va[2]*va[0] + 0*va[0]*va[2] + 1*va[0]*va[0] + 1*va[2]*va[2] + 0*vb[2]*vb[0] + 0*vb[0]*vb[2] + 1*vb[0]*vb[0] + 1*vb[2]*vb[2])/2.0; //c_xzxz = c_0202
            elast[4][5] = mass*(0*va[2]*va[0] + 0*va[0]*va[1] + 0*va[0]*va[0] + 1*va[2]*va[1] + 0*vb[2]*vb[0] + 0*vb[0]*vb[1] + 0*vb[0]*vb[0] + 1*vb[2]*vb[1])/2.0; //c_xzxy = c_0201
            elast[5][5] = mass*(0*va[1]*va[0] + 0*va[0]*va[1] + 1*va[0]*va[0] + 1*va[1]*va[1] + 0*vb[1]*vb[0] + 0*vb[0]*vb[1] + 1*vb[0]*vb[0] + 1*vb[1]*vb[1])/2.0; //c_xyxy = c_0101
            */
            elast[0][0] = (real_int)(mass)*realval_int(4.0)*((real_int)(va[0])*(real_int)(va[0]) + (real_int)(vb[0])*(real_int)(vb[0]))/realval_int(2.0); //c_xxxx = c_0000
            elast[1][1] = (real_int)(mass)*realval_int(4.0)*((real_int)(va[1])*(real_int)(va[1]) + (real_int)(vb[1])*(real_int)(vb[1]))/realval_int(2.0); //c_yyyy = c_1111
            elast[2][2] = (real_int)(mass)*realval_int(4.0)*((real_int)(va[2])*(real_int)(va[2]) + (real_int)(vb[2])*(real_int)(vb[2]))/realval_int(2.0); //c_zzzz = c_2222

            elast[0][4] = (real_int)(mass)*((real_int)(va[0])*(real_int)(va[2]) + (real_int)(va[0])*(real_int)(va[2]) + (real_int)(vb[0])*(real_int)(vb[2]) + (real_int)(vb[0])*(real_int)(vb[2]))/realval_int(2.0); //c_xxxz = c_0002
            elast[0][5] = (real_int)(mass)*((real_int)(va[0])*(real_int)(va[1]) + (real_int)(va[0])*(real_int)(va[1]) + (real_int)(vb[0])*(real_int)(vb[1]) + (real_int)(vb[0])*(real_int)(vb[1]))/realval_int(2.0); //c_xxxy = c_0001
            elast[1][3] = (real_int)(mass)*((real_int)(va[1])*(real_int)(va[2]) + (real_int)(va[1])*(real_int)(va[2]) + (real_int)(vb[1])*(real_int)(vb[2]) + (real_int)(vb[1])*(real_int)(vb[2]))/realval_int(2.0); //c_yyyz = c_1112
            elast[1][5] = (real_int)(mass)*((real_int)(va[1])*(real_int)(va[0]) + (real_int)(va[1])*(real_int)(va[0]) + (real_int)(vb[1])*(real_int)(vb[0]) + (real_int)(vb[1])*(real_int)(vb[0]))/realval_int(2.0); //c_yyxy = c_1101
            elast[2][3] = (real_int)(mass)*((real_int)(va[2])*(real_int)(va[1]) + (real_int)(va[2])*(real_int)(va[1]) + (real_int)(vb[2])*(real_int)(vb[1]) + (real_int)(vb[2])*(real_int)(vb[1]))/realval_int(2.0); //c_zzyz = c_2212
            elast[2][4] = (real_int)(mass)*((real_int)(va[2])*(real_int)(va[0]) + (real_int)(va[2])*(real_int)(va[0]) + (real_int)(vb[2])*(real_int)(vb[0]) + (real_int)(vb[2])*(real_int)(vb[0]))/realval_int(2.0); //c_zzxz = c_2202
            elast[3][3] = (real_int)(mass)*((real_int)(va[1])*(real_int)(va[1]) + (real_int)(va[2])*(real_int)(va[2]) + (real_int)(vb[1])*(real_int)(vb[1]) + (real_int)(vb[2])*(real_int)(vb[2]))/realval_int(2.0); //c_yzyz = c_1212
            elast[4][4] = (real_int)(mass)*((real_int)(va[0])*(real_int)(va[0]) + (real_int)(va[2])*(real_int)(va[2]) + (real_int)(vb[0])*(real_int)(vb[0]) + (real_int)(vb[2])*(real_int)(vb[2]))/realval_int(2.0); //c_xzxz = c_0202
            elast[5][5] = (real_int)(mass)*((real_int)(va[0])*(real_int)(va[0]) + (real_int)(va[1])*(real_int)(va[1]) + (real_int)(vb[0])*(real_int)(vb[0]) + (real_int)(vb[1])*(real_int)(vb[1]))/realval_int(2.0); //c_xyxy = c_0101

            elast[3][4] = (real_int)(mass)*((real_int)(va[1])*(real_int)(va[0]) + (real_int)(vb[1])*(real_int)(vb[0]))/realval_int(2.0); //c_yzxz = c_1202
            elast[3][5] = (real_int)(mass)*((real_int)(va[2])*(real_int)(va[0]) + (real_int)(vb[2])*(real_int)(vb[0]))/realval_int(2.0); //c_yzxy = c_1201
            elast[4][5] = (real_int)(mass)*((real_int)(va[2])*(real_int)(va[1]) + (real_int)(vb[2])*(real_int)(vb[1]))/realval_int(2.0); //c_xzxy = c_0201
            
            elast[0][1] = realval_int(0.0); //c_xxyy = c_0011
            elast[0][2] = realval_int(0.0); //c_xxzz = c_0022
            elast[0][3] = realval_int(0.0); //c_xxyz = c_0012
            elast[1][2] = realval_int(0.0); //c_yyzz = c_1122
            elast[1][4] = realval_int(0.0); //c_yyxz = c_1102
            elast[2][5] = realval_int(0.0); //c_zzxy = c_2201
        }

        // Get the coordinates of the point in the grid
        i1[0] = this->m_nxyz[0] * x[0] * this->m_invbox[0][0] - (x[0] < realval_int(0.0));
        i1[1] = this->m_nxyz[1] * x[1] * this->m_invbox[1][1] - (x[1] < realval_int(0.0));
        i1[2] = this->m_nxyz[2] * x[2] * this->m_invbox[2][2] - (x[2] < realval_int(0.0));

        // and the index constants
        iip1 = ((i1[0] + 1 + this->m_nxyz[0]) % this->m_nxyz[0])*this->m_nxyz[1]*this->m_nxyz[2];
        jjp1 = ((i1[1] + 1 + this->m_nxyz[1]) % this->m_nxyz[1])*this->m_nxyz[2];
        kkp1 = ((i1[2] + 1 + this->m_nxyz[2]) % this->m_nxyz[2]);
        iim1 = ((i1[0] + this->m_nxyz[0]) % this->m_nxyz[0])*this->m_nxyz[1]*this->m_nxyz[2];
        jjm1 = ((i1[1] + this->m_nxyz[1]) % this->m_nxyz[1])*this->m_nxyz[2];
        kkm1 = ((i1[2] + this->m_nxyz[2]) % this->m_nxyz[2]);

        // xc = vector from the corner of the point to the corner of the cell
        C = this->m_invgridsp * this->m_invgridsp;
        xc[0] = x[0]-this->m_gridsp[0]*i1[0];
        xc[1] = x[1]-this->m_gridsp[1]*i1[1];
        xc[2] = x[2]-this->m_gridsp[2]*i1[2];
        xd[0] = xc[0]-this->m_gridsp[0];
        xd[1] = xc[1]-this->m_gridsp[1];
        xd[2] = xc[2]-this->m_gridsp[2];

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

// DistributeCharge
//
// Distributes charge contributions onto a grid
// Requires:
// x          -> position of the atom
// charge     -> atomic charge
void StressGrid::DistributeCharge ( array3_ext x_ext, real_ext charge )
{
    // disabling distribute charge for now
    return;
    // Spreads the velocity in one point
    iarray i1;
    array3_int x, xc,xd;
    
    // copy the array to internal precision
    copyarray3(x_ext, x);

    int index, iip1, iim1, jjp1, jjm1, kkp1, kkm1;
    real_int factor,C;

    if ( !this->m_ierr )
    {
        // get the batch id
        int batch_id = this->m_thread_map[std::this_thread::get_id()];
        
        // select grid based on batch index
        real_int * gridc = this->p_current_gridc+batch_id*this->m_ncellsc;
            
        // Get the coordinates of the point in the grid
        i1[0] = this->m_nxyzc[0] * x[0] * this->m_invbox[0][0] - (x[0] < realval_int(0.0) );
        i1[1] = this->m_nxyzc[1] * x[1] * this->m_invbox[1][1] - (x[1] < realval_int(0.0) );
        i1[2] = this->m_nxyzc[2] * x[2] * this->m_invbox[2][2] - (x[2] < realval_int(0.0) );
        
        // and the index constants
        iip1 = ((i1[0] + 1 + this->m_nxyzc[0]) % this->m_nxyzc[0])*this->m_nxyzc[1]*this->m_nxyzc[2];
        jjp1 = ((i1[1] + 1 + this->m_nxyzc[1]) % this->m_nxyzc[1])*this->m_nxyzc[2];
        kkp1 = ((i1[2] + 1 + this->m_nxyzc[2]) % this->m_nxyzc[2]);
        iim1 = ((i1[0] + this->m_nxyzc[0]) % this->m_nxyzc[0])*this->m_nxyzc[1]*this->m_nxyzc[2];
        jjm1 = ((i1[1] + this->m_nxyzc[1]) % this->m_nxyzc[1])*this->m_nxyzc[2];
        kkm1 = ((i1[2] + this->m_nxyzc[2]) % this->m_nxyzc[2]);
        
        // xc = vector from the corner of the point to the corner of the cell
        C = this->m_invgridspc * this->m_invgridspc;
        xc[0] = x[0]-this->m_gridspc[0]*i1[0];
        xc[1] = x[1]-this->m_gridspc[1]*i1[1];
        xc[2] = x[2]-this->m_gridspc[2]*i1[2];
        xd[0] = xc[0]-this->m_gridspc[0];
        xd[1] = xc[1]-this->m_gridspc[1];
        xd[2] = xc[2]-this->m_gridspc[2];
        
        // Spread it
        gridc[iip1+jjp1+kkp1] +=  C*xc[0]*xc[1]*xc[2]*charge;
        gridc[iip1+jjp1+kkm1] += -C*xc[0]*xc[1]*xd[2]*charge;
        gridc[iip1+jjm1+kkp1] += -C*xc[0]*xd[1]*xc[2]*charge;
        gridc[iip1+jjm1+kkm1] +=  C*xc[0]*xd[1]*xd[2]*charge;
        gridc[iim1+jjp1+kkp1] += -C*xd[0]*xc[1]*xc[2]*charge;
        gridc[iim1+jjp1+kkm1] +=  C*xd[0]*xc[1]*xd[2]*charge;
        gridc[iim1+jjm1+kkp1] +=  C*xd[0]*xd[1]*xc[2]*charge;
        gridc[iim1+jjm1+kkm1] += -C*xd[0]*xd[1]*xd[2]*charge;
    }
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
//SPECIFIC DECOMPOSITIONS FOR 3,4 AND 5 AND N PARTICLES


// Decompose 3-body potentials (angles)
void StressGrid::DistributeN3( array3_int Ra, array3_int Rb, array3_int Rc, array3_int Fa, array3_int Fb, array3_int Fc, int batch_id )
{
    //Counter
    int i;

    //************************************************************************************
    // UNIT vectors between particles
    array3_int AB, AC, BC;

    // Distances
    real_int normAB,normAC,normBC;

    // (Covariant) Central Force decomposition
    real_int lab, lac, lbc;

    array3_int Fij;
    //************************************************************************************

    //Dimension and number of particles
    int nDim = 3;
    int nPart = 3;

    //Number of rows and columns
    int nRow = mds_nrow3, nCol = mds_ncol3, nRHS = 1;

    // Matrix of the system (9 equations x 3 unknowns)
    double M[mds_nrow3*mds_ncol3];
    // Vector, we want to solve M*x = b
    double b[mds_nrow3], s[mds_ncol3];

    // If the force decomposition is cCFD or CFD
    if(this->m_fdecomp == mds_ccfd || this->m_fdecomp == mds_ncfd)
    {
        diffarray3(Rb, Ra, AB, this->m_box, this->m_periodic);
        diffarray3(Rc, Ra, AC, this->m_box, this->m_periodic);
        diffarray3(Rc, Rb, BC, this->m_box, this->m_periodic);

        normAB=normarray3(AB);
        normAC=normarray3(AC);
        normBC=normarray3(BC);

        for ( i = 0; i < nCol*nRow; i++ )
        {
            M[i] = 0.0;
        }

        //Force on particle 1:
        M[nRow*0+0] = (double)AB[0]; M[nRow*1+0] = (double)AC[0];
        M[nRow*0+1] = (double)AB[1]; M[nRow*1+1] = (double)AC[1];
        M[nRow*0+2] = (double)AB[2]; M[nRow*1+2] = (double)AC[2];
        b[0] = (double)Fa[0]; b[1] = (double)Fa[1]; b[2] = (double)Fa[2];

        //Force on particle 2:
        M[nRow*0+3] = -(double)AB[0]; M[nRow*2+3] = (double)BC[0];
        M[nRow*0+4] = -(double)AB[1]; M[nRow*2+4] = (double)BC[1];
        M[nRow*0+5] = -(double)AB[2]; M[nRow*2+5] = (double)BC[2];
        b[3] = (double)Fb[0]; b[4] = (double)Fb[1]; b[5] = (double)Fb[2];

        //Force on particle 3:
        M[nRow*1+6] = -(double)AC[0]; M[nRow*2+6] = -(double)BC[0];
        M[nRow*1+7] = -(double)AC[1]; M[nRow*2+7] = -(double)BC[1];
        M[nRow*1+8] = -(double)AC[2]; M[nRow*2+8] = -(double)BC[2];
        b[6] = (double)Fc[0]; b[7] = (double)Fc[1]; b[8] = (double)Fc[2];
       
        if ( this->h_lapack[batch_id]->SolveMinNorm(nRow, nCol, M, b) )
        {
            std::cout << "ERROR::StressGrid: LAPACK solver fails\n";
            this->m_ierr = 10;
        }

        // Sum the 3 contributions to the stress
        lab = (real_int)b[0];
        lac = (real_int)b[1];
        lbc = (real_int)b[2];

        Fij[0] = lab * AB[0]; Fij[1] = lab * AB[1]; Fij[2] = lab * AB[2];
        this->DistributePairInteraction(Ra, Rb, Fij, batch_id);
        Fij[0] = lac * AC[0]; Fij[1] = lac * AC[1]; Fij[2] = lac * AC[2];
        this->DistributePairInteraction(Ra, Rc, Fij, batch_id);
        Fij[0] = lbc * BC[0]; Fij[1] = lbc * BC[1]; Fij[2] = lbc * BC[2];
        this->DistributePairInteraction(Rb, Rc, Fij, batch_id);

    }
    else if (this->m_fdecomp == mds_gld)
    {
        Fij[0] = (Fa[0]-Fb[0])/realval_int(3.0); Fij[1] = (Fa[1]-Fb[1])/realval_int(3.0); Fij[2] = (Fa[2]-Fb[2])/realval_int(3.0);
        this->DistributePairInteraction(Ra, Rb, Fij, batch_id);
        Fij[0] = (Fa[0]-Fc[0])/realval_int(3.0); Fij[1] = (Fa[1]-Fc[1])/realval_int(3.0); Fij[2] = (Fa[2]-Fc[2])/realval_int(3.0);
        this->DistributePairInteraction(Ra, Rc, Fij, batch_id);
        Fij[0] = (Fb[0]-Fc[0])/realval_int(3.0); Fij[1] = (Fb[1]-Fc[1])/realval_int(3.0); Fij[2] = (Fb[2]-Fc[2])/realval_int(3.0);
        this->DistributePairInteraction(Rb, Rc, Fij, batch_id);
    }
    
}

// Decompose Settle
void StressGrid::DistributeSettle( array3_int Ra, array3_int Rb, array3_int Rc, array3_int Fa, array3_int Fb, array3_int Fc, int batch_id )
{
    //Counter
    int i;

    //************************************************************************************
    // UNIT vectors between particles
    array3_int AB, AC, BC;

    // Distances
    real_int normAB,normAC,normBC;

    // (Covariant) Central Force decomposition
    real_int lab, lac, lbc;

    array3_int Fij;
    //************************************************************************************

    //Dimension and number of particles
    int nDim = 3;
    int nPart = 3;

    //Number of rows and columns
    int nRow = mds_nrow3, nCol = mds_ncol3, nRHS = 1;

    // Matrix of the system (12 equations x 6 unknowns)
    double M[mds_nrow3*mds_ncol3];
    // Vector, we want to solve M*x = b
    double b[mds_nrow3], s[mds_ncol3];
	
    real_int phi_ab, phi_ac, phi_bc;
    real_int kappa_ab, kappa_ac, kappa_bc;

    if (this->m_fdecomp == mds_ccfd || this->m_fdecomp == mds_ncfd || this->m_fdecomp == mds_gld )
    {
        diffarray3(Rb, Ra, AB, this->m_box, this->m_periodic);
        diffarray3(Rc, Ra, AC, this->m_box, this->m_periodic);
        diffarray3(Rc, Rb, BC, this->m_box, this->m_periodic);

        normAB=normarray3(AB);
        normAC=normarray3(AC);
        normBC=normarray3(BC);

        for ( i = 0; i < nCol*nRow; i++ )
        {
            M[i] = 0.0;
        }

        //Force on particle 1:
        M[nRow*0+0] = (double)AB[0]; M[nRow*1+0] = (double)AC[0];
        M[nRow*0+1] = (double)AB[1]; M[nRow*1+1] = (double)AC[1];
        M[nRow*0+2] = (double)AB[2]; M[nRow*1+2] = (double)AC[2];
        b[0] = (double)Fa[0]; b[1] = (double)Fa[1]; b[2] = (double)Fa[2];

        //Force on particle 2:
        M[nRow*0+3] = -(double)AB[0]; M[nRow*2+3] = (double)BC[0];
        M[nRow*0+4] = -(double)AB[1]; M[nRow*2+4] = (double)BC[1];
        M[nRow*0+5] = -(double)AB[2]; M[nRow*2+5] = (double)BC[2];
        b[3] = (double)Fb[0]; b[4] = (double)Fb[1]; b[5] = (double)Fb[2];

        //Force on particle 3:
        M[nRow*1+6] = -(double)AC[0]; M[nRow*2+6] = -(double)BC[0];
        M[nRow*1+7] = -(double)AC[1]; M[nRow*2+7] = -(double)BC[1];
        M[nRow*1+8] = -(double)AC[2]; M[nRow*2+8] = -(double)BC[2];
        b[6] = (double)Fc[0]; b[7] = (double)Fc[1]; b[8] = (double)Fc[2];

        if ( this->h_lapack[batch_id]->SolveMinNorm(nRow, nCol, M, b) )
        {
            std::cout << "ERROR::StressGrid: LAPACK solver fails\n";
            this->m_ierr = 10;
        }

        // Sum the 3 contributions to the stress
        lab = (real_int)b[0];
        lac = (real_int)b[1];
        lbc = (real_int)b[2];		
        Fij[0] = lab * AB[0]; Fij[1] = lab * AB[1]; Fij[2] = lab * AB[2];
        this->DistributePairInteraction( Ra, Rb, Fij, batch_id);
        Fij[0] = lac * AC[0]; Fij[1] = lac * AC[1]; Fij[2] = lac * AC[2];
        this->DistributePairInteraction( Ra, Rc, Fij, batch_id );
        Fij[0] = lbc * BC[0]; Fij[1] = lbc * BC[1]; Fij[2] = lbc * BC[2];
        this->DistributePairInteraction( Rb, Rc, Fij, batch_id );
		
        // Calculate scalar force and bond stiffness
        phi_ab = lab*normAB;
        phi_ac = lac*normAC;
        phi_bc = lbc*normBC;
        
        kappa_ab = realval_int(0.0); //lab
        kappa_ac = realval_int(0.0); //lac
        kappa_bc = realval_int(0.0); //lbc
        
        //Calculate Elasticity
        //DistributeElasticity_internal(darray xi, darray xj, darray xk, darray xl, double phi, double kappa)
        if (this->m_griddim == mds_griddim_xyz)
        {
            this->DistributeElasticity_internal3D(Ra, Rb, Ra, Rb, phi_ab, kappa_ab);
            this->DistributeElasticity_internal3D(Ra, Rc, Ra, Rc, phi_ac, kappa_ac);
            this->DistributeElasticity_internal3D(Rb, Rc, Rb, Rc, phi_bc, kappa_bc);
        }
        else
        {
            this->DistributeElasticity_internal1D(Ra, Rb, Ra, Rb, phi_ab, kappa_ab);
            this->DistributeElasticity_internal1D(Ra, Rc, Ra, Rc, phi_ac, kappa_ac);
            this->DistributeElasticity_internal1D(Rb, Rc, Rb, Rc, phi_bc, kappa_bc);
        }
    }
}

// Decompose 4-body potentials (dihedrals)
void StressGrid::DistributeN4( array3_int Ra, array3_int Rb, array3_int Rc, array3_int Rd, array3_int Fa, array3_int Fb, array3_int Fc, array3_int Fd, int batch_id )
{
    //Counter
    int i;

    //************************************************************************************
    // UNIT vectors between particles
    array3_int AB, AC, AD, BC, BD, CD;

    // Distances
    real_int normAB,normAC,normAD,normBC,normBD,normCD;

    // (Covariant) Central Force decomposition
    real_int lab, lac, lad, lbc, lbd, lcd;

    array3_int Fij;
    //************************************************************************************

    //Dimension and number of particles
    int nDim = 3;
    int nPart = 4;

    //Number of rows and columns
    int nRow = mds_nrow4, nCol = mds_ncol4, nRHS = 1;

    // Matrix of the system (12 equations x 6 unknowns)
    double M[mds_nrow4*mds_ncol4];
    // Vector, we want to solve M*x = b
    double b[mds_nrow4], s[mds_ncol4];

    // If the force decomposition is cCFD or CFD
    if(this->m_fdecomp == mds_ccfd || this->m_fdecomp == mds_ncfd)
    {
        diffarray3(Rb, Ra, AB, this->m_box, this->m_periodic);
        diffarray3(Rc, Ra, AC, this->m_box, this->m_periodic);
        diffarray3(Rd, Ra, AD, this->m_box, this->m_periodic);
        diffarray3(Rc, Rb, BC, this->m_box, this->m_periodic);
        diffarray3(Rd, Rb, BD, this->m_box, this->m_periodic);
        diffarray3(Rd, Rc, CD, this->m_box, this->m_periodic);

        normAB=normarray3(AB);
        normAC=normarray3(AC);
        normAD=normarray3(AD);
        normBC=normarray3(BC);
        normBD=normarray3(BD);
        normCD=normarray3(CD);

        for ( i = 0; i < nCol*nRow; i++ )
        {
            M[i] = 0.0;
        }
        //Force on particle 1:
        M[nRow*0+0] = (double)AB[0]; M[nRow*1+0] = (double)AC[0]; M[nRow*2+0] = (double)AD[0];
        M[nRow*0+1] = (double)AB[1]; M[nRow*1+1] = (double)AC[1]; M[nRow*2+1] = (double)AD[1];
        M[nRow*0+2] = (double)AB[2]; M[nRow*1+2] = (double)AC[2]; M[nRow*2+2] = (double)AD[2];
        b[0] = (double)Fa[0];
        b[1] = (double)Fa[1];
        b[2] = (double)Fa[2];

        //Force on particle 2:
        M[nRow*0+3] = -(double)AB[0]; M[nRow*3+3] = (double)BC[0]; M[nRow*4+3] = (double)BD[0];
        M[nRow*0+4] = -(double)AB[1]; M[nRow*3+4] = (double)BC[1]; M[nRow*4+4] = (double)BD[1];
        M[nRow*0+5] = -(double)AB[2]; M[nRow*3+5] = (double)BC[2]; M[nRow*4+5] = (double)BD[2];
        b[3] = (double)Fb[0];
        b[4] = (double)Fb[1];
        b[5] = (double)Fb[2];

        //Force on particle 3:
        M[nRow*1+6] = -(double)AC[0]; M[nRow*3+6] = -(double)BC[0]; M[nRow*5+6] = (double)CD[0];
        M[nRow*1+7] = -(double)AC[1]; M[nRow*3+7] = -(double)BC[1]; M[nRow*5+7] = (double)CD[1];
        M[nRow*1+8] = -(double)AC[2]; M[nRow*3+8] = -(double)BC[2]; M[nRow*5+8] = (double)CD[2];
        b[6] = (double)Fc[0];
        b[7] = (double)Fc[1];
        b[8] = (double)Fc[2];

        //Force on particle 4:
        M[nRow*2+9]  = -(double)AD[0]; M[nRow*4+9] =  -(double)BD[0]; M[nRow*5+9] =  -(double)CD[0];
        M[nRow*2+10] = -(double)AD[1]; M[nRow*4+10] = -(double)BD[1]; M[nRow*5+10] = -(double)CD[1];
        M[nRow*2+11] = -(double)AD[2]; M[nRow*4+11] = -(double)BD[2]; M[nRow*5+11] = -(double)CD[2];
        b[9]  = (double)Fd[0];
        b[10] = (double)Fd[1];
        b[11] = (double)Fd[2];
        
        if ( this->h_lapack[batch_id]->SolveMinNorm(nRow, nCol, M, b) )
        {
            std::cout << "ERROR::StressGrid: LAPACK solver fails\n";
            this->m_ierr = 10;
        }

        // Sum the 6 contributions to the stress
        
        lab = (real_int)b[0];
        lac = (real_int)b[1];
        lad = (real_int)b[2];
        lbc = (real_int)b[3];
        lbd = (real_int)b[4];
        lcd = (real_int)b[5];

        Fij[0] = lab * AB[0]; Fij[1] = lab * AB[1]; Fij[2] = lab * AB[2];
        this->DistributePairInteraction(Ra, Rb, Fij, batch_id);
        Fij[0] = lac * AC[0]; Fij[1] = lac * AC[1]; Fij[2] = lac * AC[2];
        this->DistributePairInteraction(Ra, Rc, Fij, batch_id);
        Fij[0] = lad * AD[0]; Fij[1] = lad * AD[1]; Fij[2] = lad * AD[2];
        this->DistributePairInteraction(Ra, Rd, Fij, batch_id);
        Fij[0] = lbc * BC[0]; Fij[1] = lbc * BC[1]; Fij[2] = lbc * BC[2];
        this->DistributePairInteraction(Rb, Rc, Fij, batch_id);
        Fij[0] = lbd * BD[0]; Fij[1] = lbd * BD[1]; Fij[2] = lbd * BD[2];
        this->DistributePairInteraction(Rb, Rd, Fij, batch_id);
        Fij[0] = lcd * CD[0]; Fij[1] = lcd * CD[1]; Fij[2] = lcd * CD[2];
        this->DistributePairInteraction(Rc, Rd, Fij, batch_id);

    }
    else if (this->m_fdecomp == mds_gld)
    {
        Fij[0] = (Fa[0]-Fb[0])/realval_int(4.0); Fij[1] = (Fa[1]-Fb[1])/realval_int(4.0); Fij[2] = (Fa[2]-Fb[2])/realval_int(4.0);
        this->DistributePairInteraction(Ra, Rb, Fij, batch_id);
        Fij[0] = (Fa[0]-Fc[0])/realval_int(4.0); Fij[1] = (Fa[1]-Fc[1])/realval_int(4.0); Fij[2] = (Fa[2]-Fc[2])/realval_int(4.0);
        this->DistributePairInteraction(Ra, Rc, Fij, batch_id);
        Fij[0] = (Fa[0]-Fd[0])/realval_int(4.0); Fij[1] = (Fa[1]-Fd[1])/realval_int(4.0); Fij[2] = (Fa[2]-Fd[2])/realval_int(4.0);
        this->DistributePairInteraction(Ra, Rd, Fij, batch_id);
        Fij[0] = (Fb[0]-Fc[0])/realval_int(4.0); Fij[1] = (Fb[1]-Fc[1])/realval_int(4.0); Fij[2] = (Fb[2]-Fc[2])/realval_int(4.0);
        this->DistributePairInteraction(Rb, Rc, Fij, batch_id);
        Fij[0] = (Fb[0]-Fd[0])/realval_int(4.0); Fij[1] = (Fb[1]-Fd[1])/realval_int(4.0); Fij[2] = (Fb[2]-Fd[2])/realval_int(4.0);
        this->DistributePairInteraction(Rb, Rd, Fij, batch_id);
        Fij[0] = (Fc[0]-Fd[0])/realval_int(4.0); Fij[1] = (Fc[1]-Fd[1])/realval_int(4.0); Fij[2] = (Fc[2]-Fd[2])/realval_int(4.0);
        this->DistributePairInteraction(Rc, Rd, Fij, batch_id);
    }
    
}


// Decompose 5-body potentials (CMAP)
void StressGrid::DistributeN5(array3_int Ra, array3_int Rb, array3_int Rc, array3_int Rd, array3_int Re, array3_int Fa, array3_int Fb, array3_int Fc, array3_int Fd, array3_int Fe, int batch_id)
{
    //Counter
    int i;

    //************************************************************************************
    // UNIT vectors between particles
    array3_int AB, AC, AD, AE, BC, BD, BE, CD, CE, DE;

    // Distances
    real_int normAB,normAC,normAD,normAE,normBC,normBD,normBE,normCD,normCE,normDE;

    // (Covariant) Central Force decomposition
    real_int lab, lac, lad, lae, lbc, lbd, lbe, lcd, lce, lde;

    array3_int Fij;
    //************************************************************************************

    //Dimension and number of particles
    int nDim = 3;
    int nPart = 5;

    //Number of rows and columns
    int nRow = mds_nrow5, nCol = mds_ncol5, nRHS = 1;

    // Matrix of the system (15 equations x 10 unknowns)
    double M[mds_nrow5*mds_ncol5];
    // Vector, we want to solve M*x = b
    double b[mds_nrow5], s[mds_ncol5];
    // Scalar product of the Normal and the initial CFD
    real_int prod, CaleyMengerNormal[mds_ncol5];

    // If the force decomposition is cCFD or CFD
    if(this->m_fdecomp == mds_ccfd || this->m_fdecomp == mds_ncfd)
    {
        diffarray3(Rb, Ra, AB, this->m_box, this->m_periodic);
        diffarray3(Rc, Ra, AC, this->m_box, this->m_periodic);
        diffarray3(Rd, Ra, AD, this->m_box, this->m_periodic);
        diffarray3(Re, Ra, AE, this->m_box, this->m_periodic);
        diffarray3(Rc, Rb, BC, this->m_box, this->m_periodic);
        diffarray3(Rd, Rb, BD, this->m_box, this->m_periodic);
        diffarray3(Re, Rb, BE, this->m_box, this->m_periodic);
        diffarray3(Rd, Rc, CD, this->m_box, this->m_periodic);
        diffarray3(Re, Rc, CE, this->m_box, this->m_periodic);
        diffarray3(Re, Rd, DE, this->m_box, this->m_periodic);

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

        for(i = 0; i < 3; i++)
        {
            if(normAB > mds_eps)
                AB[i]/=normAB;
            if(normAC > mds_eps)
                AC[i]/=normAC;
            if(normAD > mds_eps)
                AD[i]/=normAD;
            if(normAE > mds_eps)
                AE[i]/=normAE;
            if(normBC > mds_eps)
                BC[i]/=normBC;
            if(normBD > mds_eps)
                BD[i]/=normBD;
            if(normBE > mds_eps)
                BE[i]/=normBE;
            if(normCD > mds_eps)
                CD[i]/=normCD;
            if(normCE > mds_eps)
                CE[i]/=normCE;
            if(normDE > mds_eps)
                DE[i]/=normDE;
        }

        for ( i = 0; i < nCol*nRow; i++ )
        {
            M[i] = 0.0;
        }

        //Force on particle 1:
        M[nRow*0+0] = (double)AB[0]; M[nRow*1+0] = (double)AC[0]; M[nRow*2+0] = (double)AD[0]; M[nRow*3+0] = (double)AE[0];
        M[nRow*0+1] = (double)AB[1]; M[nRow*1+1] = (double)AC[1]; M[nRow*2+1] = (double)AD[1]; M[nRow*3+1] = (double)AE[1];
        M[nRow*0+2] = (double)AB[2]; M[nRow*1+2] = (double)AC[2]; M[nRow*2+2] = (double)AD[2]; M[nRow*3+2] = (double)AE[2];
        b[0] = (double)Fa[0];
        b[1] = (double)Fa[1];
        b[2] = (double)Fa[2];

        //Force on particle 2:
        M[nRow*0+3] = -(double)AB[0]; M[nRow*4+3] = (double)BC[0]; M[nRow*5+3] = (double)BD[0]; M[nRow*6+3] = (double)BE[0];
        M[nRow*0+4] = -(double)AB[1]; M[nRow*4+4] = (double)BC[1]; M[nRow*5+4] = (double)BD[1]; M[nRow*6+4] = (double)BE[1];
        M[nRow*0+5] = -(double)AB[2]; M[nRow*4+5] = (double)BC[2]; M[nRow*5+5] = (double)BD[2]; M[nRow*6+5] = (double)BE[2];
        b[3] = (double)Fb[0];
        b[4] = (double)Fb[1];
        b[5] = (double)Fb[2];

        //Force on particle 3:
        M[nRow*1+6] = -(double)AC[0]; M[nRow*4+6] = -(double)BC[0]; M[nRow*7+6] = (double)CD[0]; M[nRow*8+6] = (double)CE[0];
        M[nRow*1+7] = -(double)AC[1]; M[nRow*4+7] = -(double)BC[1]; M[nRow*7+7] = (double)CD[1]; M[nRow*8+7] = (double)CE[1];
        M[nRow*1+8] = -(double)AC[2]; M[nRow*4+8] = -(double)BC[2]; M[nRow*7+8] = (double)CD[2]; M[nRow*8+8] = (double)CE[2];
        b[6] = (double)Fc[0];
        b[7] = (double)Fc[1];
        b[8] = (double)Fc[2];

        //Force on particle 4:
        M[nRow*2+9]  = -(double)AD[0]; M[nRow*5+9]  = -(double)BD[0]; M[nRow*7+9]  = -(double)CD[0]; M[nRow*9+9]  = (double)DE[0];
        M[nRow*2+10] = -(double)AD[1]; M[nRow*5+10] = -(double)BD[1]; M[nRow*7+10] = -(double)CD[1]; M[nRow*9+10] = (double)DE[1];
        M[nRow*2+11] = -(double)AD[2]; M[nRow*5+11] = -(double)BD[2]; M[nRow*7+11] = -(double)CD[2]; M[nRow*9+11] = (double)DE[2];
        b[9] = (double)Fd[0];
        b[10] = (double)Fd[1];
        b[11] = (double)Fd[2];

        //Force on particle 5:
        M[nRow*3+12] = -(double)AE[0]; M[nRow*6+12] = -(double)BE[0]; M[nRow*8+12] = -(double)CE[0]; M[nRow*9+12] = -(double)DE[0];
        M[nRow*3+13] = -(double)AE[1]; M[nRow*6+13] = -(double)BE[1]; M[nRow*8+13] = -(double)CE[1]; M[nRow*9+13] = -(double)DE[1];
        M[nRow*3+14] = -(double)AE[2]; M[nRow*6+14] = -(double)BE[2]; M[nRow*8+14] = -(double)CE[2]; M[nRow*9+14] = -(double)DE[2];
        b[12] = (double)Fe[0];
        b[13] = (double)Fe[1];
        b[14] = (double)Fe[2];

        if ( this->h_lapack[batch_id]->SolveMinNorm(nRow, nCol, M, b) )
        {
            std::cout << "ERROR::StressGrid: LAPACK solver fails\n";
            this->m_ierr = 10;
        }

        //If cCFD project the least squares CFD to the shape space
        if(this->m_fdecomp == mds_ccfd)
        {
            //Calculate the normal to the Shape Space
            ShapeSpace5Normal(normAB,normAC,normAD,normAE,normBC,normBD,normBE,normCD,normCE,normDE,CaleyMengerNormal);
            
            //Covariant derivative:
            prod = realval_int(0.0);
            for ( i = 0; i < nCol; i++ )
            {
                prod +=(real_int)b[i]*CaleyMengerNormal[i];
            }

            for ( i = 0; i < nCol; i++ )
            {
                b[i] = (real_int)b[i] - prod * CaleyMengerNormal[i];
            }
        }

        // Sum the 10 contributions to the stress
        lab = (real_int)b[0];
        lac = (real_int)b[1];
        lad = (real_int)b[2];
        lae = (real_int)b[3];
        lbc = (real_int)b[4];
        lbd = (real_int)b[5];
        lbe = (real_int)b[6];
        lcd = (real_int)b[7];
        lce = (real_int)b[8];
        lde = (real_int)b[9];

        Fij[0] = lab * AB[0]; Fij[1] = lab * AB[1]; Fij[2] = lab * AB[2];
        this->DistributePairInteraction(Ra, Rb, Fij, batch_id);
        Fij[0] = lac * AC[0]; Fij[1] = lac * AC[1]; Fij[2] = lac * AC[2];
        this->DistributePairInteraction(Ra, Rc, Fij, batch_id);
        Fij[0] = lad * AD[0]; Fij[1] = lad * AD[1]; Fij[2] = lad * AD[2];
        this->DistributePairInteraction(Ra, Rd, Fij, batch_id);
        Fij[0] = lae * AE[0]; Fij[1] = lae * AE[1]; Fij[2] = lae * AE[2];
        this->DistributePairInteraction(Ra, Re, Fij, batch_id);
        Fij[0] = lbc * BC[0]; Fij[1] = lbc * BC[1]; Fij[2] = lbc * BC[2];
        this->DistributePairInteraction(Rb, Rc, Fij, batch_id);
        Fij[0] = lbd * BD[0]; Fij[1] = lbd * BD[1]; Fij[2] = lbd * BD[2];
        this->DistributePairInteraction(Rb, Rd, Fij, batch_id);
        Fij[0] = lbe * BE[0]; Fij[1] = lbe * BE[1]; Fij[2] = lbe * BE[2];
        this->DistributePairInteraction(Rb, Re, Fij, batch_id);
        Fij[0] = lcd * CD[0]; Fij[1] = lcd * CD[1]; Fij[2] = lcd * CD[2];
        this->DistributePairInteraction(Rc, Rd, Fij, batch_id);
        Fij[0] = lce * CE[0]; Fij[1] = lce * CE[1]; Fij[2] = lce * CE[2];
        this->DistributePairInteraction(Rc, Re, Fij, batch_id);
        Fij[0] = lde * DE[0]; Fij[1] = lde * DE[1]; Fij[2] = lde * DE[2];
        this->DistributePairInteraction(Rd, Re, Fij, batch_id);

    }
    else if(this->m_fdecomp == mds_gld)
    {
        Fij[0] = (Fa[0]-Fb[0])/realval_int(5.0); Fij[1] = (Fa[1]-Fb[1])/realval_int(5.0); Fij[2] = (Fa[2]-Fb[2])/realval_int(5.0);
        this->DistributePairInteraction(Ra, Rb, Fij, batch_id);
        Fij[0] = (Fa[0]-Fc[0])/realval_int(5.0); Fij[1] = (Fa[1]-Fc[1])/realval_int(5.0); Fij[2] = (Fa[2]-Fc[2])/realval_int(5.0);
        this->DistributePairInteraction(Ra, Rc, Fij, batch_id);
        Fij[0] = (Fa[0]-Fd[0])/realval_int(5.0); Fij[1] = (Fa[1]-Fd[1])/realval_int(5.0); Fij[2] = (Fa[2]-Fd[2])/realval_int(5.0);
        this->DistributePairInteraction(Ra, Rd, Fij, batch_id);
        Fij[0] = (Fa[0]-Fe[0])/realval_int(5.0); Fij[1] = (Fa[1]-Fe[1])/realval_int(5.0); Fij[2] = (Fa[2]-Fe[2])/realval_int(5.0);
        this->DistributePairInteraction(Ra, Re, Fij, batch_id);
        Fij[0] = (Fb[0]-Fc[0])/realval_int(5.0); Fij[1] = (Fb[1]-Fc[1])/realval_int(5.0); Fij[2] = (Fb[2]-Fc[2])/realval_int(5.0);
        this->DistributePairInteraction(Rb, Rc, Fij, batch_id);
        Fij[0] = (Fb[0]-Fd[0])/realval_int(5.0); Fij[1] = (Fb[1]-Fd[1])/realval_int(5.0); Fij[2] = (Fb[2]-Fd[2])/realval_int(5.0);
        this->DistributePairInteraction(Rb, Rd, Fij, batch_id);
        Fij[0] = (Fb[0]-Fe[0])/realval_int(5.0); Fij[1] = (Fb[1]-Fe[1])/realval_int(5.0); Fij[2] = (Fb[2]-Fe[2])/realval_int(5.0);
        this->DistributePairInteraction(Rb, Re, Fij, batch_id);
        Fij[0] = (Fc[0]-Fd[0])/realval_int(5.0); Fij[1] = (Fc[1]-Fd[1])/realval_int(5.0); Fij[2] = (Fc[2]-Fd[2])/realval_int(5.0);
        this->DistributePairInteraction(Rc, Rd, Fij, batch_id);
        Fij[0] = (Fc[0]-Fe[0])/realval_int(5.0); Fij[1] = (Fc[1]-Fe[1])/realval_int(5.0); Fij[2] = (Fc[2]-Fe[2])/realval_int(5.0);
        this->DistributePairInteraction(Rc, Re, Fij, batch_id);
        Fij[0] = (Fd[0]-Fe[0])/realval_int(5.0); Fij[1] = (Fd[1]-Fe[1])/realval_int(5.0); Fij[2] = (Fd[2]-Fe[2])/realval_int(5.0);
        this->DistributePairInteraction(Rd, Re, Fij, batch_id);
    }
}

// General function to decompose N-body potentials (it can be used to compute higher order terms coming from EAM for instance)
void StressGrid::DistributeNBody ( int nPart, array3_ext *R, array3_ext *F, bool distribute_stress, int batch_id)
{
    // this forces NBody to be a single threaded call
    std::lock_guard<std::mutex> lock(this->m_mutex_state);

    int i,j,k, iD, jD, n;
    array3_int F_ij_temp, Ri_temp, Rj_temp;

    // grow the temp state as needed (only allocating for a single thread here!)
    if (nPart > this->m_maxpart)
    {
        printf("batch_id(%i) is deleting p_bvec\n", batch_id);
        if (this->p_Amat  != nullptr) delete [] this->p_Amat;
        if (this->p_AmatT != nullptr) delete [] this->p_AmatT;
        if (this->p_bvec  != nullptr) delete [] this->p_bvec;
        if (this->p_Rij   != nullptr) delete [] this->p_Rij;
        if (this->p_Uij   != nullptr) delete [] this->p_Uij;
        if (this->p_Fij   != nullptr) delete [] this->p_Fij;
    
        int maxrows = mds_ndim*nPart;
        int maxcols = (nPart*(nPart-1))/2;

        this->p_Amat  = new double [maxrows*maxcols];
        this->p_AmatT = new double [maxrows*maxcols];
        this->p_bvec  = new double [maxrows];
        this->p_Rij  = new array3_int [nPart*nPart];
        this->p_Fij  = new array3_int [nPart*nPart];
        this->p_Uij  = new array3_int [maxcols];
        
        int maxrows_prev = mds_ndim*m_maxpart;
        int maxcols_prev = (m_maxpart*(m_maxpart-1))/2;
        printf("batch_id(%i) grew p_bvec from %i to %i \n", batch_id, maxcols_prev, maxcols);
        
        this->m_maxpart = nPart;
    }

    // zero the pairwise force and pairwise position arrays
    for ( i = 0; i < nPart*nPart; ++i)
    {
        this->p_Rij[i][0] = this->p_Rij[i][1] = this->p_Rij[i][2] = realval_int(0.0);
        this->p_Fij[i][0] = this->p_Fij[i][1] = this->p_Fij[i][2] = realval_int(0.0);
    }

    n = 0;
    for ( i = 0; i < nPart; i++ )
    {
        for ( j = i+1; j < nPart; j++ )
        {
            diffarray3(R[j], R[i], this->p_Uij[n], this->m_box, this->m_periodic);
            copyarray3(this->p_Uij[n], this->p_Rij[i*nPart+j]);
            scalearray3(this->p_Uij[n], realval_int(-1.0), this->p_Rij[j*nPart+i]);
            scalearray3(this->p_Uij[n], realval_int(1.0)/normarray3(this->p_Uij[n]),this->p_Uij[n]);
            n++;
        }
    }

    // If the force decomposition is cCFD or CFD
    if(this->m_fdecomp == mds_ccfd || this->m_fdecomp == mds_ncfd)
    {
        //Number of rows and columns
        int nRow;
        int nCol;

        nRow = mds_ndim * nPart;
        nCol = (nPart * (nPart - 1)) / 2;

        for ( i = 0; i < nCol*nRow; i++ )
        {
            this->p_Amat [i] = 0.0;
            this->p_AmatT[i] = 0.0;
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
                    this->p_Amat [nRow*n+(iD+k)] =  this->p_Uij[n][k];
                    this->p_Amat [nRow*n+(jD+k)] = -this->p_Uij[n][k];
                    this->p_AmatT[(iD+k)*nCol+n] =  this->p_Uij[n][k];
                    this->p_AmatT[(jD+k)*nCol+n] = -this->p_Uij[n][k];
                }
                n++;
            }

            copyarray3(F[i], &this->p_bvec[iD]);
        }

        if ( this->h_lapack[batch_id]->SolveMinNorm(nRow, nCol, this->p_Amat, this->p_bvec))
        {
            std::cout << "ERROR::StressGrid: LAPACK solver fails\n";
            this->m_ierr = 10;
        }

        if(this->m_fdecomp == mds_ccfd)
            this->h_lapack[batch_id]->QQTb( nCol, nRow, nCol, nRow-6, this->p_AmatT, this->p_bvec );

        n = 0;
        for ( i = 0; i < nPart; i++ )
        {
            for ( j = i+1; j < nPart; j++ )
            {
                scalearray3(this->p_Uij[n], (real_int)this->p_bvec[n], F_ij_temp);
                copyarray3(F_ij_temp, this->p_Fij[i*nPart+j]);
                scalearray3(F_ij_temp, realval_int(-1.0), this->p_Fij[j*nPart+i]);

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
    else if(this->m_fdecomp == mds_gld)
    {
        n = 0;
        for ( i = 0; i < nPart; i++ )
        {
            for ( j = i+1; j < nPart; j++ )
            {
                diffarray3(F[i], F[j], F_ij_temp );
                scalearray3(F_ij_temp, realval_int(1.0)/static_cast<real_int>(nPart), F_ij_temp);
                copyarray3(F_ij_temp, this->p_Fij[i*nPart+j]);
                scalearray3(F_ij_temp, realval_int(-1.0), this->p_Fij[j*nPart+i]);

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

//Auxiliary Methods to Calculate the Phi and Kappa Terms for the Born Term
// For cosine derivative functions must be called before the theta derivatives. 
// derivative identifier dertype ab = 0, bg = 1, ag = 3

// ab|bg|ag = 0 | 1 | 2

// second derivative identifier derivative dertype2

//indices are flipped be careful!

/*
 * abab | bgab | agab	= 00 | 01 | 02
 * abbg | bgbg | agbg	= 10 | 11 | 12
 * abag | bgag | agag	= 20 | 21 | 22
 */

 //Cosine and Theta Calculations (may only be needed sometimes)
real_int StressGrid::CalcCosine(real_int ab, real_int bg, real_int ag) {
	real_int costheta, numer, denom;

	numer = (ab * ab) + (bg * bg) - (ag * ag);
	denom = realval_int(2.0) * ab * bg;
	costheta = numer / denom;
	return costheta;
}

real_int StressGrid::CalcTheta(real_int costheta) {
	return acos(costheta);
}

//Derivative of Cosine Function (creates derivative vector)
void StressGrid::ThreeBodyCosineD(real_int ab, real_int bg, real_int ag, array3_int &d_cos_array) {
	real_int numer, denom;


	//dab of costheta
	numer = (ab * ab) - (bg * bg) + (ag * ag);
	denom = realval_int(2.0) * ab * ab * bg;

	d_cos_array[iab] = numer / denom;

	//dbg of costheta
	numer = -(ab * ab) + (bg * bg) + (ag * ag);
	denom = realval_int(2.0) * ab * bg * bg;

	d_cos_array[ibg] = numer / denom;

	//dag of costheta
	numer = -ag;
	denom = ab * bg;

	d_cos_array[iag] = numer / denom;
}

//Second Derivative of Cosine Function (Creates derivative matrix)
void StressGrid::ThreeBodyCosineD2(real_int ab, real_int bg, real_int ag, matrix3_int &d2_cos_array) {
	real_int numer;
	real_int denom;

	//d00 of costheta
	numer = (bg * bg) - (ag * ag);
	denom = (ab * ab * ab * bg);

	d2_cos_array[iab][iab] = numer / denom;

	// d01 and d10 of costheta

	numer = -((ab * ab) + (bg * bg) + (ag * ag));
	denom = realval_int(2.0) * (ab * ab) * (bg * bg);

	d2_cos_array[iab][ibg] = numer / denom;
	d2_cos_array[ibg][iab] = numer / denom;

	// d02 and d20 of costheta

	numer = ag;
	denom = ab * ab * bg;

	d2_cos_array[iab][iag] = numer / denom;
	d2_cos_array[iag][iab] = numer / denom;

	// d11 of costheta
	numer = (ab * ab) - (ag * ag);
	denom = (ab * bg * bg * bg);

	d2_cos_array[ibg][ibg] = numer / denom;

	// d12 and d21 of costheta
	numer = ag;
	denom = ab * bg * bg;

	d2_cos_array[iag][ibg] = numer / denom;
	d2_cos_array[ibg][iag] = numer / denom;

	// d22 of costheta
	numer = realval_int(-1.0);
	denom = ab * bg;

	d2_cos_array[iag][iag] = numer / denom;
}

//First Derivative of Theta Function (Creates 1st derivative vector)
//Need Cosine Theta
//Need Derivative of Cosine Vector
void StressGrid::ThreeBodyThetaD(real_int costheta, array3_int &d_cos_array, array3_int &d_theta_array) {
	real_int scalefactor;

	scalefactor = realval_int(-1.0) / sqrt(realval_int(1.0) - (costheta * costheta));

	for (int i = 0; i < 3; i++) {
		d_theta_array[i] = scalefactor * d_cos_array[i];
	}
}

//Second Derivative of Theta Function (Creates 2nd Derivative Matix)
//Need Cosine Theta
//Need Derivative of Cosine Vector
//Need 2nd Derivative of Cosine Matrix
void StressGrid::ThreeBodyThetaD2(real_int costheta, array3_int d_cos_array, matrix3_int &d2_cos_array, matrix3_int &d2_theta_array) {
	real_int scalefactor;
	real_int sinthetasq;

	sinthetasq = realval_int(1.0) - (costheta * costheta);

	scalefactor = realval_int(1.0) / (sinthetasq * sqrt(sinthetasq));

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			d2_theta_array[i][j] = scalefactor * ((-sinthetasq) * d2_cos_array[i][j] - costheta * d_cos_array[i] * d_cos_array[j]);
		}
	}
}

//List of 2-body Potential Auxilery Functions
//Calculate The Phi and Kappa for Harmonic Potential -> Implimented
void StressGrid::HarmonicPhiKappa(real_ext deltaR, real_ext k, real_ext &phi, real_ext &kappa)
{
    if (true == this->m_disable)
        return;

    phi = k * deltaR;
    kappa = k;
}

//Calculate the Phi and Kappa for Buckingham Potential -> Not Implimented, Nonbonded?
void StressGrid::BuckinghamPhiKappa(real_ext r, real_ext a, real_ext b, real_ext c, real_ext &phi, real_ext &kappa)
{
    if (true == this->m_disable)
        return;

    double exponent = -b * r;
    double rinv = 1 / r;
    double rinvsq = rinv * rinv;
    double rinvsix = rinvsq * rinvsq * rinvsq;

    phi = -a * b * exp(exponent) + (6 * c) * rinvsix * rinv;
    kappa = a * b * b * exp(exponent) - (42 * c) * rinvsix * rinvsq;
}

//Calculate the Phi and Kappa for the Fourth Power Potential -> Implimented g96bond
void StressGrid::FourthPowerPhiKappa(real_ext k4, real_ext dist, real_ext dist0, real_ext &phi, real_ext &kappa)
{
    if (true == this->m_disable)
        return;

    phi = k4 * dist * (dist * dist - dist0 * dist0);
    kappa = k4 * (3 * dist * dist - dist0 * dist0);
}

//Calculate the Phi and  Kappa for the Morse Potential -> Implimented
void StressGrid::MorsePhiKappa(real_ext expadeltaR, real_ext a, real_ext d, real_ext &phi, real_ext &kappa)
{
    if (true == this->m_disable)
        return;

    //expadeltaR = e^(-a*(r-r0))
    double coeffexp = 2 * a * d * expadeltaR;

    phi = coeffexp * (1 - expadeltaR);
    kappa = coeffexp * a * (2 * expadeltaR - 1);
}

//Calculate the Phi and Kappa for the Cubic Bond Potential -> Implimented
void StressGrid::CubicBondPhiKappa(real_ext deltaR, real_ext k, real_ext kcubic, real_ext &phi, real_ext &kappa)
{
    if (true == this->m_disable)
        return;

    phi = 2 * k * deltaR + 3 * k * kcubic * deltaR * deltaR;
    kappa = 2 * k + 6 * k * kcubic * deltaR;
}

//Calculate the Phi and Kappa for the FENE Potential -> Implimented
void StressGrid::FENEPhiKappa(real_ext r, real_ext k, real_ext diffratio, real_ext& phi, real_ext& kappa)
{
    if (true == this->m_disable)
        return;
    //diffratio = 1 - (r/r0)^2

    phi = k * r / diffratio;
    kappa = k * (2 - diffratio) / diffratio;
}

//List of 3-body Potential Auxiliary Functions
//Derivative Vector/Matrix form
// derivative identifier ab = 0, bg = 1, ag = 3

// ab|bg|ag = 0 | 1 | 2
/*
 * abab | bgab | agab	= 00 | 01 | 02
 * abbg | bgbg | agbg	= 10 | 11 | 12
 * abag | bgag | agag	= 20 | 21 | 22
 */

//Calculate the Phi and Kappa for the Harmonic Angle Potential -> Implimented
void StressGrid::HarmonicAnglePhiKappa(real_ext ab_ext, real_ext bg_ext, real_ext ag_ext, real_ext costheta_ext, real_ext deltatheta_ext, real_ext k_ext, array3_ext &phi, matrix3_ext &kappa) 
{
    if (true == this->m_disable)
        return;

    // convert to internal precision
    real_int ab = (real_int)ab_ext;
    real_int bg = (real_int)bg_ext;
    real_int ag = (real_int)ag_ext;
    real_int costheta = (real_int)costheta_ext;
    real_int deltatheta = (real_int)deltatheta_ext;
    real_int k = (real_int)k_ext;

    //double costh = this->CalcCosine(ab, bg, ag);
    array3_int d_cos_array;
    zeroarray3(d_cos_array);
    matrix3_int d2_cos_array;
    zeromatrix3(d2_cos_array);
    array3_int d_theta_array;
    zeroarray3(d_theta_array);
    matrix3_int d2_theta_array;
    zeromatrix3(d2_theta_array);
    this->ThreeBodyCosineD(ab, bg, ag, d_cos_array);
    this->ThreeBodyCosineD2(ab, bg, ag, d2_cos_array);
    this->ThreeBodyThetaD(costheta, d_cos_array, d_theta_array);
    this->ThreeBodyThetaD2(costheta, d_cos_array, d2_cos_array, d2_theta_array);

    //Calculate Phi Vector
    for (int i = 0; i < 3; i++) {
        phi[i] = (real_ext)(k * deltatheta * d_theta_array[i]);
    }

    //Calculate Kappa Matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            kappa[i][j] = (real_ext)(k * (d_theta_array[i] * d_theta_array[j] + deltatheta * d2_theta_array[i][j]));
        }
    }
}

//Calculate the Phi and Kappa for the Harmonic Cosine Potential -> Implimented Gromos96
void StressGrid::HarmonicCosPhiKappa(real_ext ab_ext, real_ext bg_ext, real_ext ag_ext, real_ext deltacos_ext, real_ext k_ext, array3_ext &phi, matrix3_ext &kappa)
{
    if (true == this->m_disable)
        return;

    // convert to internal precision
    real_int ab = (real_int)ab_ext;
    real_int bg = (real_int)bg_ext;
    real_int ag = (real_int)ag_ext;
    real_int deltacos = (real_int)deltacos_ext;
    real_int k = (real_int)k_ext;

    //double costh = this->CalcCosine(ab, bg, ag);
    array3_int d_cos_array;
    zeroarray3(d_cos_array);
    matrix3_int d2_cos_array;
    zeromatrix3(d2_cos_array);
    this->ThreeBodyCosineD(ab, bg, ag, d_cos_array);
    this->ThreeBodyCosineD2(ab, bg, ag, d2_cos_array);

    //Calculate Phi Vector
    for (int i = 0; i < 3; i++) {
        phi[i] = (real_ext)(k * deltacos * d_cos_array[i]);
    }

    //Calculate Kappa Matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            kappa[i][j] = (real_ext)(k * (d_cos_array[i] * d_cos_array[j] + deltacos * d2_cos_array[i][j]) );
        }
    }
}

//Calculate the Phi and Kappa for the Urey-Bradley Potential -> Not Implimented needs an overhaul
void StressGrid::UreyBradleyPhiKappa(real_ext ab_ext, real_ext bg_ext, real_ext ag_ext, real_ext costheta_ext, real_ext deltaRag_ext, real_ext deltatheta_ext, real_ext ktheta_ext, real_ext kUB_ext, array3_ext &phi, matrix3_ext &kappa)
{
    if (true == this->m_disable)
        return;

    // convert to internal precision
    real_int ab = (real_int)ab_ext;
    real_int bg = (real_int)bg_ext;
    real_int ag = (real_int)ag_ext;
    real_int costheta = (real_int)costheta_ext;
    real_int deltaRag = (real_int)deltaRag_ext;
    real_int deltatheta = (real_int)deltatheta_ext;
    real_int ktheta = (real_int)ktheta;
    real_int kUB = (real_int)kUB;

    //double costh = this->CalcCosine(ab, bg, ag);
    array3_int d_cos_array;
    zeroarray3(d_cos_array);

    matrix3_int d2_cos_array;
    zeromatrix3(d2_cos_array);

    array3_int d_theta_array;
    zeroarray3(d_theta_array);

    matrix3_int d2_theta_array;
    zeromatrix3(d2_theta_array);

    this->ThreeBodyCosineD(ab, bg, ag, d_cos_array);
    this->ThreeBodyCosineD2(ab, bg, ag, d2_cos_array);
    this->ThreeBodyThetaD(costheta, d_cos_array, d_theta_array);
    this->ThreeBodyThetaD2(costheta, d_cos_array, d2_cos_array, d2_theta_array);

    //Calculate Phi ab(0) and bg(1) for phi vector
    for (int i = 0; i < 3; i++) {
        phi[i] = (real_ext)(ktheta * deltatheta * d_theta_array[i]);
    }

    phi[iag] = (real_ext)(phi[iag] + kUB * deltaRag);

    //Calculate Kappa Matrix (will calculate kappa[2][2] separately)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            kappa[i][j] = (real_ext)(ktheta * (d_theta_array[i] * d_theta_array[j] + deltatheta * d2_theta_array[i][j]));
        }
    }
    kappa[iag][iag] = (real_ext)(kappa[iag][iag] + kUB);
}

//Calculate the Phi and Kappa due to the Bond Bond Cross Potential -> Implimented Gromos96
void StressGrid::BondBondCrossPhiKappa(real_ext k, real_ext deltarab, real_ext deltarbg, array3_ext &phi, matrix3_ext &kappa)
{
    if (true == this->m_disable)
        return;

    //Calculate Phi
    phi[0] = k * deltarbg;
    phi[1] = k * deltarab;
    phi[2] = 0;

    //Calculate Kappa
    //Zero Terms
    kappa[0][0] = 0;
    kappa[1][1] = 0;
    kappa[2][2] = 0;
    kappa[0][2] = 0;
    kappa[2][0] = 0;
    kappa[1][2] = 0;
    kappa[2][1] = 0;

    //Constant Terms
    kappa[0][1] = k;
    kappa[1][0] = k;
}

//Calculate the Phi and Kappa for the Bond Angle Cross Potential -> Implimented Gromos96
void StressGrid::BondAngleCrossPhiKappa(real_ext k, real_ext deltarab, real_ext deltarbg, real_ext deltarag, array3_ext &phi, matrix3_ext &kappa)
{
    if (true == this->m_disable)
        return;

    //Calculate Phi
    phi[0] = k * deltarag;
    phi[1] = phi[0];
    phi[2] = k * (deltarab + deltarbg);

    //Calculate Kappa
    //Zero Terms
    kappa[0][0] = 0;
    kappa[1][1] = 0;
    kappa[2][2] = 0;
    kappa[0][1] = 0;
    kappa[1][0] = 0;

    //Constant Terms
    kappa[0][2] = k;
    kappa[2][0] = k;
    kappa[1][2] = k;
    kappa[2][1] = k;
}

//Calculate the Phi and Kappa for Quartic Angle Potential -> Implimented
void StressGrid::QuarticAnglePhiKappa(real_ext ab_ext, real_ext bg_ext, real_ext ag_ext, real_ext costheta_ext, real_ext deltatheta_ext, real_ext (&coeff)[5], array3_ext &phi, matrix3_ext &kappa)
{
    if (true == this->m_disable)
        return;

    // convert to internal precision
    real_int ab = (real_int)ab_ext;
    real_int bg = (real_int)bg_ext;
    real_int ag = (real_int)ag_ext;
    real_int costheta = (real_int)costheta_ext;
    real_int deltatheta = (real_int)deltatheta_ext;

    //double costh = this->CalcCosine(ab, bg, ag);
    array3_int d_cos_array;
    zeroarray3(d_cos_array);

    matrix3_int d2_cos_array;
    zeromatrix3(d2_cos_array);

    array3_int d_theta_array;
    zeroarray3(d_theta_array);

    matrix3_int d2_theta_array;
    zeromatrix3(d2_theta_array);

    this->ThreeBodyCosineD(ab, bg, ag, d_cos_array);
    this->ThreeBodyCosineD2(ab, bg, ag, d2_cos_array);
    this->ThreeBodyThetaD(costheta, d_cos_array, d_theta_array);
    this->ThreeBodyThetaD2(costheta, d_cos_array, d2_cos_array, d2_theta_array);


    real_int deltathetasq = deltatheta * deltatheta;
    real_int deltathetacube = deltathetasq * deltatheta;

    //Calculate the Finate Sums
    real_int phiconst = (real_int)coeff[1] + realval_int(2.0) * (real_int)coeff[2] * deltatheta + realval_int(3.0) * (real_int)coeff[3] * deltathetasq + realval_int(4.0) * (real_int)coeff[4] * deltathetacube;
    real_int kappaconst = realval_int(2.0) * (real_int)coeff[2] + realval_int(6.0) * (real_int)coeff[3] * deltatheta + realval_int(12.0) * (real_int)coeff[4] * deltathetasq + realval_int(20.0);

    //Calculate Phi Vector
    for (int i = 0; i < 3; i++) {
        phi[i] = (real_int)(phiconst * d_theta_array[i]);
    }

    //Calculate Kappa Matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            kappa[i][j] = (real_int)(phiconst * d2_theta_array[i][j] + kappaconst * d_theta_array[i] * d_theta_array[j]);
        }
    }
}

/*Tetrahedron Calculations, use tetrahedron vector and tetrahedron matrix
 *Keep in mind all calculations find the dihedral angle between bonds ab and ge.
 *This dihedral angle is called thetabg as it is the twisting between beta and gamma.
 *
 *First Derivative: Define Tetrahedron Derivative Vector
 *
 * ab | ag | ae | bg | be | ge | = 0 | 1 | 2 | 3 | 4 | 5 |
 *
 * Second Derivative: Define Tetrahedron Derivative Matrix
 * abab | agab | aeab | bgab | beab | geab | = 00 | 01 | 02 | 03 | 04 | 05 |
 * abag | agag | aeag | bgag | beag | geag | = 10 | 11 | 12 | 13 | 14 | 15 |
 * abae | agae | aeae | bgae | beae | geae | = 20 | 21 | 22 | 23 | 24 | 25 |
 * abbg | agbg | aebg | bgbg | bebg | gebg | = 30 | 31 | 32 | 33 | 34 | 35 |
 * abbe | agbe | aebe | bgbe | bebe | gebe | = 40 | 41 | 42 | 43 | 44 | 45 |
 * abge | agge | aege | bgge | bege | gege | = 50 | 51 | 52 | 53 | 54 | 55 |
 *
*/
real_int StressGrid::CalcCosineDihedral(real_int ab, real_int ag, real_int ae, real_int bg, real_int be, real_int ge) {
    real_int numer, denom;

    numer = -(bg * bg * bg * bg) + (bg * bg) * ((ab * ab) - realval_int(2.0) * (ae * ae) + (ag * ag) + (be * be) + (ge * ge)) + ((ab * ab) - (ag * ag)) * (-(be * be) + (ge * ge));
    denom = sqrt((-ab + ag + bg) * (ab - ag + bg) * (ab + ag - bg) * (ab + ag + bg) * (-be + bg + ge) * (be - bg + ge) * (be + bg - ge) * (be + bg + ge));

    return numer / denom;
}

/*
*generate computation code in python
*void four_body_cosine_derivative() {
*}
*
*void four_body_cosine_derivative2() {
*}
*/

void StressGrid::FourBodyThetaD(real_int costheta, array6_int &d_cos_di_array, array6_int &d_theta_di_array) {
    real_int scalefactor;

    scalefactor = realval_int(-1.0) / sqrt(realval_int(1.0) - (costheta * costheta));

    for (int i = 0; i < 6; i++) {
        d_theta_di_array[i] = scalefactor * d_cos_di_array[i];
    }
}

void StressGrid::FourBodyThetaD2(real_int costheta, array6_int &d_cos_di_array, matrix6_int &d2_cos_di_array, matrix6_int &d2_theta_di_array) {
    real_int scalefactor;
    real_int sinthetasq;

    sinthetasq = realval_int(1.0) - (costheta * costheta);
    scalefactor = realval_int(1.0) / (sinthetasq * sqrt(sinthetasq));

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            d2_theta_di_array[i][j] = scalefactor * ((-sinthetasq) * d2_cos_di_array[i][j] - costheta * d_cos_di_array[i] * d_cos_di_array[j]);
        }
    }
}

//List of 4-body Potential Auxilery Functions
//Unimplimented, Will Be implimented Later
/*
*void impdihed_harmonic_phi_kappa(){
*
*}
*
*void propdihed_periodic_phi_kappa(){
*
*}
*
*void propdihed_fourier_phi_kappa(){
*
*}
*/
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
