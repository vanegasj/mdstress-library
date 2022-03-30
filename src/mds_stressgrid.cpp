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
    this->h_lapack        = NULL;
    this->p_Amat          = NULL;
    this->p_AmatT         = NULL;
    this->p_bvec          = NULL;
    this->p_Rij           = NULL;
    this->p_Fij           = NULL;
    this->p_Uij           = NULL;
    this->p_current_grid  = NULL;
    this->p_current_gridtot = NULL;
    this->p_current_grid_elborn  = NULL;
    this->p_current_grid_elkin  = NULL;
    this->p_current_gridc = NULL;
    this->p_sum_grid      = NULL;
    this->p_avg_grid      = NULL;
    this->p_avg_gridtot   = NULL;
    this->p_sum_gridc     = NULL;
    this->p_sum_grid_elcovar = NULL;
    this->p_sum_grid_elkin = NULL;
    this->p_sum_grid_elborn = NULL;
    this->p_sum_grid_volcovar = NULL;
    this->p_sum_volume    = NULL;
    this->p_molecule_id   = NULL;
    this->p_radii         = NULL;
    this->p_positions     = NULL;
    this->p_pos_gridc     = NULL;

    this->m_nodispcor   = false;
    this->m_cuda        = false;
    this->m_periodic[0] = false;
    this->m_periodic[1] = false;
    this->m_periodic[2] = false;
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
    if (this->p_Amat                 != NULL ) delete [] this->p_Amat;
    if (this->p_AmatT                != NULL ) delete [] this->p_AmatT;
    if (this->p_bvec                 != NULL ) delete [] this->p_bvec;
    if (this->p_Rij                  != NULL ) delete [] this->p_Rij;
    if (this->p_Fij                  != NULL ) delete [] this->p_Fij;
    if (this->p_Uij                  != NULL ) delete [] this->p_Uij;
    if (this->p_current_grid         != NULL ) delete [] this->p_current_grid;
    if (this->p_current_gridtot      != NULL ) delete [] this->p_current_gridtot;
    if (this->p_current_grid_elborn  != NULL ) delete [] this->p_current_grid_elborn;
    if (this->p_current_grid_elkin   != NULL ) delete [] this->p_current_grid_elkin;
    if (this->p_current_gridc        != NULL ) delete [] this->p_current_gridc;
    if (this->p_sum_grid             != NULL ) delete [] this->p_sum_grid;
    if (this->p_avg_grid             != NULL ) delete [] this->p_avg_grid;
    if (this->p_avg_gridtot          != NULL ) delete [] this->p_avg_gridtot;
    if (this->p_sum_grid_elcovar     != NULL ) delete [] this->p_sum_grid_elcovar;
    if (this->p_sum_grid_elkin       != NULL ) delete [] this->p_sum_grid_elkin;
    if (this->p_sum_grid_elborn      != NULL ) delete [] this->p_sum_grid_elborn;
    if (this->p_sum_grid_volcovar    != NULL ) delete [] this->p_sum_grid_volcovar;
    if (this->p_sum_gridc            != NULL ) delete [] this->p_sum_gridc;
    if (this->p_sum_volume           != NULL ) delete [] this->p_sum_volume;
    if (this->p_molecule_id          != NULL ) delete [] this->p_molecule_id;
    if (this->p_radii                != NULL ) delete [] this->p_radii;
    if (this->p_positions            != NULL ) delete [] this->p_positions;
    if (this->p_pos_gridc            != NULL ) delete [] this->p_pos_gridc;
    if (this->h_lapack               != NULL )
    {
        for (int i = 0; i < m_max_threads; ++i)
            delete this->h_lapack[i];
        delete this->h_lapack;
    }
    
    this->m_avg_boxvol    = 0.0;
    this->m_var_boxvol    = 0.0;
    this->m_maxpart       = 0;
    this->p_Amat          = NULL;
    this->p_AmatT         = NULL;
    this->p_bvec          = NULL;
    this->p_Rij           = NULL;
    this->p_Fij           = NULL;
    this->p_Uij           = NULL;
    this->p_current_grid  = NULL;
    this->p_current_gridtot  = NULL;
    this->p_current_grid_elborn  = NULL;
    this->p_current_grid_elkin  = NULL;
    this->p_current_gridc = NULL;
    this->p_sum_grid      = NULL;
    this->p_avg_grid      = NULL;
    this->p_avg_gridtot   = NULL;
    this->p_sum_grid_elcovar      = NULL;
    this->p_sum_grid_elkin      = NULL;
    this->p_sum_grid_elborn      = NULL;
    this->p_sum_grid_volcovar      = NULL;
    this->p_sum_gridc     = NULL;
    this->p_sum_volume    = NULL;
    this->p_molecule_id   = NULL;
    this->p_radii         = NULL;
    this->p_positions     = NULL;
    this->p_pos_gridc     = NULL;
    this->h_lapack        = NULL;

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
            
            if ( iszeromatrix(this->m_box) )
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
            
            if ( iszeromatrix(this->m_box) )
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
                this->p_pos_gridc     = new darray  [this->m_ncellsc];
                this->p_sum_gridc     = new double  [this->m_ncellsc];
                this->p_current_gridc = new double  [this->m_ncellsc*this->m_max_threads];
                
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
            this->p_radii = new double[this->m_ncells];
            this->p_positions = new double[3*this->m_ncells];
            this->p_sum_volume = new double[this->m_ncells];

            // set them to defaults
            for ( int i=0; i < this->m_ncells; ++i )
            {
                this->p_molecule_id[i] = 0;
                this->p_radii[i] = 0.001;
                this->p_sum_volume[i] = 0.0;
        
                this->p_positions[3*i] = 0.0;
                this->p_positions[3*i+1] = 0.0;
                this->p_positions[3*i+2] = 0.0;
            }
        }

        //Give size to current and sum grid
        this->p_sum_grid            = new dmatrix [this->m_ncells];
        this->p_avg_grid            = new dmatrix [this->m_ncells];
        this->p_avg_gridtot         = new dmatrix [1];
        this->p_sum_grid_elcovar    = new dmatrix6 [this->m_ncells];
        this->p_sum_grid_elkin      = new dmatrix6 [this->m_ncells];
        this->p_sum_grid_elborn     = new dmatrix6 [this->m_ncells];
        this->p_sum_grid_volcovar   = new dmatrix [this->m_ncells];
        this->p_current_grid_elkin  = new dmatrix6 [this->m_ncells*this->m_max_threads];
        this->p_current_grid_elborn = new dmatrix6 [this->m_ncells*this->m_max_threads];
        this->p_current_grid        = new dmatrix [this->m_ncells*this->m_max_threads];
        this->p_current_gridtot     = new dmatrix [1];

        //Set all to zero
        this->m_nframes = 0;
        this->m_avg_boxvol = 0.0;
        this->m_var_boxvol = 0.0;
        zeromatrix(p_current_gridtot[0]);
        zeromatrix(p_avg_gridtot[0]);
        for (int i=0; i < this->m_ncells; i++)
        {
            zeromatrix(this->p_sum_grid[i]);
            zeromatrix(this->p_avg_grid[i]);
            zeromatrix(this->p_sum_grid_volcovar[i]);
            zeromatrix6(this->p_sum_grid_elcovar[i]);
            zeromatrix6(this->p_sum_grid_elborn[i]);
            zeromatrix6(this->p_sum_grid_elkin[i]);
        }
        for (int i=0; i < this->m_ncells*this->m_max_threads; i++)
        {
            zeromatrix(this->p_current_grid[i]);
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
    }
}

void StressGrid::SetPeriodicBoundaries(bool x, bool y, bool z, bool enforce)
{ 
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
void StressGrid::UpdateBoxSpacings ( dmatrix box )
{
    if ( !this->m_ierr )
    {
        // every thread must process this latch before proceeding
        static barrier ubs_entry(this->m_max_threads);
        ubs_entry.count_down_and_wait();

        // only thread 0 performs update
        if (this->m_thread_map[std::this_thread::get_id()] == 0)
        {
            copymatrix( box, this->m_box);
            inversematrix( this->m_box, this->m_invbox );

            this->m_gridsp[0] = this->m_box[0][0]/static_cast<double>(this->m_nxyz[0]);
            this->m_gridsp[1] = this->m_box[1][1]/static_cast<double>(this->m_nxyz[1]);
            this->m_gridsp[2] = this->m_box[2][2]/static_cast<double>(this->m_nxyz[2]);
            this->m_gridsp[3] = this->m_gridsp[0]*this->m_gridsp[1];
            this->m_gridsp[4] = this->m_gridsp[0]*this->m_gridsp[2];
            this->m_gridsp[5] = this->m_gridsp[1]*this->m_gridsp[2];
            this->m_gridsp[6] = this->m_gridsp[0]*this->m_gridsp[1]*this->m_gridsp[2];
            this->m_invgridsp =  1.0/(this->m_gridsp[0]*this->m_gridsp[1]*this->m_gridsp[2]);

            if (this->m_gridctype != mds_gridc_off)
            {
                this->m_gridspc[0] = this->m_box[0][0]/static_cast<double>(this->m_nxyzc[0]);
                this->m_gridspc[1] = this->m_box[1][1]/static_cast<double>(this->m_nxyzc[1]);
                this->m_gridspc[2] = this->m_box[2][2]/static_cast<double>(this->m_nxyzc[2]);
                this->m_gridspc[3] = this->m_gridspc[0]*this->m_gridspc[1];
                this->m_gridspc[4] = this->m_gridspc[0]*this->m_gridspc[2];
                this->m_gridspc[5] = this->m_gridspc[1]*this->m_gridspc[2];
                this->m_gridspc[6] = this->m_gridspc[0]*this->m_gridspc[1]*this->m_gridspc[2];
                this->m_invgridspc = 1.0/(this->m_gridspc[0]*this->m_gridspc[1]*this->m_gridspc[2]);

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

            summatrix( this->m_box, this->m_sumbox, this->m_sumbox );
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
            double beta = this->m_ewaldcoeff_q;
            // do coulomb distribute stress here
            for (int i = thread_id; i < this->m_ncellsc; i+=this->m_max_threads)
            {
                // calculate the charge
                double qi = this->p_current_gridc[i];
                if (abs(qi) > 1E-16)
                {
                    qi /= this->m_invgridspc;
                    // calculate the indices
                    for (int j = i+1; j < this->m_ncellsc; j+=1)
                    {
                        // calculate the charge
                        double qj = this->p_current_gridc[j];
                        if (abs(qj) > 1E-16)
                        {
                            qj /= this->m_invgridspc;

                            // calculate r
                            darray diff;
                            diffarray(this->p_pos_gridc[i], this->p_pos_gridc[j], diff, this->m_box, this->m_periodic);

                            // correct Ri
                            darray Ri;
                            sumarray(this->p_pos_gridc[j], diff, Ri);

                            // calculate r and rinv
                            double r2 = diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2];
                            double r = sqrt(r2);
                            double rinv = 1.0/r;

                            // calculate the force
                            //double F = -this->m_epsfac*(qi*qj)*(rinv*rinv*rinv);
                            double F = -this->m_epsfac*qi*qj*(2*beta*exp(-beta*beta*r2)/sqrt(PI) - erf(beta*r)*rinv)*rinv*rinv;
                            // calculate the force vectors
                            darray Fij = {
                                F*diff[0],
                                F*diff[1],
                                F*diff[2],
                                };

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
                    summatrix(this->p_current_grid[i], this->p_current_grid[i+j*this->m_ncells], this->p_current_grid[i] );
                    zeromatrix(this->p_current_grid[i+j*this->m_ncells]);
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

            at each iteraton we first increment the counter
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
            double Vnew, deltaV, deltaV2;
            Vnew = this->m_box[0][0]*this->m_box[1][1]*this->m_box[2][2];
            deltaV = Vnew - this->m_avg_boxvol;
            this->m_avg_boxvol += deltaV/this->m_nframes;
            deltaV2 = Vnew - this->m_avg_boxvol;
            this->m_var_boxvol += deltaV*deltaV2;
            //printf("m_avg_boxvol = %8.8e, m_var_boxvol = %8.8e, deltaV = %8.8e\n", this->m_avg_boxvol, this->m_var_boxvol, deltaV);

            if (this->m_spatatom == mds_spat)
            {
                for (int i = 0; i < this->m_ncells; i++)
                {
                    scalesummatrix( 1.0/this->m_ncells, this->p_current_grid[i], this->p_current_gridtot[0]); // compute y = sigma_total_kl = sum(sigma_local_kl)/m_ncells
                    //summatrix( this->p_current_gridtot[0], this->p_current_grid[i], this->p_current_gridtot[0]); // compute sigma_total_kl = y
                }
                scalesummatrix(-1.0, this->p_avg_gridtot[0], this->p_current_gridtot[0]); // subtract meany from y and store back into y (which now becomes dy)
                scalesummatrix(1.0/this->m_nframes, this->p_current_gridtot[0], this->p_avg_gridtot[0]); //compute meany
                dmatrix6 tmp_covar[1];
                dmatrix dx[1];
                for (int i = 0; i < this->m_ncells; i++)
                {
                    summatrix( this->p_sum_grid[i], this->p_current_grid[i], this->p_sum_grid[i] );
                    summatrix6( this->p_sum_grid_elborn[i], this->p_current_grid_elborn[i], this->p_sum_grid_elborn[i] );
                    summatrix6( this->p_sum_grid_elkin[i], this->p_current_grid_elkin[i], this->p_sum_grid_elkin[i] );
                    scalematrix(this->p_avg_grid[i], -1.0, dx[0]); // dx = -meanx
                    summatrix(dx[0], this->p_current_grid[i], dx[0]); // dx += x
                    scalesummatrix(1.0/this->m_nframes, dx[0], this->p_avg_grid[i]); //compute meanx
                    scalesummatrix(deltaV, dx[0], this->p_sum_grid_volcovar[i]); // computing covar(sigma_local_ij, Vol)
                    //printf("covar(sigma_local_ij,Vol) = %8.8e\n", this->p_sum_grid_volcovar[i][0][0]);
                    matrixouterprod( dx[0], this->p_current_gridtot[0], tmp_covar[0]); // dx*dy
                    //matrixouterprod( dx[0], dx[0], tmp_covar[0]); // dx*dy
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
                double vor_box[3];
                vor_box[0] = this->m_box[0][0];
                vor_box[1] = this->m_box[1][1];
                vor_box[2] = this->m_box[2][2];

                double gfxy,gfxz;
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
                        double px = this->p_positions[3*i];
                        double py = this->p_positions[3*i+1];
                        double pz = this->p_positions[3*i+2];

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
                double last_volume = 0.0;

                int cells_computed = 0;
                if (vl.start())
                {
                    do{
                        if (this->p_radii[pid] > 0.0 && vorcon.compute_cell(c,vl))
                        {
                            // count the cells
                            cells_computed += 1;

                            // get the volume of this cell
                            last_volume = c.volume();
                            last_pid = pid;

                            // mark the last volume encountered
                            this_molecule = this->p_molecule_id[pid];

                            scalematrix( this->p_current_grid[pid], 1.0/last_volume, this->p_current_grid[pid] );
                            summatrix( this->p_sum_grid[pid], this->p_current_grid[pid], this->p_sum_grid[pid] );

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
                                scalematrix( this->p_current_grid[pid], 1.0/last_volume, this->p_current_grid[pid] );
                                summatrix( this->p_sum_grid[last_pid], this->p_current_grid[pid], this->p_sum_grid[last_pid] );
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

            zeromatrix(this->p_current_gridtot[0]);
            for(int i = 0; i < this->m_ncells; ++i)
            {
                zeromatrix(this->p_current_grid[i]);
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

void StressGrid::DispersionCorrection (double shift)
{
    if (this->m_nodispcor == false)
    {
        // select the correct grid
        int batch_id = this->m_thread_map[std::this_thread::get_id()];
        dmatrix * grid = this->p_current_grid+batch_id*this->m_ncells;

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
            zeromatrix(this->p_current_grid[i+thread_id*this->m_max_threads]);
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
            zeromatrix(this->p_current_gridtot[0]);
            zeromatrix(this->p_avg_gridtot[0]);
            for( int i=0; i<this->m_ncells; i++ )
            {
                zeromatrix ( this->p_sum_grid[i] );
                zeromatrix ( this->p_avg_grid[i] );
                zeromatrix ( this->p_sum_grid_volcovar[i] );
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
    if ( !this->m_ierr)
    {
            int                Dtype=1;
            dmatrix            avgbox;
            std::string        outname, charge_outname,  elborn_outname, elcovar_outname, elkin_outname, eltotal_outname;
            std::ostringstream outnumber;
            FILE              *outfile,*charge_outfile, *elborn_outfile, *elcovar_outfile, *elkin_outfile, *eltotal_outfile;
            double             stressfac, stress2fac, covfac, covfac2;

            outnumber << this->m_nreset;
            size_t lastindex = this->m_filename.find_last_of(".");
            std::string rawname = this->m_filename.substr(0, lastindex);

            //Change output format if the user specifies a filename with a .dat extension
            outname = this->m_filename + outnumber.str();
            if (outname.find(".dat") == std::string::npos)
                outname = outname + "." + mds_fileext;

            elcovar_outname = rawname + "_elcovar.dat" + outnumber.str();
            if (elcovar_outname.find(".dat") == std::string::npos)
                elcovar_outname = elcovar_outname + "." + mds_fileext;

            elkin_outname = rawname + "_elkin.dat" + outnumber.str();
            if (elkin_outname.find(".dat") == std::string::npos)
                elkin_outname = elkin_outname + "." + mds_fileext;

            elborn_outname = rawname + "_elborn.dat" + outnumber.str();
            if (elborn_outname.find(".dat") == std::string::npos)
                elborn_outname = elborn_outname + "." + mds_fileext;

            eltotal_outname = rawname + "_eltotal.dat" + outnumber.str();
            if (eltotal_outname.find(".dat") == std::string::npos)
                eltotal_outname = eltotal_outname + "." + mds_fileext;

            if (this->m_spatatom == mds_spat)
                Dtype = 1;
            else if (this->m_spatatom == mds_atom)
                Dtype = 2;

            outfile = fopen(outname.c_str(), "wb" );
            fwrite(&Dtype, sizeof(int), 1, outfile);

            // use different dtype for elasticity file
            Dtype = 6;
            elcovar_outfile = fopen(elcovar_outname.c_str(), "wb" );
            fwrite(&Dtype, sizeof(int), 1, elcovar_outfile);
            elborn_outfile = fopen(elborn_outname.c_str(), "wb" );
            fwrite(&Dtype, sizeof(int), 1, elborn_outfile);
            elkin_outfile = fopen(elkin_outname.c_str(), "wb" );
            fwrite(&Dtype, sizeof(int), 1, elkin_outfile);
            eltotal_outfile = fopen(eltotal_outname.c_str(), "wb" );
            fwrite(&Dtype, sizeof(int), 1, eltotal_outfile);

            //Divide sumbox with respect to the number of frames to get the avg
            scalematrix( this->m_sumbox, 1.0/this->m_nframes, avgbox);
            fwrite(avgbox, sizeof(dmatrix), 1, outfile);
            fwrite(avgbox, sizeof(dmatrix), 1, elcovar_outfile);
            fwrite(avgbox, sizeof(dmatrix), 1, elborn_outfile);
            fwrite(avgbox, sizeof(dmatrix), 1, elkin_outfile);
            fwrite(avgbox, sizeof(dmatrix), 1, eltotal_outfile);

            if (this->m_spatatom == mds_spat)
            {
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
                fwrite(&this->m_ncells, sizeof(int), 1, outfile);
                fwrite(&this->m_ncells, sizeof(int), 1, elcovar_outfile);
                fwrite(&this->m_ncells, sizeof(int), 1, elborn_outfile);
                fwrite(&this->m_ncells, sizeof(int), 1, elkin_outfile);
                fwrite(&this->m_ncells, sizeof(int), 1, eltotal_outfile);
            }
            stressfac = mds_units/this->m_nframes;
            stress2fac = mds_units*mds_units/this->m_nframes;
            //this->m_var_boxvol /= this->m_nframes;
            //covfac = -mds_units*mds_units*this->m_sum_boxvol/(this->m_temperature*KBfac*this->m_nframes*this->m_ncells);
            covfac = -mds_units*mds_units*this->m_avg_boxvol/(this->m_temperature*KBfac*this->m_nframes);
            covfac2 = 0.0;
            if (this->m_var_boxvol > 1e-12){
                covfac2 = mds_units*this->m_avg_boxvol/this->m_var_boxvol; // we don't need to divide by m_nframes here since the Var(Vol) is also not divided by m_nframes and the factors cancel out
            }
            dmatrix6 *tmp_covar;
            dmatrix IM;
            tmp_covar = new dmatrix6[1];
            IM[0][0] = 1.0; IM[0][1] = 0.0; IM[0][2] = 0.0;
            IM[1][0] = 0.0; IM[1][1] = 1.0; IM[1][2] = 0.0;
            IM[2][0] = 0.0; IM[2][1] = 0.0; IM[2][2] = 1.0;
            for ( int i = 0; i < this->m_ncells; i++ )
            {
                scalematrix(this->p_sum_grid[i], stressfac, this->p_sum_grid[i]);
                scalematrix6(this->p_sum_grid_elcovar[i], covfac, this->p_sum_grid_elcovar[i]);
                scalematrix6(this->p_sum_grid_elborn[i], stressfac, this->p_sum_grid_elborn[i]);
                scalematrix6(this->p_sum_grid_elkin[i], stressfac, this->p_sum_grid_elkin[i]);
                scalematrix(this->p_sum_grid_volcovar[i], covfac2, this->p_sum_grid_volcovar[i]); // <V>*Cov(sigma_local_ij*V)/Var(V)
                matrixouterprod(this->p_sum_grid_volcovar[i], IM, tmp_covar[0]); // <V>*Cov(sigma_local_ij*V)/Var(V)*delta_kl
                summatrix6(this->p_sum_grid_elcovar[i], tmp_covar[0], this->p_sum_grid_elcovar[i]);
            }

            fwrite(this->p_sum_grid, sizeof(dmatrix), this->m_ncells, outfile);
            fwrite(this->p_sum_grid_elcovar, sizeof(dmatrix6), this->m_ncells, elcovar_outfile);
            fwrite(this->p_sum_grid_elborn, sizeof(dmatrix6), this->m_ncells, elborn_outfile);
            fwrite(this->p_sum_grid_elkin, sizeof(dmatrix6), this->m_ncells, elkin_outfile);

            // append the volume data
            if (this->m_spatatom == mds_atom)
            {
                for ( int i = 0; i < this->m_ncells; i++ )
                    this->p_sum_volume[i] /= this->m_nframes;
                fwrite(this->p_sum_volume, sizeof(double), this->m_ncells, outfile);
            }

            fclose(outfile);
            fclose(elcovar_outfile);
            fclose(elborn_outfile);
            fclose(elkin_outfile);
            for ( int i = 0; i < this->m_ncells; i++ )
            {
                summatrix6(this->p_sum_grid_elcovar[i], this->p_sum_grid_elborn[i], this->p_sum_grid_elcovar[i]); // add up all the contributions to the total elasticity tensor
                summatrix6(this->p_sum_grid_elcovar[i], this->p_sum_grid_elkin[i], this->p_sum_grid_elcovar[i]);
            }

            fwrite(this->p_sum_grid_elcovar, sizeof(dmatrix6), this->m_ncells, eltotal_outfile);
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
                fwrite(avgbox, sizeof(dmatrix), 1, charge_outfile);

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
                }

                fwrite(this->p_sum_gridc, sizeof(double), this->m_ncellsc, charge_outfile);
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
void StressGrid::SetVoronoiRadius(double radius, int atomID)
{
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
        this->p_radii[atomID] = std::max(radius,0.001);
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
void StressGrid::AddVoronoiAtom(double px, double py, double pz, int atomID, int moleID)
{
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
void StressGrid::DistributeInteraction(int nAtoms, darray *R, darray *F, int *atomIDs = NULL)
{
    int    n;
    int    i,j;
    double temp;
    dmatrix stress;

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
                    this->DistributePairInteraction( R[0], R[1], F[0], batch_id );
                    break;

                case 3:
                    this->DistributeN3( R[0], R[1], R[2],F[0], F[1], F[2], batch_id );
                    break;

                case -3:
                    this->DistributeSettle( R[0], R[1], R[2],F[0], F[1], F[2], batch_id );
                    break;

                case 4:
                    this->DistributeN4( R[0], R[1], R[2], R[3], F[0], F[1], F[2], F[3], batch_id );
                    break;

                case 5:
                    this->DistributeN5( R[0], R[1], R[2], R[3], R[4], F[0], F[1], F[2], F[3], F[4], batch_id );
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

            if (atomIDs == NULL)
            {
                std::cout << "ERROR:: the atomIDs array is NULL. Cannot calculate the stress/atom.";
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
                summatrix (this->p_current_grid[atomIDs[n]],stress,this->p_current_grid[atomIDs[n]]);
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
void StressGrid::ComputeNbodyPairForces(int nAtoms, darray *R, darray *F, int *atomIDs = NULL)
{
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
void StressGrid::DistributePairInteraction( darray xi, darray xj, darray F, int batch_id )
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

void StressGrid::DistributePairInteraction1D(darray xi, darray xj, darray F, int batch_id )
{
    // select grid based on batch index
    dmatrix * grid = this->p_current_grid+batch_id*this->m_ncells;
    
    //------------------------------------------------------------------------------------
    // Calculate the stress tensor
    darray diff;
    diffarray( xj, xi, diff, this->m_box, this->m_periodic);
    dmatrix stress;
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
    int x = this->m_nxyz[this->m_griddim] * xi[this->m_griddim] * this->m_invbox[this->m_griddim][this->m_griddim] - (xi[this->m_griddim] < 0.0);
    const int i2 = this->m_nxyz[this->m_griddim] * xj[this->m_griddim] * this->m_invbox[this->m_griddim][this->m_griddim] - (xj[this->m_griddim] < 0.0);
    const int c = (i2>x)-(x>i2);
    const double t_c1 = xi[this->m_griddim] / (xi[this->m_griddim]-xj[this->m_griddim]);
    const double t_c2 = this->m_gridsp[this->m_griddim] / (xi[this->m_griddim]-xj[this->m_griddim]);
    const double C = 0.5*this->m_invgridsp*this->m_invgridsp;
    double d_cgrid = xi[this->m_griddim]-(x+0.5)*this->m_gridsp[this->m_griddim];
    int xn = x+(c+1)/2;

    //------------------------------------------------------------------------------------
    // Distribute the stress

    // track previous time of crossing and check that sum is complete (?)
    double oldt = 0.0; 

    // fix the number of iterations
    const int iterations = c*(i2-x);
    for (int count = 0; count <= iterations; ++count)
    {
        // there is always iterations+1, where the last iteration deals
        // with any residual
        double newt = (iterations == count) ? 1.0 : t_c1-xn*t_c2;

        // work out the parametric time constants
        const double t12 = oldt*oldt;
        const double t22 = newt*newt;
        const double dt1 = newt - oldt;
        const double dt2 = t22 - t12;

        const int p1 = ((x + 1 + this->m_nxyz[this->m_griddim]) % this->m_nxyz[this->m_griddim]);
        const int m1 = ((x + this->m_nxyz[this->m_griddim]) % this->m_nxyz[this->m_griddim]);

        // the composite constants in terms of i, j, k
        const double D1 = this->m_gridsp[5-this->m_griddim]*(2.0*d_cgrid*dt1+diff[this->m_griddim]*dt2);
        const double D2 = this->m_gridsp[6]*dt1;
        scalesummatrix(C*( D1 + D2), stress, grid[p1]);
        scalesummatrix(C*(-D1 + D2), stress, grid[m1]);
        
        d_cgrid -= c * m_gridsp[this->m_griddim];
        oldt = newt;
        
        x += c;
        xn += c;
    }
}

void StressGrid::DistributePairInteraction3D(darray xi, darray xj, darray F, int batch_id )
{
    // this is the 3D case
    dmatrix * grid = this->p_current_grid+batch_id*this->m_ncells;
    
    //------------------------------------------------------------------------------------
    // Calculate the stress tensor
    darray diff;
    diffarray( xj, xi, diff, this->m_box, this->m_periodic);

    dmatrix stress;
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
    darray t,t_c1,t_c2,d_cgrid;
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
    t[0] = (c[0] == 0) ? 1.1 : t_c1[0]-xn[0]*t_c2[0];
    t[1] = (c[1] == 0) ? 1.1 : t_c1[1]-xn[1]*t_c2[1];
    t[2] = (c[2] == 0) ? 1.1 : t_c1[2]-xn[2]*t_c2[2];

    //------------------------------------------------------------------------------------
    // Distribute the stress

    // now the position/spatial constants
    const double C = 0.125*this->m_invgridsp*this->m_invgridsp;
    const double axy = diff[0]*diff[1];
    const double axz = diff[0]*diff[2];
    const double ayz = diff[1]*diff[2];
    const double axyz = diff[0]*ayz; 
    
    // track previous time of crossing
    double oldt = 0.0; 
    
    const int iterations = c[0]*(i2[0]-x[0]) + c[1]*(i2[1]-x[1]) + c[2]*(i2[2]-x[2]);
    for (int count = 0; count <= iterations; ++count)
    {
        // figure out index
        const int cmp0x = ((t[0]<t[1]+mds_eps) + (t[0]<t[2]+mds_eps))/2;
        const int cmp1x = ((t[1]<t[0]+mds_eps) + (t[1]<t[2]+mds_eps))/2;
        const int cmp2x = ((t[2]<t[0]+mds_eps) + (t[2]<t[1]+mds_eps))/2;
        const int iX = (1-cmp0x)*(cmp1x+2*(1-cmp1x)*cmp2x);
        const double newt = (iterations == count) ? 1.0 : t[iX];

        // work out the parametric time constants
        const double t12 = oldt*oldt;
        const double t22 = newt*newt;
        const double dt1 = newt - oldt;
        const double dt2 = t22 - t12;
        const double dt3 = 4.0*(t22*newt - t12*oldt)/3.0;
        const double dt4 = t22*t22 - t12*t12;

        // additional constants
        const double bxy = d_cgrid[0]*d_cgrid[1];
        const double bxz = d_cgrid[0]*d_cgrid[2];
        const double byz = d_cgrid[1]*d_cgrid[2];
        const double bxyz = d_cgrid[0]*byz;
    
        const int iip1 = ((x[0] + 1 + this->m_nxyz[0]) % this->m_nxyz[0])*this->m_nxyz[1]*this->m_nxyz[2];
        const int jjp1 = ((x[1] + 1 + this->m_nxyz[1]) % this->m_nxyz[1])*this->m_nxyz[2];
        const int kkp1 = ((x[2] + 1 + this->m_nxyz[2]) % this->m_nxyz[2]);
        const int iim1 = ((x[0] + this->m_nxyz[0]) % this->m_nxyz[0])*this->m_nxyz[1]*this->m_nxyz[2];
        const int jjm1 = ((x[1] + this->m_nxyz[1]) % this->m_nxyz[1])*this->m_nxyz[2];
        const int kkm1 = ((x[2] + this->m_nxyz[2]) % this->m_nxyz[2]);

        // the composite constants in terms of i, j, k
        const double D[8] = {
            8.0*bxyz*dt1 + 4.0*(diff[0]*byz+diff[1]*bxz+diff[2]*bxy)*dt2
                + 2.0*(d_cgrid[0]*ayz+d_cgrid[1]*axz+d_cgrid[2]*axy)*dt3 + 2.0*axyz*dt4,
            this->m_gridsp[0]*(4.0*byz*dt1 + 2.0*(diff[1]*d_cgrid[2]+diff[2]*d_cgrid[1])*dt2 + ayz*dt3),
            this->m_gridsp[1]*(4.0*bxz*dt1 + 2.0*(diff[0]*d_cgrid[2]+diff[2]*d_cgrid[0])*dt2 + axz*dt3),
            this->m_gridsp[2]*(4.0*bxy*dt1 + 2.0*(diff[0]*d_cgrid[1]+diff[1]*d_cgrid[0])*dt2 + axy*dt3),
            this->m_gridsp[3]*(2.0*d_cgrid[2]*dt1+diff[2]*dt2),
            this->m_gridsp[4]*(2.0*d_cgrid[1]*dt1+diff[1]*dt2),
            this->m_gridsp[5]*(2.0*d_cgrid[0]*dt1+diff[0]*dt2),
            this->m_gridsp[6]*dt1,
        };

        // perform the sums into the grid
        scalesummatrix(C*( D[0] + D[1] + D[2] + D[3] + D[4] + D[5] + D[6] + D[7]), stress, grid[iip1 + jjp1 + kkp1]);
        scalesummatrix(C*(-D[0] - D[1] - D[2] + D[3] - D[4] + D[5] + D[6] + D[7]), stress, grid[iip1 + jjp1 + kkm1]);
        scalesummatrix(C*(-D[0] - D[1] + D[2] - D[3] + D[4] - D[5] + D[6] + D[7]), stress, grid[iip1 + jjm1 + kkp1]);
        scalesummatrix(C*( D[0] + D[1] - D[2] - D[3] - D[4] - D[5] + D[6] + D[7]), stress, grid[iip1 + jjm1 + kkm1]);
        scalesummatrix(C*(-D[0] + D[1] - D[2] - D[3] + D[4] + D[5] - D[6] + D[7]), stress, grid[iim1 + jjp1 + kkp1]);
        scalesummatrix(C*( D[0] - D[1] + D[2] - D[3] - D[4] + D[5] - D[6] + D[7]), stress, grid[iim1 + jjp1 + kkm1]);
        scalesummatrix(C*( D[0] - D[1] - D[2] + D[3] + D[4] - D[5] - D[6] + D[7]), stress, grid[iim1 + jjm1 + kkp1]);
        scalesummatrix(C*(-D[0] + D[1] + D[2] + D[3] - D[4] - D[5] - D[6] + D[7]), stress, grid[iim1 + jjm1 + kkm1]);

        d_cgrid[iX] -= c[iX] * m_gridsp[iX];
        oldt = t[iX];
        
        x[iX] += c[iX];
        xn[iX] += c[iX];

        // Next cross point:
        t[iX] = t_c1[iX]-xn[iX]*t_c2[iX];
    }
}


void StressGrid::DistributePairElast(darray xi, darray xj, darray xk, darray xl, double phi, double kappa)
{
    // this is the 3D case
    int batch_id = this->m_thread_map[std::this_thread::get_id()];
    dmatrix6 * gridElast = this->p_current_grid_elborn+batch_id*this->m_ncells;

    //------------------------------------------------------------------------------------
    // Calculate the stress tensor
    darray diff, diff2;
    double rinv, rinv2, rinv3;
    diffarray( xj, xi, diff, this->m_box, this->m_periodic);
    diffarray( xl, xk, diff2, this->m_box, this->m_periodic);
    rinv = 1.0/normarray(diff);
    rinv2 = rinv/normarray(diff2);
    rinv3 = rinv*rinv*rinv;

    // Stiffness matrix in Voigt notation
    // 0 = xx; 1 = yy; 2 = zz; 3 = yz or zy; 4 = xz or zx; 5 = xy or yx
    // All indices                         Voigt indices           Stress indices
    // ( xxxx xxyy xxzz xxyz xxxz xxxy ) = ( 00 01 02 03 04 05 ) = [ 0000 0011 0022 0012 0002 0001 ]
    // (      yyyy yyzz yyyz yyxz yyxy ) = (    11 12 13 14 15 ) = [      1111 1122 1112 1102 1101 ]
    // (           zzzz zzyz zzxz zzxy ) = (       22 23 24 25 ) = [           2222 2212 2202 2201 ]
    // (                yzyz yzxz yzxy ) = (          33 34 35 ) = [                1212 1202 1201 ]
    // (                     xzxz xzxy ) = (             44 45 ) = [                     0202 0201 ]
    // (                          xyxy ) = (                55 ) = [                          0101 ]

    dmatrix6 elast;
    elast[0][0] = kappa*diff[0]*diff[0]*diff2[0]*diff2[0]*rinv2 - phi*diff[0]*diff[0]*diff[0]*diff[0]*rinv3;
    elast[0][1] = kappa*diff[0]*diff[0]*diff2[1]*diff2[1]*rinv2 - phi*diff[0]*diff[0]*diff[1]*diff[1]*rinv3;
    elast[0][2] = kappa*diff[0]*diff[0]*diff2[2]*diff2[2]*rinv2 - phi*diff[0]*diff[0]*diff[2]*diff[2]*rinv3;
    elast[0][3] = kappa*diff[0]*diff[0]*diff2[1]*diff2[2]*rinv2 - phi*diff[0]*diff[0]*diff[1]*diff[2]*rinv3;
    elast[0][4] = kappa*diff[0]*diff[0]*diff2[0]*diff2[2]*rinv2 - phi*diff[0]*diff[0]*diff[0]*diff[2]*rinv3;
    elast[0][5] = kappa*diff[0]*diff[0]*diff2[0]*diff2[1]*rinv2 - phi*diff[0]*diff[0]*diff[0]*diff[1]*rinv3;
    elast[1][1] = kappa*diff[1]*diff[1]*diff2[1]*diff2[1]*rinv2 - phi*diff[1]*diff[1]*diff[1]*diff[1]*rinv3;
    elast[1][2] = kappa*diff[1]*diff[1]*diff2[2]*diff2[2]*rinv2 - phi*diff[1]*diff[1]*diff[2]*diff[2]*rinv3;
    elast[1][3] = kappa*diff[1]*diff[1]*diff2[1]*diff2[2]*rinv2 - phi*diff[1]*diff[1]*diff[1]*diff[2]*rinv3;
    elast[1][4] = kappa*diff[1]*diff[1]*diff2[0]*diff2[2]*rinv2 - phi*diff[1]*diff[1]*diff[0]*diff[2]*rinv3;
    elast[1][5] = kappa*diff[1]*diff[1]*diff2[0]*diff2[1]*rinv2 - phi*diff[1]*diff[1]*diff[0]*diff[1]*rinv3;
    elast[2][2] = kappa*diff[2]*diff[2]*diff2[2]*diff2[2]*rinv2 - phi*diff[2]*diff[2]*diff[2]*diff[2]*rinv3;
    elast[2][3] = kappa*diff[2]*diff[2]*diff2[1]*diff2[2]*rinv2 - phi*diff[2]*diff[2]*diff[1]*diff[2]*rinv3;
    elast[2][4] = kappa*diff[2]*diff[2]*diff2[0]*diff2[2]*rinv2 - phi*diff[2]*diff[2]*diff[0]*diff[2]*rinv3;
    elast[2][5] = kappa*diff[2]*diff[2]*diff2[0]*diff2[1]*rinv2 - phi*diff[2]*diff[2]*diff[0]*diff[1]*rinv3;
    elast[3][3] = kappa*diff[1]*diff[2]*diff2[1]*diff2[2]*rinv2 - phi*diff[1]*diff[2]*diff[1]*diff[2]*rinv3;
    elast[3][4] = kappa*diff[1]*diff[2]*diff2[0]*diff2[2]*rinv2 - phi*diff[1]*diff[2]*diff[0]*diff[2]*rinv3;
    elast[3][5] = kappa*diff[1]*diff[2]*diff2[0]*diff2[1]*rinv2 - phi*diff[1]*diff[2]*diff[0]*diff[1]*rinv3;
    elast[4][4] = kappa*diff[0]*diff[2]*diff2[0]*diff2[2]*rinv2 - phi*diff[0]*diff[2]*diff[0]*diff[2]*rinv3;
    elast[4][5] = kappa*diff[0]*diff[2]*diff2[0]*diff2[1]*rinv2 - phi*diff[0]*diff[2]*diff[0]*diff[1]*rinv3;
    elast[5][5] = kappa*diff[0]*diff[1]*diff2[0]*diff2[1]*rinv2 - phi*diff[0]*diff[1]*diff[0]*diff[1]*rinv3;

    // calculate the grid coordinates (no pbc) for the extreme points
    iarray x, xn, i2, c;
    darray t,t_c1,t_c2,d_cgrid;
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
    t[0] = (c[0] == 0) ? 1.1 : t_c1[0]-xn[0]*t_c2[0];
    t[1] = (c[1] == 0) ? 1.1 : t_c1[1]-xn[1]*t_c2[1];
    t[2] = (c[2] == 0) ? 1.1 : t_c1[2]-xn[2]*t_c2[2];

    //------------------------------------------------------------------------------------
    // Distribute the stress

    // now the position/spatial constants
    const double C = 0.125*this->m_invgridsp*this->m_invgridsp;
    const double axy = diff[0]*diff[1];
    const double axz = diff[0]*diff[2];
    const double ayz = diff[1]*diff[2];
    const double axyz = diff[0]*ayz; 
    
    // track previous time of crossing
    double oldt = 0.0; 
    
    const int iterations = c[0]*(i2[0]-x[0]) + c[1]*(i2[1]-x[1]) + c[2]*(i2[2]-x[2]);
    for (int count = 0; count <= iterations; ++count)
    {
        // figure out index
        const int cmp0x = ((t[0]<t[1]+mds_eps) + (t[0]<t[2]+mds_eps))/2;
        const int cmp1x = ((t[1]<t[0]+mds_eps) + (t[1]<t[2]+mds_eps))/2;
        const int cmp2x = ((t[2]<t[0]+mds_eps) + (t[2]<t[1]+mds_eps))/2;
        const int iX = (1-cmp0x)*(cmp1x+2*(1-cmp1x)*cmp2x);
        const double newt = (iterations == count) ? 1.0 : t[iX];

        // work out the parametric time constants
        const double t12 = oldt*oldt;
        const double t22 = newt*newt;
        const double dt1 = newt - oldt;
        const double dt2 = t22 - t12;
        const double dt3 = 4.0*(t22*newt - t12*oldt)/3.0;
        const double dt4 = t22*t22 - t12*t12;

        // additional constants
        const double bxy = d_cgrid[0]*d_cgrid[1];
        const double bxz = d_cgrid[0]*d_cgrid[2];
        const double byz = d_cgrid[1]*d_cgrid[2];
        const double bxyz = d_cgrid[0]*byz;
    
        const int iip1 = ((x[0] + 1 + this->m_nxyz[0]) % this->m_nxyz[0])*this->m_nxyz[1]*this->m_nxyz[2];
        const int jjp1 = ((x[1] + 1 + this->m_nxyz[1]) % this->m_nxyz[1])*this->m_nxyz[2];
        const int kkp1 = ((x[2] + 1 + this->m_nxyz[2]) % this->m_nxyz[2]);
        const int iim1 = ((x[0] + this->m_nxyz[0]) % this->m_nxyz[0])*this->m_nxyz[1]*this->m_nxyz[2];
        const int jjm1 = ((x[1] + this->m_nxyz[1]) % this->m_nxyz[1])*this->m_nxyz[2];
        const int kkm1 = ((x[2] + this->m_nxyz[2]) % this->m_nxyz[2]);

        // the composite constants in terms of i, j, k
        const double D[8] = {
            8.0*bxyz*dt1 + 4.0*(diff[0]*byz+diff[1]*bxz+diff[2]*bxy)*dt2
                + 2.0*(d_cgrid[0]*ayz+d_cgrid[1]*axz+d_cgrid[2]*axy)*dt3 + 2.0*axyz*dt4,
            this->m_gridsp[0]*(4.0*byz*dt1 + 2.0*(diff[1]*d_cgrid[2]+diff[2]*d_cgrid[1])*dt2 + ayz*dt3),
            this->m_gridsp[1]*(4.0*bxz*dt1 + 2.0*(diff[0]*d_cgrid[2]+diff[2]*d_cgrid[0])*dt2 + axz*dt3),
            this->m_gridsp[2]*(4.0*bxy*dt1 + 2.0*(diff[0]*d_cgrid[1]+diff[1]*d_cgrid[0])*dt2 + axy*dt3),
            this->m_gridsp[3]*(2.0*d_cgrid[2]*dt1+diff[2]*dt2),
            this->m_gridsp[4]*(2.0*d_cgrid[1]*dt1+diff[1]*dt2),
            this->m_gridsp[5]*(2.0*d_cgrid[0]*dt1+diff[0]*dt2),
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
void StressGrid::DistributeKinetic(double mass, darray x, darray va, darray vb = NULL, int atomID = -1)
{
    dmatrix stress;

    // Spreads the velocity in one point
    iarray i1;
    darray xc,xd;
    int index, iip1, iim1, jjp1, jjm1, kkp1, kkm1;
    double factor,C;

    if ( !this->m_ierr )
    {
        // get the batch id
        int batch_id = this->m_thread_map[std::this_thread::get_id()];
            
        if (vb == NULL)
        {
            stress[0][0] = -mass*va[0]*va[0];
            stress[0][1] = -mass*va[0]*va[1];
            stress[0][2] = -mass*va[0]*va[2];
            stress[1][0] = -mass*va[1]*va[0];
            stress[1][1] = -mass*va[1]*va[1];
            stress[1][2] = -mass*va[1]*va[2];
            stress[2][0] = -mass*va[2]*va[0];
            stress[2][1] = -mass*va[2]*va[1];
            stress[2][2] = -mass*va[2]*va[2];
        }
        else
        {
            stress[0][0] = -mass*(va[0]*va[0] + vb[0]*vb[0])/2;
            stress[0][1] = -mass*(va[0]*va[1] + vb[0]*vb[1])/2;
            stress[0][2] = -mass*(va[0]*va[2] + vb[0]*vb[2])/2;
            stress[1][0] = -mass*(va[1]*va[0] + vb[1]*vb[0])/2;
            stress[1][1] = -mass*(va[1]*va[1] + vb[1]*vb[1])/2;
            stress[1][2] = -mass*(va[1]*va[2] + vb[1]*vb[2])/2;
            stress[2][0] = -mass*(va[2]*va[0] + vb[2]*vb[0])/2;
            stress[2][1] = -mass*(va[2]*va[1] + vb[2]*vb[1])/2;
            stress[2][2] = -mass*(va[2]*va[2] + vb[2]*vb[2])/2;
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
            xc[0] = x[0]-this->m_gridsp[0]*i1[0];
            xc[1] = x[1]-this->m_gridsp[1]*i1[1];
            xc[2] = x[2]-this->m_gridsp[2]*i1[2];
            xd[0] = xc[0]-this->m_gridsp[0];
            xd[1] = xc[1]-this->m_gridsp[1];
            xd[2] = xc[2]-this->m_gridsp[2];
            
            // select grid based on batch index
            dmatrix * grid = this->p_current_grid+batch_id*this->m_ncells;

            // Spread it
            scalesummatrix( C*xc[0]*xc[1]*xc[2],stress,grid[iip1+jjp1+kkp1]);
            scalesummatrix(-C*xc[0]*xc[1]*xd[2],stress,grid[iip1+jjp1+kkm1]);
            scalesummatrix(-C*xc[0]*xd[1]*xc[2],stress,grid[iip1+jjm1+kkp1]);
            scalesummatrix( C*xc[0]*xd[1]*xd[2],stress,grid[iip1+jjm1+kkm1]);
            scalesummatrix(-C*xd[0]*xc[1]*xc[2],stress,grid[iim1+jjp1+kkp1]);
            scalesummatrix( C*xd[0]*xc[1]*xd[2],stress,grid[iim1+jjp1+kkm1]);
            scalesummatrix( C*xd[0]*xd[1]*xc[2],stress,grid[iim1+jjm1+kkp1]);
            scalesummatrix(-C*xd[0]*xd[1]*xd[2],stress,grid[iim1+jjm1+kkm1]);
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
                summatrix (this->p_current_grid[atomID],stress,this->p_current_grid[atomID]);
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
void StressGrid::DistributeKineticElast(double mass, darray x, darray va, darray vb = NULL)
{
    dmatrix6 elast;

    // Spreads the velocity in one point
    iarray i1;
    darray xc,xd,pa,pb;
    int index, iip1, iim1, jjp1, jjm1, kkp1, kkm1;
    double factor,C;
    if ( !this->m_ierr )
    {
        // get the batch id
        int batch_id = this->m_thread_map[std::this_thread::get_id()];
        // select grid based on batch index
        dmatrix6 * grid = this->p_current_grid_elkin+batch_id*this->m_ncells;

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
        if (vb == NULL)
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
            elast[0][0] = mass*4.0*va[0]*va[0]; //c_xxxx = c_0000
            elast[0][1] = 0.0; //c_xxyy = c_0011
            elast[0][2] = 0.0; //c_xxzz = c_0022
            elast[0][3] = 0.0; //c_xxyz = c_0012
            elast[0][4] = mass*(va[0]*va[2] + va[0]*va[2]); //c_xxxz = c_0002
            elast[0][5] = mass*(va[0]*va[1] + va[0]*va[1]); //c_xxxy = c_0001
            elast[1][1] = mass*4.0*va[1]*va[1]; //c_yyyy = c_1111
            elast[1][2] = 0.0; //c_yyzz = c_1122
            elast[1][3] = mass*(va[1]*va[2] + va[1]*va[2]); //c_yyyz = c_1112
            elast[1][4] = 0.0; //c_yyxz = c_1102
            elast[1][5] = mass*(va[1]*va[0] + va[1]*va[0]); //c_yyxy = c_1101
            elast[2][2] = mass*4.0*va[2]*va[2]; //c_zzzz = c_2222
            elast[2][3] = mass*(va[2]*va[1] + va[2]*va[1]); //c_zzyz = c_2212
            elast[2][4] = mass*(va[2]*va[0] + va[2]*va[0]); //c_zzxz = c_2202
            elast[2][5] = 0.0; //c_zzxy = c_2201
            elast[3][3] = mass*(va[1]*va[1] + va[2]*va[2]); //c_yzyz = c_1212
            elast[3][4] = mass*va[1]*va[0]; //c_yzxz = c_1202
            elast[3][5] = mass*va[2]*va[0]; //c_yzxy = c_1201
            elast[4][4] = mass*(va[0]*va[0] + va[2]*va[2]); //c_xzxz = c_0202
            elast[4][5] = mass*va[2]*va[1]; //c_xzxy = c_0201
            elast[5][5] = mass*(va[0]*va[0] + va[1]*va[1]); //c_xyxy = c_0101
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
            elast[0][0] = mass*4.0*(va[0]*va[0] + vb[0]*vb[0])/2.0; //c_xxxx = c_0000
            elast[0][1] = 0.0; //c_xxyy = c_0011
            elast[0][2] = 0.0; //c_xxzz = c_0022
            elast[0][3] = 0.0; //c_xxyz = c_0012
            elast[0][4] = mass*(va[0]*va[2] + va[0]*va[2] + vb[0]*vb[2] + vb[0]*vb[2])/2.0; //c_xxxz = c_0002
            elast[0][5] = mass*(va[0]*va[1] + va[0]*va[1] + vb[0]*vb[1] + vb[0]*vb[1])/2.0; //c_xxxy = c_0001
            elast[1][1] = mass*4.0*(va[1]*va[1] + vb[1]*vb[1])/2.0; //c_yyyy = c_1111
            elast[1][2] = 0.0; //c_yyzz = c_1122
            elast[1][3] = mass*(va[1]*va[2] + va[1]*va[2] + vb[1]*vb[2] + vb[1]*vb[2])/2.0; //c_yyyz = c_1112
            elast[1][4] = 0.0; //c_yyxz = c_1102
            elast[1][5] = mass*(va[1]*va[0] + va[1]*va[0] + vb[1]*vb[0] + vb[1]*vb[0])/2.0; //c_yyxy = c_1101
            elast[2][2] = mass*4.0*(va[2]*va[2] + vb[2]*vb[2])/2.0; //c_zzzz = c_2222
            elast[2][3] = mass*(va[2]*va[1] + va[2]*va[1] + vb[2]*vb[1] + vb[2]*vb[1])/2.0; //c_zzyz = c_2212
            elast[2][4] = mass*(va[2]*va[0] + va[2]*va[0] + vb[2]*vb[0] + vb[2]*vb[0])/2.0; //c_zzxz = c_2202
            elast[2][5] = 0.0; //c_zzxy = c_2201
            elast[3][3] = mass*(va[1]*va[1] + va[2]*va[2] + vb[1]*vb[1] + vb[2]*vb[2])/2.0; //c_yzyz = c_1212
            elast[3][4] = mass*(va[1]*va[0] + vb[1]*vb[0])/2.0; //c_yzxz = c_1202
            elast[3][5] = mass*(va[2]*va[0] + vb[2]*vb[0])/2.0; //c_yzxy = c_1201
            elast[4][4] = mass*(va[0]*va[0] + va[2]*va[2] + vb[0]*vb[0] + vb[2]*vb[2])/2.0; //c_xzxz = c_0202
            elast[4][5] = mass*(va[2]*va[1] + vb[2]*vb[1])/2.0; //c_xzxy = c_0201
            elast[5][5] = mass*(va[0]*va[0] + va[1]*va[1] + vb[0]*vb[0] + vb[1]*vb[1])/2.0; //c_xyxy = c_0101
        }

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
void StressGrid::DistributeCharge ( darray x, double charge )
{
    // Spreads the velocity in one point
    iarray i1;
    darray xc,xd;
    int index, iip1, iim1, jjp1, jjm1, kkp1, kkm1;
    double factor,C;

    if ( !this->m_ierr )
    {
        // get the batch id
        int batch_id = this->m_thread_map[std::this_thread::get_id()];
        
        // select grid based on batch index
        double * gridc = this->p_current_gridc+batch_id*this->m_ncellsc;
            
        // Get the coordinates of the point in the grid
        i1[0] = this->m_nxyzc[0] * x[0] * this->m_invbox[0][0] - (x[0] < 0.0);
        i1[1] = this->m_nxyzc[1] * x[1] * this->m_invbox[1][1] - (x[1] < 0.0);
        i1[2] = this->m_nxyzc[2] * x[2] * this->m_invbox[2][2] - (x[2] < 0.0);
        
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
void StressGrid::DistributeN3( darray Ra, darray Rb, darray Rc, darray Fa, darray Fb, darray Fc, int batch_id )
{
    //Counter
    int i;

    //************************************************************************************
    // UNIT vectors between particles
    darray AB, AC, BC;

    // Distances
    double normAB,normAC,normBC;

    // (Covariant) Central Force decomposition
    double lab, lac, lbc;

    darray Fij;
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
        diffarray(Rb, Ra, AB, this->m_box, this->m_periodic);
        diffarray(Rc, Ra, AC, this->m_box, this->m_periodic);
        diffarray(Rc, Rb, BC, this->m_box, this->m_periodic);

        normAB=normarray(AB);
        normAC=normarray(AC);
        normBC=normarray(BC);

        for ( i = 0; i < nCol*nRow; i++ )
        {
            M[i] = 0.0;
        }

        //Force on particle 1:
        M[nRow*0+0] = AB[0]; M[nRow*1+0] = AC[0];
        M[nRow*0+1] = AB[1]; M[nRow*1+1] = AC[1];
        M[nRow*0+2] = AB[2]; M[nRow*1+2] = AC[2];
        b[0] = Fa[0]; b[1] = Fa[1]; b[2] = Fa[2];

        //Force on particle 2:
        M[nRow*0+3] = -AB[0]; M[nRow*2+3] = BC[0];
        M[nRow*0+4] = -AB[1]; M[nRow*2+4] = BC[1];
        M[nRow*0+5] = -AB[2]; M[nRow*2+5] = BC[2];
        b[3] = Fb[0]; b[4] = Fb[1]; b[5] = Fb[2];

        //Force on particle 3:
        M[nRow*1+6] = -AC[0]; M[nRow*2+6] = -BC[0];
        M[nRow*1+7] = -AC[1]; M[nRow*2+7] = -BC[1];
        M[nRow*1+8] = -AC[2]; M[nRow*2+8] = -BC[2];
        b[6] = Fc[0]; b[7] = Fc[1]; b[8] = Fc[2];
       
        if ( this->h_lapack[batch_id]->SolveMinNorm(nRow, nCol, M, b) )
        {
            std::cout << "ERROR::StressGrid: LAPACK solver fails\n";
            this->m_ierr = 10;
        }

        // Sum the 3 contributions to the stress
        lab = b[0];
        lac = b[1];
        lbc = b[2];

        Fij[0] = lab * AB[0]; Fij[1] = lab * AB[1]; Fij[2] = lab * AB[2];
        this->DistributePairInteraction(Ra, Rb, Fij, batch_id);
        Fij[0] = lac * AC[0]; Fij[1] = lac * AC[1]; Fij[2] = lac * AC[2];
        this->DistributePairInteraction(Ra, Rc, Fij, batch_id);
        Fij[0] = lbc * BC[0]; Fij[1] = lbc * BC[1]; Fij[2] = lbc * BC[2];
        this->DistributePairInteraction(Rb, Rc, Fij, batch_id);

    }
    else if (this->m_fdecomp == mds_gld)
    {
        Fij[0] = (Fa[0]-Fb[0])/3.0; Fij[1] = (Fa[1]-Fb[1])/3.0; Fij[2] = (Fa[2]-Fb[2])/3.0;
        this->DistributePairInteraction(Ra, Rb, Fij, batch_id);
        Fij[0] = (Fa[0]-Fc[0])/3.0; Fij[1] = (Fa[1]-Fc[1])/3.0; Fij[2] = (Fa[2]-Fc[2])/3.0;
        this->DistributePairInteraction(Ra, Rc, Fij, batch_id);
        Fij[0] = (Fb[0]-Fc[0])/3.0; Fij[1] = (Fb[1]-Fc[1])/3.0; Fij[2] = (Fb[2]-Fc[2])/3.0;
        this->DistributePairInteraction(Rb, Rc, Fij, batch_id);
    }
    
}

// Decompose Settle
void StressGrid::DistributeSettle( darray Ra, darray Rb, darray Rc, darray Fa, darray Fb, darray Fc, int batch_id )
{
    //Counter
    int i;

    //************************************************************************************
    // UNIT vectors between particles
    darray AB, AC, BC;

    // Distances
    double normAB,normAC,normBC;

    // (Covariant) Central Force decomposition
    double lab, lac, lbc;

    darray Fij;
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
	
	double phi_ab, phi_ac, phi_bc;
	double kappa_ab, kappa_ac, kappa_bc;

    if (this->m_fdecomp == mds_ccfd || this->m_fdecomp == mds_ncfd || this->m_fdecomp == mds_gld )
    {
        diffarray(Rb, Ra, AB, this->m_box, this->m_periodic);
        diffarray(Rc, Ra, AC, this->m_box, this->m_periodic);
        diffarray(Rc, Rb, BC, this->m_box, this->m_periodic);

        normAB=normarray(AB);
        normAC=normarray(AC);
        normBC=normarray(BC);

        for ( i = 0; i < nCol*nRow; i++ )
        {
            M[i] = 0.0;
        }

        //Force on particle 1:
        M[nRow*0+0] = AB[0]; M[nRow*1+0] = AC[0];
        M[nRow*0+1] = AB[1]; M[nRow*1+1] = AC[1];
        M[nRow*0+2] = AB[2]; M[nRow*1+2] = AC[2];
        b[0] = Fa[0]; b[1] = Fa[1]; b[2] = Fa[2];

        //Force on particle 2:
        M[nRow*0+3] = -AB[0]; M[nRow*2+3] = BC[0];
        M[nRow*0+4] = -AB[1]; M[nRow*2+4] = BC[1];
        M[nRow*0+5] = -AB[2]; M[nRow*2+5] = BC[2];
        b[3] = Fb[0]; b[4] = Fb[1]; b[5] = Fb[2];

        //Force on particle 3:
        M[nRow*1+6] = -AC[0]; M[nRow*2+6] = -BC[0];
        M[nRow*1+7] = -AC[1]; M[nRow*2+7] = -BC[1];
        M[nRow*1+8] = -AC[2]; M[nRow*2+8] = -BC[2];
        b[6] = Fc[0]; b[7] = Fc[1]; b[8] = Fc[2];

        if ( this->h_lapack[batch_id]->SolveMinNorm(nRow, nCol, M, b) )
        {
            std::cout << "ERROR::StressGrid: LAPACK solver fails\n";
            this->m_ierr = 10;
        }

        // Sum the 3 contributions to the stress
        lab = b[0];
        lac = b[1];
        lbc = b[2];

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

		kappa_ab = 0.0; //lab;
		kappa_ac = 0.0; //lac;
		kappa_bc = 0.0; //lbc;

		//Calculate Elasticity
		//DistributePairElast(darray xi, darray xj, darray xk, darray xl, double phi, double kappa)
		this->DistributePairElast(Ra, Rb, Ra, Rb, phi_ab, kappa_ab);
		this->DistributePairElast(Ra, Rc, Ra, Rc, phi_ac, kappa_ac);
		this->DistributePairElast(Rb, Rc, Rb, Rc, phi_bc, kappa_bc);

    }
}

// Decompose 4-body potentials (dihedrals)
void StressGrid::DistributeN4( darray Ra, darray Rb, darray Rc, darray Rd, darray Fa, darray Fb, darray Fc, darray Fd, int batch_id )
{
    //Counter
    int i;

    //************************************************************************************
    // UNIT vectors between particles
    darray AB, AC, AD, BC, BD, CD;

    // Distances
    double normAB,normAC,normAD,normBC,normBD,normCD;

    // (Covariant) Central Force decomposition
    double lab, lac, lad, lbc, lbd, lcd;

    darray Fij;
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
        diffarray(Rb, Ra, AB, this->m_box, this->m_periodic);
        diffarray(Rc, Ra, AC, this->m_box, this->m_periodic);
        diffarray(Rd, Ra, AD, this->m_box, this->m_periodic);
        diffarray(Rc, Rb, BC, this->m_box, this->m_periodic);
        diffarray(Rd, Rb, BD, this->m_box, this->m_periodic);
        diffarray(Rd, Rc, CD, this->m_box, this->m_periodic);

        normAB=normarray(AB);
        normAC=normarray(AC);
        normAD=normarray(AD);
        normBC=normarray(BC);
        normBD=normarray(BD);
        normCD=normarray(CD);

        for ( i = 0; i < nCol*nRow; i++ )
        {
            M[i] = 0.0;
        }
        //Force on particle 1:
        M[nRow*0+0] = AB[0]; M[nRow*1+0] = AC[0]; M[nRow*2+0] = AD[0];
        M[nRow*0+1] = AB[1]; M[nRow*1+1] = AC[1]; M[nRow*2+1] = AD[1];
        M[nRow*0+2] = AB[2]; M[nRow*1+2] = AC[2]; M[nRow*2+2] = AD[2];
        b[0] = Fa[0]; b[1] = Fa[1]; b[2] = Fa[2];

        //Force on particle 2:
        M[nRow*0+3] = -AB[0]; M[nRow*3+3] = BC[0]; M[nRow*4+3] = BD[0];
        M[nRow*0+4] = -AB[1]; M[nRow*3+4] = BC[1]; M[nRow*4+4] = BD[1];
        M[nRow*0+5] = -AB[2]; M[nRow*3+5] = BC[2]; M[nRow*4+5] = BD[2];
        b[3] = Fb[0]; b[4] = Fb[1]; b[5] = Fb[2];

        //Force on particle 3:
        M[nRow*1+6] = -AC[0]; M[nRow*3+6] = -BC[0]; M[nRow*5+6] = CD[0];
        M[nRow*1+7] = -AC[1]; M[nRow*3+7] = -BC[1]; M[nRow*5+7] = CD[1];
        M[nRow*1+8] = -AC[2]; M[nRow*3+8] = -BC[2]; M[nRow*5+8] = CD[2];
        b[6] = Fc[0]; b[7] = Fc[1]; b[8] = Fc[2];

        //Force on particle 4:
        M[nRow*2+9]  = -AD[0]; M[nRow*4+9] = -BD[0]; M[nRow*5+9] = -CD[0];
        M[nRow*2+10] = -AD[1]; M[nRow*4+10] = -BD[1]; M[nRow*5+10] = -CD[1];
        M[nRow*2+11] = -AD[2]; M[nRow*4+11] = -BD[2]; M[nRow*5+11] = -CD[2];
        b[9] = Fd[0]; b[10] = Fd[1]; b[11] = Fd[2];
        
        if ( this->h_lapack[batch_id]->SolveMinNorm(nRow, nCol, M, b) )
        {
            std::cout << "ERROR::StressGrid: LAPACK solver fails\n";
            this->m_ierr = 10;
        }

        // Sum the 6 contributions to the stress
        
        lab = b[0];
        lac = b[1];
        lad = b[2];
        lbc = b[3];
        lbd = b[4];
        lcd = b[5];

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
        Fij[0] = (Fa[0]-Fb[0])/4.0; Fij[1] = (Fa[1]-Fb[1])/4.0; Fij[2] = (Fa[2]-Fb[2])/4.0;
        this->DistributePairInteraction(Ra, Rb, Fij, batch_id);
        Fij[0] = (Fa[0]-Fc[0])/4.0; Fij[1] = (Fa[1]-Fc[1])/4.0; Fij[2] = (Fa[2]-Fc[2])/4.0;
        this->DistributePairInteraction(Ra, Rc, Fij, batch_id);
        Fij[0] = (Fa[0]-Fd[0])/4.0; Fij[1] = (Fa[1]-Fd[1])/4.0; Fij[2] = (Fa[2]-Fd[2])/4.0;
        this->DistributePairInteraction(Ra, Rd, Fij, batch_id);
        Fij[0] = (Fb[0]-Fc[0])/4.0; Fij[1] = (Fb[1]-Fc[1])/4.0; Fij[2] = (Fb[2]-Fc[2])/4.0;
        this->DistributePairInteraction(Rb, Rc, Fij, batch_id);
        Fij[0] = (Fb[0]-Fd[0])/4.0; Fij[1] = (Fb[1]-Fd[1])/4.0; Fij[2] = (Fb[2]-Fd[2])/4.0;
        this->DistributePairInteraction(Rb, Rd, Fij, batch_id);
        Fij[0] = (Fc[0]-Fd[0])/4.0; Fij[1] = (Fc[1]-Fd[1])/4.0; Fij[2] = (Fc[2]-Fd[2])/4.0;
        this->DistributePairInteraction(Rc, Rd, Fij, batch_id);
    }
    
}


// Decompose 5-body potentials (CMAP)
void StressGrid::DistributeN5(darray Ra, darray Rb, darray Rc, darray Rd, darray Re, darray Fa, darray Fb, darray Fc, darray Fd, darray Fe, int batch_id)
{
    //Counter
    int i;

    //************************************************************************************
    // UNIT vectors between particles
    darray AB, AC, AD, AE, BC, BD, BE, CD, CE, DE;

    // Distances
    double normAB,normAC,normAD,normAE,normBC,normBD,normBE,normCD,normCE,normDE;

    // (Covariant) Central Force decomposition
    double lab, lac, lad, lae, lbc, lbd, lbe, lcd, lce, lde;

    darray Fij;
    //************************************************************************************

    //Dimension and number of particles
    int nDim = 3;
    int nPart = 5;

    //Number of rows and columns
    int nRow = mds_nrow5, nCol = mds_ncol5, nRHS = 1;

    // Matrix of the system (15 equations x 10 unknowns)
    double M[mds_nrow5*mds_ncol5];
    // Vector, we want to solve M*x = b
    double b[mds_nrow5], s[mds_ncol5], CaleyMengerNormal[mds_ncol5];
    // Scalar product of the Normal and the initial CFD
    double prod;

    // If the force decomposition is cCFD or CFD
    if(this->m_fdecomp == mds_ccfd || this->m_fdecomp == mds_ncfd)
    {
        diffarray(Rb, Ra, AB, this->m_box, this->m_periodic);
        diffarray(Rc, Ra, AC, this->m_box, this->m_periodic);
        diffarray(Rd, Ra, AD, this->m_box, this->m_periodic);
        diffarray(Re, Ra, AE, this->m_box, this->m_periodic);
        diffarray(Rc, Rb, BC, this->m_box, this->m_periodic);
        diffarray(Rd, Rb, BD, this->m_box, this->m_periodic);
        diffarray(Re, Rb, BE, this->m_box, this->m_periodic);
        diffarray(Rd, Rc, CD, this->m_box, this->m_periodic);
        diffarray(Re, Rc, CE, this->m_box, this->m_periodic);
        diffarray(Re, Rd, DE, this->m_box, this->m_periodic);

        normAB=normarray(AB);
        normAC=normarray(AC);
        normAD=normarray(AD);
        normAE=normarray(AE);
        normBC=normarray(BC);
        normBD=normarray(BD);
        normBE=normarray(BE);
        normCD=normarray(CD);
        normCE=normarray(CE);
        normDE=normarray(DE);

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
        M[nRow*0+0] = AB[0]; M[nRow*1+0] = AC[0]; M[nRow*2+0] = AD[0]; M[nRow*3+0] = AE[0];
        M[nRow*0+1] = AB[1]; M[nRow*1+1] = AC[1]; M[nRow*2+1] = AD[1]; M[nRow*3+1] = AE[1];
        M[nRow*0+2] = AB[2]; M[nRow*1+2] = AC[2]; M[nRow*2+2] = AD[2]; M[nRow*3+2] = AE[2];
        b[0] = Fa[0]; b[1] = Fa[1]; b[2] = Fa[2];

        //Force on particle 2:
        M[nRow*0+3] = -AB[0]; M[nRow*4+3] = BC[0]; M[nRow*5+3] = BD[0]; M[nRow*6+3] = BE[0];
        M[nRow*0+4] = -AB[1]; M[nRow*4+4] = BC[1]; M[nRow*5+4] = BD[1]; M[nRow*6+4] = BE[1];
        M[nRow*0+5] = -AB[2]; M[nRow*4+5] = BC[2]; M[nRow*5+5] = BD[2]; M[nRow*6+5] = BE[2];
        b[3] = Fb[0]; b[4] = Fb[1]; b[5] = Fb[2];

        //Force on particle 3:
        M[nRow*1+6] = -AC[0]; M[nRow*4+6] = -BC[0]; M[nRow*7+6] = CD[0]; M[nRow*8+6] = CE[0];
        M[nRow*1+7] = -AC[1]; M[nRow*4+7] = -BC[1]; M[nRow*7+7] = CD[1]; M[nRow*8+7] = CE[1];
        M[nRow*1+8] = -AC[2]; M[nRow*4+8] = -BC[2]; M[nRow*7+8] = CD[2]; M[nRow*8+8] = CE[2];
        b[6] = Fc[0]; b[7] = Fc[1]; b[8] = Fc[2];

        //Force on particle 4:
        M[nRow*2+9]  = -AD[0]; M[nRow*5+9]  = -BD[0]; M[nRow*7+9]  = -CD[0]; M[nRow*9+9]  = DE[0];
        M[nRow*2+10] = -AD[1]; M[nRow*5+10] = -BD[1]; M[nRow*7+10] = -CD[1]; M[nRow*9+10] = DE[1];
        M[nRow*2+11] = -AD[2]; M[nRow*5+11] = -BD[2]; M[nRow*7+11] = -CD[2]; M[nRow*9+11] = DE[2];
        b[9] = Fd[0]; b[10] = Fd[1]; b[11] = Fd[2];

        //Force on particle 5:
        M[nRow*3+12] = -AE[0]; M[nRow*6+12] = -BE[0]; M[nRow*8+12] = -CE[0]; M[nRow*9+12] = -DE[0];
        M[nRow*3+13] = -AE[1]; M[nRow*6+13] = -BE[1]; M[nRow*8+13] = -CE[1]; M[nRow*9+13] = -DE[1];
        M[nRow*3+14] = -AE[2]; M[nRow*6+14] = -BE[2]; M[nRow*8+14] = -CE[2]; M[nRow*9+14] = -DE[2];
        b[12] = Fe[0]; b[13] = Fe[1]; b[14] = Fe[2];

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
            prod = 0.0;
            for ( i = 0; i < nCol; i++ )
            {
                prod += b[i]*CaleyMengerNormal[i];
            }

            for ( i = 0; i < nCol; i++ )
            {
                b[i] = b[i] - prod * CaleyMengerNormal[i];
            }
        }

        // Sum the 10 contributions to the stress
        lab = b[0];
        lac = b[1];
        lad = b[2];
        lae = b[3];
        lbc = b[4];
        lbd = b[5];
        lbe = b[6];
        lcd = b[7];
        lce = b[8];
        lde = b[9];

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
        Fij[0] = (Fa[0]-Fb[0])/5.0; Fij[1] = (Fa[1]-Fb[1])/5.0; Fij[2] = (Fa[2]-Fb[2])/5.0;
        this->DistributePairInteraction(Ra, Rb, Fij, batch_id);
        Fij[0] = (Fa[0]-Fc[0])/5.0; Fij[1] = (Fa[1]-Fc[1])/5.0; Fij[2] = (Fa[2]-Fc[2])/5.0;
        this->DistributePairInteraction(Ra, Rc, Fij, batch_id);
        Fij[0] = (Fa[0]-Fd[0])/5.0; Fij[1] = (Fa[1]-Fd[1])/5.0; Fij[2] = (Fa[2]-Fd[2])/5.0;
        this->DistributePairInteraction(Ra, Rd, Fij, batch_id);
        Fij[0] = (Fa[0]-Fe[0])/5.0; Fij[1] = (Fa[1]-Fe[1])/5.0; Fij[2] = (Fa[2]-Fe[2])/5.0;
        this->DistributePairInteraction(Ra, Re, Fij, batch_id);
        Fij[0] = (Fb[0]-Fc[0])/5.0; Fij[1] = (Fb[1]-Fc[1])/5.0; Fij[2] = (Fb[2]-Fc[2])/5.0;
        this->DistributePairInteraction(Rb, Rc, Fij, batch_id);
        Fij[0] = (Fb[0]-Fd[0])/5.0; Fij[1] = (Fb[1]-Fd[1])/5.0; Fij[2] = (Fb[2]-Fd[2])/5.0;
        this->DistributePairInteraction(Rb, Rd, Fij, batch_id);
        Fij[0] = (Fb[0]-Fe[0])/5.0; Fij[1] = (Fb[1]-Fe[1])/5.0; Fij[2] = (Fb[2]-Fe[2])/5.0;
        this->DistributePairInteraction(Rb, Re, Fij, batch_id);
        Fij[0] = (Fc[0]-Fd[0])/5.0; Fij[1] = (Fc[1]-Fd[1])/5.0; Fij[2] = (Fc[2]-Fd[2])/5.0;
        this->DistributePairInteraction(Rc, Rd, Fij, batch_id);
        Fij[0] = (Fc[0]-Fe[0])/5.0; Fij[1] = (Fc[1]-Fe[1])/5.0; Fij[2] = (Fc[2]-Fe[2])/5.0;
        this->DistributePairInteraction(Rc, Re, Fij, batch_id);
        Fij[0] = (Fd[0]-Fe[0])/5.0; Fij[1] = (Fd[1]-Fe[1])/5.0; Fij[2] = (Fd[2]-Fe[2])/5.0;
        this->DistributePairInteraction(Rd, Re, Fij, batch_id);
    }
}

// General function to decompose N-body potentials (it can be used to compute higher order terms coming from EAM for instance)
void StressGrid::DistributeNBody ( int nPart, darray *R, darray *F, bool distribute_stress, int batch_id)
{
    std::lock_guard<std::mutex> lock(this->m_mutex_state);

    int i,j,k, iD, jD, n;
    darray F_ij_temp;

    // grow the temp state as needed
    if (nPart > this->m_maxpart)
    {
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
        this->p_bvec  = new double [maxcols];
        this->p_Rij  = new darray [nPart*nPart];
        this->p_Fij  = new darray [nPart*nPart];
        this->p_Uij  = new darray [maxcols];
        
        this->m_maxpart = nPart;
    }

    // zero the pairwise force and pairwise position arrays
    for ( i = 0; i < nPart*nPart; ++i)
    {
        this->p_Rij[i][0] = this->p_Rij[i][1] = this->p_Rij[i][2] = 0.0;
        this->p_Fij[i][0] = this->p_Fij[i][1] = this->p_Fij[i][2] = 0.0;
    }

    n = 0;
    for ( i = 0; i < nPart; i++ )
    {
        for ( j = i+1; j < nPart; j++ )
        {
            diffarray(R[j], R[i], this->p_Uij[n], this->m_box, this->m_periodic);
            copyarray(this->p_Uij[n], this->p_Rij[i*nPart+j]);
            scalearray(this->p_Uij[n], -1.0, this->p_Rij[j*nPart+i]);
            scalearray(this->p_Uij[n],1.0/normarray(this->p_Uij[n]),this->p_Uij[n]);
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

            copyarray(F[i], &this->p_bvec[iD]);
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
                scalearray(this->p_Uij[n], this->p_bvec[n], F_ij_temp);
                copyarray(F_ij_temp, this->p_Fij[i*nPart+j]);
                scalearray(F_ij_temp, -1.0, this->p_Fij[j*nPart+i]);

                if (distribute_stress)
                    this->DistributePairInteraction(R[i], R[j], F_ij_temp, batch_id);

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
                diffarray(F[i], F[j], F_ij_temp );
                scalearray(F_ij_temp, 1.0/static_cast<double>(nPart), F_ij_temp);
                copyarray(F_ij_temp, this->p_Fij[i*nPart+j]);
                scalearray(F_ij_temp, -1.0, this->p_Fij[j*nPart+i]);

                if (distribute_stress)
                    this->DistributePairInteraction(R[i], R[j], F_ij_temp, batch_id);
                n++;
            }
        }
    }
}

//Auxilery Methods to Calculate the Phi and Kappa Terms for the Born Term
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
double StressGrid::CalcCosine(double ab, double bg, double ag) {
	double costheta, numer, denom;

	numer = (ab * ab) + (bg * bg) - (ag * ag);
	denom = 2 * ab * bg;
	costheta = numer / denom;
	return costheta;
}

double StressGrid::CalcTheta(double costheta) {
	return acos(costheta);
}

//Derivative of Cosine Function (creates derivative vector)
void StressGrid::ThreeBodyCosineD(double ab, double bg, double ag, darray &d_cos_array) {
	double numer, denom;


	//dab of costheta
	numer = (ab * ab) - (bg * bg) + (ag * ag);
	denom = 2 * ab * ab * bg;

	d_cos_array[0] = numer / denom;

	//dbg of costheta
	numer = -(ab * ab) + (bg * bg) + (ag * ag);
	denom = 2 * ag * bg * bg;

	d_cos_array[1] = numer / denom;

	//dag of costheta
	numer = -ag;
	denom = ab * bg;

	d_cos_array[2] = numer / denom;
}

//Second Derivative of Cosine Function (Creates derivative matrix)
void StressGrid::ThreeBodyCosineD2(double ab, double bg, double ag, dmatrix &d2_cos_array) {
	double numer;
	double denom;

	//d00 of costheta
	numer = (bg * bg) - (ag * ag);
	denom = (ab * ab * ab * bg);

	d2_cos_array[0][0] = numer / denom;

	// d01 and d10 of costheta

	numer = -((ab * ab) + (bg * bg) + (ag * ag));
	denom = 2 * (ab * ab) * (bg * bg);

	d2_cos_array[0][1] = numer / denom;
	d2_cos_array[1][0] = d2_cos_array[0][1];

	// d02 and d20 of costheta

	numer = ag;
	denom = ab * ab * bg;

	d2_cos_array[0][2] = numer / denom;
	d2_cos_array[2][0] = d2_cos_array[0][2];

	// d11 of costheta
	numer = (ab * ab) - (ag * ag);
	denom = (ab * bg * bg * bg);

	d2_cos_array[1][1] = numer / denom;

	// d12 and d21 of costheta
	numer = ag;
	denom = ab * bg * bg;

	d2_cos_array[1][2] = numer / denom;
	d2_cos_array[2][1] = d2_cos_array[1][2];

	// d22 of costheta
	numer = -1;
	denom = ab * bg;

	d2_cos_array[2][2] = numer / denom;
}

//First Derivative of Theta Function (Creates 1st derivative vector)
//Need Cosine Theta
//Need Derivative of Cosine Vector
void StressGrid::ThreeBodyThetaD(double costheta, darray &d_cos_array, darray &d_theta_array) {
	double scalefactor;

	scalefactor = -1 / sqrt(1 - (costheta * costheta));

	for (int i = 0; i < 3; i++) {
		d_theta_array[i] = scalefactor * d_cos_array[i];
	}
}

//Second Derivative of Theta Function (Creates 2nd Derivative Matix)
//Need Cosine Theta
//Need Derivative of Cosine Vector
//Need 2nd Derivative of Cosine Matrix
void StressGrid::ThreeBodyThetaD2(double costheta, darray d_cos_array, dmatrix &d2_cos_array, dmatrix &d2_theta_array) {
	double scalefactor;
	double sinthetasq;

	sinthetasq = 1 - (costheta * costheta);

	scalefactor = 1 / (sinthetasq * sqrt(sinthetasq));

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			d2_theta_array[i][j] = scalefactor * ((-sinthetasq) * d2_cos_array[i][j] - costheta * d_cos_array[i] * d_cos_array[j]);
		}
	}
}

//List of 2-body Potential Auxilery Functions
//Calculate The Phi and Kappa for Harmonic Potential -> Implimented
void StressGrid::HarmonicPhiKappa(double deltaR, double k, double& phi, double& kappa) {

	phi = k * deltaR;
	kappa = k;
}

//Calculate the Phi and Kappa for Buckingham Potential -> Not Implimented, Nonbonded?
void StressGrid::BuckinghamPhiKappa(double r, double a, double b, double c, double& phi, double& kappa) {

	double exponent = -b * r;
	double rinv = 1 / r;
	double rinvsq = rinv * rinv;
	double rinvsix = rinvsq * rinvsq * rinvsq;

	phi = -a * b * exp(exponent) + (6 * c) * rinvsix * rinv;
	kappa = a * b * b * exp(exponent) - (42 * c) * rinvsix * rinvsq;
}

//Calculate the Phi and Kappa for the Fourth Power Potential -> Implimented g96bond
void StressGrid::FourthPowerPhiKappa(double k4, double dist, double dist0, double& phi, double& kappa) {

	phi = k4 * dist * (dist * dist - dist0 * dist0);
	kappa = k4 * (3 * dist * dist - dist0 * dist0);
}

//Calculate the Phi and  Kappa for the Morse Potential -> Implimented
void StressGrid::MorsePhiKappa(double expadeltaR, double a, double d, double& phi, double& kappa) {
	//expadeltaR = e^(-a*(r-r0))
	double coeffexp = 2 * a * d * expadeltaR;

	phi = coeffexp * (1 - expadeltaR);
	kappa = coeffexp * a * (2 * expadeltaR - 1);
}

//Calculate the Phi and Kappa for the Cubic Bond Potential -> Implimented
void StressGrid::CubicBondPhiKappa(double deltaR, double k, double kcubic, double& phi, double& kappa) {

	phi = 2 * k * deltaR + 3 * k * kcubic * deltaR * deltaR;
	kappa = 2 * k + 6 * k * kcubic * deltaR;
}

//Calculate the Phi and Kappa for the FENE Potential -> Implimented
void StressGrid::FENEPhiKappa(double r, double k, double diffratio, double& phi, double& kappa) {
	//diffratio = 1 - (r/r0)^2

	phi = k * r / diffratio;
	kappa = k * (2 - diffratio) / diffratio;
}

//List of 3-body Potential Auxilery Functions
//Derivative Vector/Matrix form
// derivative identifier ab = 0, bg = 1, ag = 3

// ab|bg|ag = 0 | 1 | 2
/*
 * abab | bgab | agab	= 00 | 01 | 02
 * abbg | bgbg | agbg	= 10 | 11 | 12
 * abag | bgag | agag	= 20 | 21 | 22
 */

//Calculate the Phi and Kappa for the Harmonic Angle Potential -> Implimented
void StressGrid::HarmonicAnglePhiKappa(double ab, double bg, double ag, double costheta, double deltatheta, double k, darray &phi, dmatrix &kappa) {

	//double costh = this->CalcCosine(ab, bg, ag);
	darray d_cos_array;
	zeroarray(d_cos_array);
	dmatrix d2_cos_array;
	zeromatrix(d2_cos_array);
	darray d_theta_array;
	zeroarray(d_theta_array);
	dmatrix d2_theta_array;
	zeromatrix(d2_theta_array);
	this->ThreeBodyCosineD(ab, bg, ag, d_cos_array);
	this->ThreeBodyCosineD2(ab, bg, ag, d2_cos_array);
	this->ThreeBodyThetaD(costheta, d_cos_array, d_theta_array);
	this->ThreeBodyThetaD2(costheta, d_cos_array, d2_cos_array, d2_theta_array);


	//Calculate Phi Vector
	for (int i = 0; i < 3; i++) {
		phi[i] = k * deltatheta * d_theta_array[i];
	}

	//Calculate Kappa Matrix
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			kappa[i][j] = k * (d_theta_array[i] * d_theta_array[j] + deltatheta * d2_theta_array[i][j]);
		}
	}
}

//Calculate the Phi and Kappa for the Harmonic Cosine Potential -> Implimented Gromos96
void StressGrid::HarmonicCosPhiKappa(double ab, double bg, double ag, double deltacos, double k, darray &phi, dmatrix &kappa) {

	//double costh = this->CalcCosine(ab, bg, ag);
	darray d_cos_array;
	zeroarray(d_cos_array);
	dmatrix d2_cos_array;
	zeromatrix(d2_cos_array);
	this->ThreeBodyCosineD(ab, bg, ag, d_cos_array);
	this->ThreeBodyCosineD2(ab, bg, ag, d2_cos_array);


	//Calculate Phi Vector
	for (int i = 0; i < 3; i++) {
		phi[i] = k * deltacos * d_cos_array[i];
	}

	//Calculate Kappa Matrix
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			kappa[i][j] = k * (d_cos_array[i] * d_cos_array[j] + deltacos * d2_cos_array[i][j]);
		}
	}

}

//Calculate the Phi and Kappa for the Urey-Bradley Potential -> Not Implimented needs an overhaul
void StressGrid::UreyBradleyPhiKappa(double ab, double bg, double ag, double costheta, double deltaRag, double deltatheta, double ktheta, double kUB, darray &phi, dmatrix &kappa) {

	//double costh = this->CalcCosine(ab, bg, ag);
	darray d_cos_array;
	zeroarray(d_cos_array);
	dmatrix d2_cos_array;
	zeromatrix(d2_cos_array);
	darray d_theta_array;
	zeroarray(d_theta_array);
	dmatrix d2_theta_array;
	zeromatrix(d2_theta_array);
	this->ThreeBodyCosineD(ab, bg, ag, d_cos_array);
	this->ThreeBodyCosineD2(ab, bg, ag, d2_cos_array);
	this->ThreeBodyThetaD(costheta, d_cos_array, d_theta_array);
	this->ThreeBodyThetaD2(costheta, d_cos_array, d2_cos_array, d2_theta_array);

	//Calculate Phi ab(0) and bg(1) for phi vector
	for (int i = 0; i < 2; i++) {
		if (i == 2) {
			phi[i] = ktheta * deltatheta * d_theta_array[i] + kUB * deltaRag;
		}
		else {
			phi[i] = ktheta * deltatheta * d_theta_array[i];
		}
	}

	//Calculate Kappa Matrix (will calculate kappa[2][2] separately)
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (i == 2 && j == 2) {
				kappa[i][j] = ktheta * (d_theta_array[i] * d_theta_array[j] + deltatheta * d2_theta_array[i][j]) + kUB;
			}
			else {
				kappa[i][j] = ktheta * (d_theta_array[i] * d_theta_array[j] + deltatheta * d2_theta_array[i][j]);
			}
		}
	}
}

//Calculate the Phi and Kappa due to the Bond Bond Cross Potential -> Implimented Gromos96
void StressGrid::BondBondCrossPhiKappa(double k, double deltarab, double deltarbg, darray &phi, dmatrix &kappa) {

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
void StressGrid::BondAngleCrossPhiKappa(double k, double deltarab, double deltarbg, double deltarag, darray &phi, dmatrix &kappa) {

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
void StressGrid::QuarticAnglePhiKappa(double ab, double bg, double ag, double costheta, double deltatheta, double (&coeff)[5], darray &phi, dmatrix &kappa) {

	//double costh = this->CalcCosine(ab, bg, ag);
	darray d_cos_array;
	zeroarray(d_cos_array);
	dmatrix d2_cos_array;
	zeromatrix(d2_cos_array);
	darray d_theta_array;
	zeroarray(d_theta_array);
	dmatrix d2_theta_array;
	zeromatrix(d2_theta_array);
	this->ThreeBodyCosineD(ab, bg, ag, d_cos_array);
	this->ThreeBodyCosineD2(ab, bg, ag, d2_cos_array);
	this->ThreeBodyThetaD(costheta, d_cos_array, d_theta_array);
	this->ThreeBodyThetaD2(costheta, d_cos_array, d2_cos_array, d2_theta_array);


	double deltathetasq = deltatheta * deltatheta;
	double deltathetacube = deltathetasq * deltatheta;

	//Calculate the Finate Sums
	double phiconst = coeff[1] + 2 * coeff[2] * deltatheta + 3 * coeff[3] * deltathetasq + 4 * coeff[4] * deltathetacube;
	double kappaconst = 2 * coeff[2] + 6 * coeff[3] * deltatheta + 12 * coeff[4] * deltathetasq + 20;

	//Calculate Phi Vector
	for (int i = 0; i < 3; i++) {
		phi[i] = phiconst * d_theta_array[i];
	}

	//Calculate Kappa Matrix
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			kappa[i][j] = phiconst * d2_theta_array[i][j] + kappaconst * d_theta_array[i] * d_theta_array[j];
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
double StressGrid::CalcCosineDihedral(double ab, double ag, double ae, double bg, double be, double ge) {
	double numer, denom;

	numer = -(bg * bg * bg * bg) + (bg * bg) * ((ab * ab) - 2 * (ae * ae) + (ag * ag) + (be * be) + (ge * ge)) + ((ab * ab) - (ag * ag)) * (-(be * be) + (ge * ge));
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

void StressGrid::FourBodyThetaD(double costheta, darray6 &d_cos_di_array, darray6 &d_theta_di_array) {
	double scalefactor;

	scalefactor = -1 / sqrt(1 - (costheta * costheta));

	for (int i = 0; i < 6; i++) {
		d_theta_di_array[i] = scalefactor * d_cos_di_array[i];
	}
}

void StressGrid::FourBodyThetaD2(double costheta, darray6 &d_cos_di_array, dmatrix6 &d2_cos_di_array, dmatrix6 &d2_theta_di_array) {
	double scalefactor;
	double sinthetasq;

	sinthetasq = 1 - (costheta * costheta);
	scalefactor = 1 / (sinthetasq * sqrt(sinthetasq));

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
