/*=========================================================================

  Module    : MDStress
  File      : mds_stressgrid.h
  Authors   : A. Torres-Sanchez and J. M. Vanegas
  Modified  :
  Purpose   : Compute the local stress from MD trajectories
  Date      : 25/03/2015
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

#ifndef mds_stressgrid_h
#define mds_stressgrid_h

#include "mds_common.h"
#include "mds_defines.h"
#include "mds_basicops.h"
#include "mds_cmenger.h"
#include "mds_lapack.h"

class  mds::StressGrid
{
    public :
        const char *GetNameOfClass() const
        {   return "StressGrid";     }

        void Enable()
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            if (this->m_disable == true && this->m_initialized == true)
                this->m_disable = false;
        }
        void Disable()
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_disable = true;
        }

        bool CheckInit()
        {
            return this->m_initialized;
        }

        void SetThreadIDs(int thread_id, int max_threads)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_thread_map[std::this_thread::get_id()] = thread_id;
            this->m_max_threads = max_threads;
        }
        
        void SetNumberOfAtoms(int n)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_nAtoms = n;
        }
        int  GetNumberOfAtoms( )
        {
            return this->m_nAtoms;
        }
        
        void SetPeriodicBoundaries(bool x, bool y, bool z, bool enforce);
        void  GetPeriodicBoundaries(bool & x, bool & y, bool & z, bool enforce)
        { 
            x = this->m_periodic[0];
            y = this->m_periodic[1];
            z = this->m_periodic[2];
            enforce = this->m_periodic[3];
        }

        void SetNumberOfGridCellsX(int nx)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_nxyz[0] = nx;
        }
        void SetNumberOfGridCellsY(int ny)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_nxyz[1] = ny;
        }
        void SetNumberOfGridCellsZ(int nz)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_nxyz[2] = nz;
        }
        int  GetNumberOfGridCellsX( )
        {   return this->m_nxyz[0]; }
        int  GetNumberOfGridCellsY( )
        {   return this->m_nxyz[1]; }
        int  GetNumberOfGridCellsZ( )
        {   return this->m_nxyz[2]; }

        void SetSpacing(real_ext d)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_spacing = (real_int)d;
        }
        real_ext  GetSpacing( )
        {   return (real_ext)this->m_spacing; }
        real_ext  GetSpacingX( )
        {   return (real_ext)this->m_gridsp[0]; }
        real_ext  GetSpacingY( )
        {   return (real_ext)this->m_gridsp[1]; }
        real_ext  GetSpacingZ( )
        {   return (real_ext)this->m_gridsp[2]; }

        void SetForceDecomposition ( int fdecomp )
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            m_fdecomp = fdecomp;
        }
        int GetForceDecomposition ( void ) const
        {   return m_fdecomp;   }

        void SetStressType ( int spatatom )
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_spatatom = spatatom;
        }
        int GetStressType ( void ) const
        {   return this->m_spatatom;   }

        void SetContribType ( int contrib )
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_contrib = contrib;
        }
        int GetContribType ( void ) const
        {   return this->m_contrib;   }

        void SetMinDihAngle (real_ext mindihangle)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_mindihangle = (real_int)mindihangle;
        }
        real_ext GetMinDihAngle ( )
        {   return (real_ext)this->m_mindihangle;   }

        void ComputeNbodyPairForces ( int nAtoms, array3_ext *R, array3_ext *F, int *atomIDs );

        array3_int * GetNbodyDecompPairForces()
        {   return this->p_Fij;    }
        array3_int * GetNbodyDecompPairVectors()
        {   return this->p_Rij;    }

        void SetBox(matrix3_ext box, int epc)
        {   
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            if (epc != 0)
                this->m_pcoupl = true;
            for (int i = 0; i < mds_ndim; i++ )
                for (int j = 0; j < mds_ndim; j++)
                    this->m_box[i][j] = (real_int)box[i][j];
        }

        void SetMaxCluster ( int maxClust )
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_maxClust = maxClust;
        }
        int GetMaxCluster ( ) const
        {   return this->m_maxClust;  }
        
        void SetFileName ( const char *filename )
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_filename.assign(filename);
        }
        
        void SetTemperature(real_ext temperature)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_temperature = (real_int)temperature;
        }
        
        real_ext GetTemperature(real_ext temperature)
        {
            return (real_ext)this->m_temperature;
        }
        
        void EnableCuda()
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_cuda = true;
        }
        
        void DisableDispersionCorrection()
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_nodispcor = true;
        }
        
        /** Returns the kind of error produced by the class. Values:
         * 0: No error
         * 1: the stress type is not correct
         * 2: the number of cells at one side is negative
         * 3: the local spacing is too small
         * 4: the box is not set
         * 5: the force decomposition is incorrect
         * 6: the number of atoms is not set
         * 7: the contribution is not correct
         * 8: the filename is not set
         * 9: DistributeInteraction has been called with an incorrect number of atoms 
         * 10: Lapack failed 
         * 11: Voronoi cell count did not match natoms */
        int GetError ( )
        {   return this->m_ierr;  }
        
        void Init ( );
        
        void UpdateBoxSpacings ( matrix3_ext box );
        
        void SumGrid ( );
        
        void DispersionCorrection (real_ext shift);
        
        void Reset  ( );
        
        void Write  ( );
        void WriteForceDecomposition ( );
        
        void WriteAndReset ( )
        {   this->Write();  this->Reset();  }
        
        void SetVoronoiRadius(real_ext radius, int atomID);

        void AddVoronoiAtom(real_ext px, real_ext py, real_ext pz, int atomID, int moleID);

        void DistributeInteraction(
                int nAtoms,
                array3_ext *R,
                array3_ext *F,
                int *atomIDs );
        void DistributeElasticity(
                array3_ext xi_ext,
                array3_ext xj_ext,
                array3_ext xk_ext,
                array3_ext xl_ext,
                real_ext phi_ext,
                real_ext kappa_ext);
        void DistributeKinetic(
                real_ext mass,
                array3_ext x,
                array3_ext va,
                array3_ext vb,
                int atomID);
        void DistributeKineticElast(
                real_ext mass,
                array3_ext x,
                array3_ext va,
                array3_ext vb);
        
        
        StressGrid();
        ~StressGrid();
        
    private:
        int         m_nAtoms;         ///< Number of atoms
        iarray      m_nxyz;           ///< Number of grid cells in the x direction
        int         m_griddim;        ///< the type of grid
        long        m_ncells;         ///< Total number of cells in the calculation
        int         m_maxClust;
        real_int    m_spacing;        ///< spacing requested for the grid
        matrix3_int m_box;            ///< Actual box
        int         m_spatatom;       ///< enSpat or enAtom
        int         m_fdecomp;        ///< which force decomposition
        int         m_contrib;        ///< which contribution
        std::string m_filename;       ///< body of the filename where the stress is stored
        bool        m_nodispcor;      ///< disables dispersion correction if set to true
        bool        m_cuda;           ///< enables cuda
        bool        m_initialized;    ///< disable mdstress library operations
        bool        m_disable;        ///< disable mdstress library operations
        barray      m_periodic;       ///< mark dimensions as periodic
        real_int    m_mindihangle;
        int         m_maxpart;        ///< used to allocate Rij and Fij
        int         m_max_threads;    ///< number of threads to use
        real_int    m_temperature;
        bool        m_pcoupl;

        int      m_ierr;                     ///< error type: 0, 1, etc (See GetError() function)
        int      m_nframes;                  ///< Number of frames
        int      m_nreset;                   ///< Number of resets (for writing files)
        matrix3_int m_sumbox;                ///< Average box
        matrix3_int m_invbox;                ///< Inverse of the box
        real_int m_avg_boxvol;               ///< Average box volume
        real_int m_var_boxvol;               ///< Variance of box volume
        real_int m_gridsp[7];                ///< grid spacing
        real_int m_invgridsp;                ///< inverse of grid spacing
        Lapack **h_lapack;                   ///< mds_lapack: solves underdetermined/overdetermined systems of equations and projects solution onto shape space
        double  *p_Amat;                     ///< matrix for linear systems (for systems with more than 5 particles)
        double  *p_AmatT;                    ///< transpose of the matrix (used for projecting solution onto the shape space)
        double  *p_bvec;                     ///< vector for solving the linear system    
        array3_int *p_Rij;                   ///< distance vectors
        array3_int *p_Fij;                   ///< force vectors
        array3_int *p_Uij;                   ///< distance vectors
        matrix3_int *p_current_grid;         ///< Grid (either nx*ny*nz or nAtoms)
        matrix3_int *p_current_gridtot;      ///< Grid of size 1
        matrix6_int *p_current_grid_elborn;  ///< Grid (either nx*ny*nz or nAtoms)
        matrix6_int *p_current_grid_elkin;   ///< Grid (either nx*ny*nz or nAtoms)
        matrix3_int *p_sum_grid;             ///< Sum Grid
        matrix3_int *p_avg_grid;             ///< Sum Grid
        matrix3_int *p_avg_gridtot;          ///< Total Sum Grid
        matrix3_int *p_sum_grid_volcovar;    ///< Cov(sigma_local_ij, V)
        matrix3_int *p_sum_gridtot_volcovar; ///< Cov(sigma_total_ij, V)
        matrix6_int *p_sum_grid_elcovar;     ///< Elasticity Grid Covariance Term
        matrix6_int *p_sum_grid_elborn;      ///< Elasticity Grid Born Term
        matrix6_int *p_sum_grid_elkin;       ///< Elasticity Grid Kinetic Term
        real_int *p_sum_volume;              ///< Sum of volumes when using mds_atom
        real_int *p_radii;                   ///< the radius of an atomic site
        real_int *p_positions;               ///< the position of an atomic site
        int     *p_molecule_id;              ///< The molecule an atomic site belongs to
        
        std::map<std::thread::id,int> m_thread_map;
        std::mutex m_mutex_state;

        void Clear();

        int CheckSettings();

        void DistributeElasticity_internal1D(
                array3_int xi,
                array3_int xj,
                array3_int xk,
                array3_int xl,
                real_int phi,
                real_int kappa);
        void DistributeElasticity_internal3D(
                array3_int xi,
                array3_int xj,
                array3_int xk,
                array3_int xl,
                real_int phi,
                real_int kappa);
        
        void DistributePairInteraction(
                array3_int R1,
                array3_int R2,
                array3_int F,
                int batch_id);
        void DistributePairInteraction1D(
                array3_int R1,
                array3_int R2,
                array3_int F,
                int batch_id);
        void DistributePairInteraction3D(
                array3_int R1,
                array3_int R2,
                array3_int F,
                int batch_id);
        
        void DistributeN3(
                array3_int Ra,
                array3_int Rb,
                array3_int Rc,
                array3_int Fa,
                array3_int Fb,
                array3_int Fc,
                int batch_id);
        void DistributeSettle(
                array3_int Ra,
                array3_int Rb,
                array3_int Rc,
                array3_int Fa,
                array3_int Fb,
                array3_int Fc,
                int batch_id);
        void DistributeN4(
                array3_int Ra,
                array3_int Rb,
                array3_int Rc,
                array3_int Rd,
                array3_int Fa,
                array3_int Fb,
                array3_int Fc,
                array3_int Fd,
                int batch_id);
        void DistributeN5(
                array3_int Ra,
                array3_int Rb,
                array3_int Rc,
                array3_int Rd,
                array3_int Re,
                array3_int Fa,
                array3_int Fb,
                array3_int Fc,
                array3_int Fd,
                array3_int Fe,
                int batch_id);
        void DistributeNBody(
                int nPart,
                array3_ext *R,
                array3_ext *F,
                bool distritube_stress,
                int batch_id);
};
#endif // mds_stressgrid_h
