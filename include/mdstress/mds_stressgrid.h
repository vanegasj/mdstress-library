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

/** \class StressGrid
* \brief ROOT OF ALL EVIL. This class handles computations of the local stress from MD trajectories
* Once all settings are given, the function DistributeInteraction and DistributeKinetic can be used
* to distribute N-body interactions between particles and kinetic contributions respectively.
* 
* \b INPUT: 
* \param [in] nAtoms    Number of atoms
* \param [in] nx        Number of grid cells in the x direction
* \param [in] ny        Number of grid cells in the y direction
* \param [in] nz        Number of grid cells in the z direction
* \param [in] ncells    Total number of cells in the calculation
* \param [in] maxClust  Maximum cluster size (number of particles in a potential)
* \param [in] spacing   Spacing requested for the grid
* \param [in] box       Actual box
* \param [in] spatatom  enSpat or enAtom (kind of stress)
* \param [in] fdecomp   Force decomposition
* \param [in] contrib   Contribution
* \param [in] filename  Body of the filename where the stress is stored
* \n
* \b OUTPUT:
* \param [out] lapack       mds_lapack: solves underdetermined/overdetermined systems of equations and projects solution onto shape space
* \param [out] Amat         matrix for linear systems (for systems with more than 5 particles)
* \param [out] AmatT        transpose of the matrix (used for projecting solution onto the shape space)
* \param [out] bvec         vector for solving the linear system    
* \param [out] R_ij         distance vectors
* \param [out] F_ij         force vectors
* \param [out] ierr         error type: 0, 1, etc (See GetError() function)
* \param [out] nframes      Number of frames
* \param [out] nreset       Number of resets (for writing files)
* \param [out] sumbox       Average box
* \param [out] invbox       Inverse of the box
* \param [out] gridsp       grid spacing
* \param [out] invgridsp    inverse of grid spacing
* \param [out] current_grid Grid (either nx*ny*nz or nAtoms)
* \param [out] sum_grid     Sum Grid
* \n
* 
*/

#include "mds_common.h"
#include "mds_defines.h"
#include "mds_basicops.h"
#include "mds_cmenger.h"
#include "mds_lapack.h"

class  mds::StressGrid
{
    public :
        /** Return the name of this class as a string. */
        const char *GetNameOfClass() const
        {   return "StressGrid";     }
        
        
        /** Enable/Disable MDStress Library: */
        //@{
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
        //@}
        
        /** Set max threads: */
        //@{
        void SetThreadIDs(int thread_id, int max_threads)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_thread_map[std::this_thread::get_id()] = thread_id;
            this->m_max_threads = max_threads;
        }
        //@}
        
        /** Set/Get number of atoms: */
        //@{
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
        //@}
        
        /** Set/Get periodic boundary conditions: */
        //@{
        void SetPeriodicBoundaries(bool x, bool y, bool z, bool enforce);
        void  GetPeriodicBoundaries(bool & x, bool & y, bool & z, bool enforce)
        { 
            x = this->m_periodic[0];
            y = this->m_periodic[1];
            z = this->m_periodic[2];
            enforce = this->m_periodic[3];
        }
        //@}
        
        /** Set/Get number of grid cells in each direction: */
        //@{
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
        void SetNumberOfGridCellsXC(int nxc)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_nxyzc[0] = nxc;
        }
        void SetNumberOfGridCellsYC(int nyc)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_nxyzc[1] = nyc;
        }
        void SetNumberOfGridCellsZC(int nzc)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_nxyzc[2] = nzc;
        }
        int  GetNumberOfGridCellsXC( )
        {   return this->m_nxyzc[0]; }
        int  GetNumberOfGridCellsYC( )
        {   return this->m_nxyzc[1]; }
        int  GetNumberOfGridCellsZC( )
        {   return this->m_nxyzc[2]; }
        //@}
        
        /** Set/Get spacing in each direction: */
        //@{
        void SetSpacing(real_ext d)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_spacing = (real_int)d;
        }
        real_ext  GetSpacingX( )
        {   return (real_ext)this->m_gridsp[0]; }
        real_ext  GetSpacingY( )
        {   return (real_ext)this->m_gridsp[1]; }
        real_ext  GetSpacingZ( )
        {   return (real_ext)this->m_gridsp[2]; }
        void SetSpacingc(real_ext dc)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_spacingc = dc;
        }
        real_ext  GetSpacingXC( )
        {   return this->m_gridspc[0]; }
        real_ext  GetSpacingYC( )
        {   return this->m_gridspc[1]; }
        real_ext  GetSpacingZC( )
        {   return this->m_gridspc[2]; }
        //@}        
        
         /**Set/Get force decomposition */
        //@{
        void SetForceDecomposition ( int fdecomp )
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            m_fdecomp = fdecomp;
        }
        int GetForceDecomposition ( void ) const
        {   return m_fdecomp;   }
        //@}
        
         /**Set/Get stress type */
        //@{
        void SetStressType ( int spatatom )
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_spatatom = spatatom;
        }
        int GetStressType ( void ) const
        {   return this->m_spatatom;   }
        //@}
        
         /**Set/Get contrib type */
        //@{
        void SetContribType ( int contrib )
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_contrib = contrib;
        }
        int GetContribType ( void ) const
        {   return this->m_contrib;   }
        //@}
        
         /**Set/Get mindihangle */
        //@{
        void SetMinDihAngle (real_ext mindihangle)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_mindihangle = (real_int)mindihangle;
        }
        real_ext GetMinDihAngle ( )
        {   return (real_ext)this->m_mindihangle;   }
        //@}
         
         /**Set/Get Charge Cutoff*/
        //@{
        void SetChargeParams(int gridctype, real_ext epsfac, real_ext rcoulomb, real_ext ewaldcoeff_q)
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_gridctype = gridctype;
            this->m_epsfac = (real_int)epsfac;
            this->m_rcoulomb = (real_int)rcoulomb;
            this->m_ewaldcoeff_q = (real_int)ewaldcoeff_q;
        }
        //@}
        
        /** Compute N-body force decomposition: */
        //@{
        /** ComputeNbodyPairForces
         *
         * This function reads the number of atoms, the atoms' labels and their
         * respective positions and forces, and computes the N-body pairwise forces and vectors
         * without distributing the stress.
         * nAtoms  -> number of atoms of the contribution
         * R       -> positions of the atoms
         * F       -> forces on the atoms
         * atomIDs -> labels of the atoms (optional, only needed if calculating stress/atom) */
        void ComputeNbodyPairForces ( int nAtoms, array3_ext *R, array3_ext *F, int *atomIDs );

        array3_int * GetNbodyDecompPairForces()
        {   return this->p_Fij;    }
        array3_int * GetNbodyDecompPairVectors()
        {   return this->p_Rij;    }
        //@}

        /**Set box: */
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

        /**Set the maximum cluster size (by default mds_maxpart) */
        void SetMaxCluster ( int maxClust )
        {
            if (true == this->m_disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_maxClust = maxClust;
        }
        int GetMaxCluster ( ) const
        {   return this->m_maxClust;  }
        
        /**Set name of the files (extension .mds and numbered by resets) */
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
        
        /** This function initialize the grid depending on the settings. If the settings are incorrect, 
         * it throws an error */
        void Init ( );
        
        /** This function updates the box, invboxm and also computes the new spacings */
        void UpdateBoxSpacings ( matrix3_ext box );
        
        /** This function sums current_grid to sum_grid and sets current_grid to 0. It also increases the frame counter */
        void SumGrid ( );
        
        /** This function shifts current_grid by shift/ngrid */
        void DispersionCorrection (real_ext shift);
        
        /** Set both sum_grid and current_grid to zero. Sum the number of resets (this is used for 
         * printing files) and set the number of frames to zero */
        void Reset  ( );
        
        /**Writes file with average stress to grid using the filename set by the user*/
        void Write  ( );
        void WriteForceDecomposition ( );
        
        /**Writes and resets*/
        void WriteAndReset ( )
        {   this->Write();  this->Reset();  }
        
        /** AddVoronoiRadius
         *
         * Adds a particle radius and all calls must be completed before calling voronoi init
         * Requires:
         * radius  -> radius of the atom
         * atomID  -> label of the atom */
        void SetVoronoiRadius(real_ext radius, int atomID);

        /** AddVoronoiAtom
         *
         * Adds a particle position to the voronoi container
         * Requires:
         * px      -> position in the x dimension
         * py      -> position in the y dimension
         * pz      -> position in the z dimension
         * radius  -> radius of the atom
         * atomID  -> label of the atom
         * moleID  -> label of the molecule this atom belongs to */
        void AddVoronoiAtom(real_ext px, real_ext py, real_ext pz, int atomID, int moleID);

        /** DistributeStress
         *
         * ROOT OF ALL EVIL
         * This function reads the number of atoms, the atoms' labels and their
         * respective positions and forces, and calls the functions in charge of distributing the stress
         * on the grid depending on the local stress flags and the kind of interaction
         * Requires:
         * nAtoms  -> number of atoms of the contribution
         * R       -> positions of the atoms
         * F       -> forces on the atoms
         * atomIDs -> labels of the atoms (optional, only needed if calculating stress/atom) */
        void DistributeInteraction ( int nAtoms, array3_ext *R, array3_ext *F, int *atomIDs );
        
        /**
         *
         * This Function Computes the Born term of the Elasticity Tensor for Pairwise interactions in 3 Dimensions
	 * The Formula is from Tadmor Modeling Materials Section 8 Equation 8.84
         * Currently only Lennard Jones Elasticity is being Implimented, Will impliment coulomb Later
         * Based upon ROOT OF ALL EVIL (Take Care when debugging)
         * Coordinates xi and xj correspond to the first pair of particles (alpha and beta in Tadmor's notation) over which the Born term will be distributed
         * Coordinates xk and xl correspond to the second pair of particles in a multibody potential
         * For simple pairwise potentials, xk and xi should be the same and also xj and xl
         * */
        void DistributeElasticity (array3_ext xi_ext, array3_ext xj_ext, array3_ext xk_ext, array3_ext xl_ext, real_ext phi_ext, real_ext kappa_ext);
        
        /** DistributeKinetic
         *
         * Distributes interactions onto the grid
         * Requires:
         * mass       -> mass of the particle
         * x          -> position of the atom
         * va         -> velocity of the particle at time t for vel-verlet, or at t-dt/2 for leapfrog integrators
         * vb         -> velocity of the particle at time t+dt/2. This value is optional and only needed for leapfrog integrators
         * atomID     -> ID of the atom (optional, only needed if calculating stress/atom)
         *
         * For leapfrog integrators we know va(t-dt/2) and vb(t+dt/2), but we want the contribution at v(t) which we don't know.
         * So we take the average kinetic contribution from the velocities at each half step -m*(va(t-dt/2)^2 + vb(t+dt/2)^2)/2
         * Warning! this is not the same as simply taking the average of the half-step velocities, which would be incorrect.
         *
         * For velocity-verlet integrators we know va at the same time step, t, as the positions so the contribution is -m*va(t)*va(t)
         * */
        //@{
        void DistributeKinetic   ( real_ext mass, array3_ext x, array3_ext va, array3_ext vb, int atomID  );
        //@}
		
		/** DistributeKineticElast
		 * needs to be filled with some information
		 * */
		void DistributeKineticElast(real_ext mass, array3_ext x, array3_ext va, array3_ext vb);
        
        /** DistributeCharge
         *
         * Distributes charge contributions onto a grid
         * Requires:
         * x          -> position of the atom
         * charge     -> atomic charge
         * atomID     -> ID of the atom (optional, only needed if calculating stress/atom) */
        void DistributeCharge    ( array3_ext x, real_ext charge );
		
        /** MDStress Auxiliary Functions for the Different potentials
         * */
        // 2 Body Potentials
        void HarmonicPhiKappa   (real_ext deltaR, real_ext k, real_ext& phi, real_ext& kappa);
        
        void BuckinghamPhiKappa   (real_ext r, real_ext a, real_ext b, real_ext c, real_ext& phi, real_ext& kappa);
        
        void FourthPowerPhiKappa   (real_ext k4, real_ext dist, real_ext dist0, real_ext& phi, real_ext& kappa);
        
        void MorsePhiKappa   (real_ext expadeltaR, real_ext a, real_ext d, real_ext& phi, real_ext& kappa);
        
        void CubicBondPhiKappa   (real_ext deltaR, real_ext k, real_ext kcubic, real_ext& phi, real_ext& kappa);
        
        void FENEPhiKappa(real_ext r, real_ext k, real_ext diffratio, real_ext& phi, real_ext& kappa);
        
        // 3 Body Potentials
        void HarmonicAnglePhiKappa(real_ext ab, real_ext bg, real_ext ag, real_ext costheta, real_ext deltatheta, real_ext k, array3_ext &phi, matrix3_ext &kappa);
        
        void HarmonicCosPhiKappa(real_ext ab, real_ext bg, real_ext ag, real_ext deltacos, real_ext k, array3_ext &phi, matrix3_ext &kappa);
        
        void UreyBradleyPhiKappa(real_ext ab, real_ext bg, real_ext ag, real_ext costheta, real_ext deltaRag, real_ext deltatheta, real_ext ktheta, real_ext kUB, array3_ext &phi, matrix3_ext &kappa);
        
        void BondBondCrossPhiKappa(real_ext k, real_ext deltarab, real_ext deltarbg, array3_ext &phi, matrix3_ext &kappa);
        
        void BondAngleCrossPhiKappa(real_ext k, real_ext deltarab, real_ext deltarbg, real_ext deltarag, array3_ext &phi, matrix3_ext &kappa);
        
        void QuarticAnglePhiKappa(real_ext ab, real_ext bg, real_ext ag, real_ext costheta, real_ext deltatheta, real_ext (&coeff)[5], array3_ext &phi, matrix3_ext &kappa);
        
        /** Constructor */
        StressGrid( );

        /** Destructor */
        ~StressGrid();
        
    private:
        /** @name Inputs*/
        //@{
        int         m_nAtoms;         ///< Number of atoms
        iarray      m_nxyz;           ///< Number of grid cells in the x direction
        iarray      m_nxyzc;          ///< Number of grid cells in the x direction
        int         m_griddim;        ///< the type of grid
        long        m_ncells;         ///< Total number of cells in the calculation
        long        m_ncellsc;        ///< Total number of cells in the charge grid
        int         m_maxClust;
        real_int    m_spacing;        ///< spacing requested for the grid
        real_int    m_spacingc;       ///< spacing requested for the charge grid
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
        real_int    m_rcoulomb;
        real_int    m_epsfac;
        real_int    m_ewaldcoeff_q;
        int         m_gridctype;
        int         m_maxpart;        ///< used to allocate Rij and Fij
        int         m_max_threads;    ///< number of threads to use
        real_int    m_temperature;
        bool        m_pcoupl;
        //@}
    
        /** @name Outputs*/
        //@{
        int      m_ierr;                  ///< error type: 0, 1, etc (See GetError() function)
        int      m_nframes;               ///< Number of frames
        int      m_nreset;                ///< Number of resets (for writing files)
        matrix3_int m_sumbox;             ///< Average box
        matrix3_int m_invbox;             ///< Inverse of the box
        real_int m_avg_boxvol;            ///< Average box volume
        real_int m_var_boxvol;            ///< Variance of box volume
        real_int m_gridsp[7];             ///< grid spacing
        real_int m_gridspc[7];            ///< grid spacing
        real_int m_invgridsp;             ///< inverse of grid spacing
        real_int m_invgridspc;            ///< inverse of grid spacing
        Lapack **h_lapack;                ///< mds_lapack: solves underdetermined/overdetermined systems of equations and projects solution onto shape space
        double  *p_Amat;                  ///< matrix for linear systems (for systems with more than 5 particles)
        double  *p_AmatT;                 ///< transpose of the matrix (used for projecting solution onto the shape space)
        double  *p_bvec;                  ///< vector for solving the linear system    
        array3_int *p_Rij;                 ///< distance vectors
        array3_int *p_Fij;                 ///< force vectors
        array3_int *p_Uij;                 ///< distance vectors
        matrix3_int *p_current_grid;      ///< Grid (either nx*ny*nz or nAtoms)
        matrix3_int *p_current_gridtot;   ///< Grid of size 1
        matrix6_int *p_current_grid_elborn; ///< Grid (either nx*ny*nz or nAtoms)
        matrix6_int *p_current_grid_elkin;  ///< Grid (either nx*ny*nz or nAtoms)
        real_int    *p_current_gridc;       ///< Grid (either nx*ny*nz or nAtoms) (ewald)
        matrix3_int *p_sum_grid;            ///< Sum Grid
        matrix3_int *p_avg_grid;            ///< Sum Grid
        matrix3_int *p_avg_gridtot;         ///< Total Sum Grid
        matrix3_int *p_sum_grid_volcovar;  ///< Cov(sigma_local_ij, V)
        matrix3_int *p_sum_gridtot_volcovar;  ///< Cov(sigma_total_ij, V)
        matrix6_int *p_sum_grid_elcovar;    ///< Elasticity Grid Covariance Term
        matrix6_int *p_sum_grid_elborn;     ///< Elasticity Grid Born Term
        matrix6_int *p_sum_grid_elkin;      ///< Elasticity Grid Kinetic Term
        real_int *p_sum_gridc;              ///< Sum Grid (charge)
        real_int *p_sum_volume;             ///< Sum of volumes when using mds_atom
        real_int *p_radii;                  ///< the radius of an atomic site
        real_int *p_positions;              ///< the position of an atomic site
        array3_int *p_pos_gridc;             ///< the position of an charge grid sites
        int     *p_molecule_id;             ///< The molecule an atomic site belongs to
        //@}
        
        /** @name Threads*/
        //@{
        std::map<std::thread::id,int> m_thread_map;
        std::mutex m_mutex_state;
        //@}

        /** Method to delete the preallocated member variables */
        void Clear();

        /** This function is provided to avoid misleadings parameters, or to identify bad settings */
        int CheckSettings();

        void DistributeElasticity_internal1D(array3_int xi, array3_int xj, array3_int xk, array3_int xl, real_int phi, real_int kappa);
        void DistributeElasticity_internal3D(array3_int xi, array3_int xj, array3_int xk, array3_int xl, real_int phi, real_int kappa);
        
        /** DistributePairInteraction
         *
         * Distributes interactions onto locals_grid (from the initial grid point to the last grid point)
         * Requires:
         * R1   -> position of particle I (A)
         * R2   -> position of particle J (B)
         * F    -> pairwise force */
        void DistributePairInteraction     ( array3_int R1, array3_int R2, array3_int F, int batch_id );
        void DistributePairInteraction1D   ( array3_int R1, array3_int R2, array3_int F, int batch_id );
        void DistributePairInteraction3D   ( array3_int R1, array3_int R2, array3_int F, int batch_id );
        
        /** Decompose 3-body potentials (angles)*/
        void DistributeN3                  ( array3_int Ra, array3_int Rb, array3_int Rc, array3_int Fa, array3_int Fb, array3_int Fc, int batch_id );
        
        /** Decompose Settle */
        void DistributeSettle              ( array3_int Ra, array3_int Rb, array3_int Rc, array3_int Fa, array3_int Fb, array3_int Fc, int batch_id );
        
        /** Decompose 4-body potentials (dihedrals) */
        void DistributeN4                  ( array3_int Ra, array3_int Rb, array3_int Rc, array3_int Rd, array3_int Fa, array3_int Fb, array3_int Fc, array3_int Fd, int batch_id );
        
        /** Decompose 5-body potentials (CMAP) */
        void DistributeN5                  ( array3_int Ra, array3_int Rb, array3_int Rc, array3_int Rd, array3_int Re, array3_int Fa, array3_int Fb, array3_int Fc, array3_int Fd, array3_int Fe, int batch_id);
        
        /** General function to decompose N-body potentials (it can be used to compute higher order terms coming from EAM for instance) */
        void DistributeNBody               ( int nPart, array3_ext *R, array3_ext *F, bool distritube_stress, int batch_id);
        
        //Cosine Derivatives
        //3 Body Triangle Derivatives
        real_int CalcCosine                  (real_int ab, real_int bg, real_int ag); 
        
        real_int CalcTheta                   (real_int costheta);
        
        void ThreeBodyCosineD              (real_int ab, real_int bg, real_int ag, array3_int &d_cos_array);
        
        void ThreeBodyCosineD2             (real_int ab, real_int bg, real_int ag, matrix3_int &d2_cos_array);
        
        void ThreeBodyThetaD               (real_int costheta, array3_int &d_cos_array, array3_int &d_theta_array);
        
        void ThreeBodyThetaD2              (real_int costheta, array3_int d_cos_array, matrix3_int &d2_cos_array, matrix3_int &d2_theta_array);
        
        real_int CalcCosineDihedral          (real_int ab, real_int ag, real_int ae, real_int bg, real_int be, real_int ge);
        
        void FourBodyThetaD                (real_int costheta, array6_int &d_cos_di_array, array6_int &d_theta_di_array);
        
        void FourBodyThetaD2               (real_int costheta, array6_int &d_cos_di_array, matrix6_int &d2_cos_di_array, matrix6_int &d2_theta_di_array);
};
#endif // mds_stressgrid_h
