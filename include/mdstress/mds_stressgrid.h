/*=========================================================================

  Module    : MDStress
  File      : mds_stressgrid.h
  Authors   : A. Torres-Sanchez, J. M. Vanegas and Benjamin E. Himberg
  Modified  :
  Purpose   : Compute the local stress from MD trajectories
  Date      : 04/01/2023
  Version   :
  Changes   :

     http://mdstress.org

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

     Please, report any bug to us:
     torres.sanchez.a@gmail.com
     juan.m.vanegas@gmail.com
     bhimberg@gmail.com
=========================================================================*/

#ifndef MDS_STRESSGRID_H
#define MDS_STRESSGRID_H

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
            if (this->state.disable == true && this->state.initialized == true)
                this->state.disable = false;
        }
        void Disable()
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.disable = true;
        }

        bool CheckInit()
        {
            return this->state.initialized;
        }

        void SetThreadIDs(int thread_id, int max_threads)
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_thread_map[std::this_thread::get_id()] = thread_id;
            this->m_max_threads = max_threads;
        }

        void SetNumberOfAtoms(int n)
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.nAtoms = n;
        }
        int  GetNumberOfAtoms( )
        {
            return this->state.nAtoms;
        }

        void SetPeriodicBoundaries(bool x, bool y, bool z, bool enforce);
        void  GetPeriodicBoundaries(bool & x, bool & y, bool & z, bool enforce)
        {
            x = this->state.periodic[0];
            y = this->state.periodic[1];
            z = this->state.periodic[2];
            enforce = this->state.periodic[3];
        }

        void SetNumberOfGridCellsX(int nx)
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.nxyz[0] = nx;
        }
        void SetNumberOfGridCellsY(int ny)
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.nxyz[1] = ny;
        }
        void SetNumberOfGridCellsZ(int nz)
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.nxyz[2] = nz;
        }
        int  GetNumberOfGridCellsX( )
        {   return this->state.nxyz[0]; }
        int  GetNumberOfGridCellsY( )
        {   return this->state.nxyz[1]; }
        int  GetNumberOfGridCellsZ( )
        {   return this->state.nxyz[2]; }

        void SetSpacing(real_ext d)
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.spacing = (real_mds)d;
        }
        real_ext  GetSpacing( )
        {   return (real_ext)this->state.spacing; }
        real_ext  GetSpacingX( )
        {   return (real_ext)this->state.gridsp[0]; }
        real_ext  GetSpacingY( )
        {   return (real_ext)this->state.gridsp[1]; }
        real_ext  GetSpacingZ( )
        {   return (real_ext)this->state.gridsp[2]; }

        void SetForceDecomposition ( int fdecomp )
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.fdecomp = fdecomp;
        }
        int GetForceDecomposition ( void ) const
        {   return this->state.fdecomp;   }

        void SetStressType ( int spatatom )
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.spatatom = spatatom;
        }
        int GetStressType ( void ) const
        {   return this->state.spatatom;   }

        void SetContribType ( int contrib )
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.contrib = contrib;
        }
        int GetContribType ( void ) const
        {   return this->state.contrib;   }

        void SetMinDihAngle (real_ext mindihangle)
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.mindihangle = (real_mds)mindihangle;
        }
        real_ext GetMinDihAngle ( )
        {   return (real_ext)this->state.mindihangle;   }

        void ComputeNbodyPairForces ( int nAtoms, array3_ext *R, array3_ext *F, int *atomIDs );

        array3_mds * GetNbodyDecompPairForces()
        {   return this->alloc.Fij;    }
        array3_mds * GetNbodyDecompPairVectors()
        {   return this->alloc.Rij;    }

        void SetBox(matrix3_ext box, int epc)
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            if (epc != 0)
                this->state.pcoupl = true;
            for (int i = 0; i < mds_ndim; i++ )
                for (int j = 0; j < mds_ndim; j++)
                    this->state.box[i][j] = (real_mds)box[i][j];
        }

        void SetMaxCluster ( int maxClust )
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.maxClust = maxClust;
        }
        int GetMaxCluster ( ) const
        {   return this->state.maxClust;  }

        void SetFileName ( const char *filename )
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_filename.assign(filename);
        }

        void SetTemperature(real_ext temperature)
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.temperature = (real_mds)temperature;
        }

        real_ext GetTemperature(real_ext temperature)
        {
            return (real_ext)this->state.temperature;
        }

        void EnableCuda()
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.cuda = true;
        }

        void DisableDispersionCorrection()
        {
            if (true == this->state.disable)
                return;

            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.nodispcor = true;
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
        {   return this->state.ierr;  }

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
                const array3_ext & xi_ext,
                const array3_ext & xj_ext,
                const array3_ext & xk_ext,
                const array3_ext & xl_ext,
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

        // counters and runtime parameters
        typedef struct {
            int         nAtoms;
            iarray      nxyz;
            int         griddim;
            long        ncells;
            int         maxClust;
            real_mds    spacing;
            matrix3_mds box;
            int         spatatom;
            int         fdecomp;
            int         contrib;
            bool        nodispcor;
            bool        cuda;
            bool        initialized;
            bool        disable;
            barray      periodic;
            real_mds    mindihangle;
            int         maxpart;
            real_mds    temperature;
            bool        pcoupl;
            int         ierr;
            int         nframes;
            int         nreset;
            matrix3_mds sumbox;
            matrix3_mds invbox;
            real_mds    avg_boxvol;
            real_mds    var_boxvol;
            real_mds    gridsp[7];
            real_mds    invgridsp;
        } state_t;
    private:
        state_t state;

        // allocations
        struct MDS_ALLOC_STRUCT {
            double      *Amat;
            double      *AmatT;
            double      *bvec;
            array3_mds  *Rij;
            array3_mds  *Fij;
            array3_mds  *Uij;
            matrix3_mds *current_grid;
            matrix3_mds *current_gridtot;
            matrix6_mds *current_grid_elborn;
            matrix6_mds *current_grid_elkin;
            matrix3_mds *sum_grid;
            matrix3_mds *avg_grid;
            matrix3_mds *avg_gridtot;
            matrix6_mds *sum_grid_elcovar;
            matrix6_mds *sum_grid_elkin;
            matrix6_mds *sum_grid_elborn;
            matrix3_mds *sum_grid_volcovar;
            matrix3_mds *sum_gridtot_volcovar;
            real_mds    *sum_volume;
            int         *molecule_id;
            real_mds    *radii;
            real_mds    *positions;
            Lapack     **lapack;
        } alloc;

        // objects
        std::string m_filename;
        int m_max_threads;
        std::map<std::thread::id,int> m_thread_map;
        std::mutex m_mutex_state;

        void Clear();

        void DistributeElasticity_mdsernal3D(
                const array3_mds & xi,
                const array3_mds & xj,
                const array3_mds & xk,
                const array3_mds & xl,
                real_mds phi,
                real_mds kappa);

        void DistributePairInteraction(
                const array3_mds & R1,
                const array3_mds & R2,
                const array3_mds & F,
                int batch_id);
        void DistributePairInteraction3D(
                const array3_mds & R1,
                const array3_mds & R2,
                const array3_mds & F,
                int batch_id,
                const matrix3_mds stress,
                const array3_mds diff);

        void DistributeN3(
                const array3_mds & Ra,
                const array3_mds & Rb,
                const array3_mds & Rc,
                const array3_mds & Fa,
                const array3_mds & Fb,
                const array3_mds & Fc,
                int batch_id);
        void DistributeSettle(
                const array3_mds & Ra,
                const array3_mds & Rb,
                const array3_mds & Rc,
                const array3_mds & Fa,
                const array3_mds & Fb,
                const array3_mds & Fc,
                int batch_id);

        void DistributeN4(
                const array3_mds & Ra,
                const array3_mds & Rb,
                const array3_mds & Rc,
                const array3_mds & Rd,
                const array3_mds & Fa,
                const array3_mds & Fb,
                const array3_mds & Fc,
                const array3_mds & Fd,
                int batch_id);
        void DistributeN5(
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
                int batch_id);

        void DistributeNBody(
                int nPart,
                array3_ext *R,
                array3_ext *F,
                bool distritube_stress,
                int batch_id);
};
#endif//MDS_STRESSGRID_H
