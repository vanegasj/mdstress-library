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

// counters and runtime parameters
namespace mds {
    typedef struct {
        int         nAtoms;
        long        nCells;
        iarray      gridCells;
        int         gridDims;
        real_mds    gridSpacing;
        int         maxClust;
        matrix3_mds box;
        int         spatatom;
        int         fdecomp;
        int         contrib;
        bool        nodispcor;
        bool        cuda;
        bool        initialized;
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

    // allocations
    typedef struct {
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
    } alloc_t;
}

class  mds::StressGrid
{
    public :
        const char *GetNameOfClass() const
        {   return "StressGrid";    }

        bool CheckInit()
        {   return this->state.initialized;    }

        void SetThreadIDs(int thread_id, int max_threads)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_thread_map[std::this_thread::get_id()] = thread_id;
            this->m_max_threads = max_threads;
        }

        void SetNumberOfAtoms(int n)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.nAtoms = n;
        }
        int  GetNumberOfAtoms( )
        {    return this->state.nAtoms;    }

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
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.gridCells[0] = nx;
        }
        void SetNumberOfGridCellsY(int ny)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.gridCells[1] = ny;
        }
        void SetNumberOfGridCellsZ(int nz)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.gridCells[2] = nz;
        }
        int  GetNumberOfGridCellsX( )
        {   return this->state.gridCells[0]; }
        int  GetNumberOfGridCellsY( )
        {   return this->state.gridCells[1]; }
        int  GetNumberOfGridCellsZ( )
        {   return this->state.gridCells[2]; }

        void SetSpacing(real_ext d)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.gridSpacing = (real_mds)d;
        }
        real_ext  GetSpacing( )
        {   return (real_ext)this->state.gridSpacing; }
        real_ext  GetSpacingX( )
        {   return (real_ext)this->state.gridsp[0]; }
        real_ext  GetSpacingY( )
        {   return (real_ext)this->state.gridsp[1]; }
        real_ext  GetSpacingZ( )
        {   return (real_ext)this->state.gridsp[2]; }

        void SetForceDecomposition ( int fdecomp )
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.fdecomp = fdecomp;
        }
        int GetForceDecomposition ( void ) const
        {   return this->state.fdecomp;   }

        void SetStressType ( int spatatom )
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.spatatom = spatatom;
        }
        int GetStressType ( void ) const
        {   return this->state.spatatom;   }

        void SetContribType ( int contrib )
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.contrib = contrib;
        }
        int GetContribType ( void ) const
        {   return this->state.contrib;   }

        void SetMinDihAngle (real_ext mindihangle)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.mindihangle = (real_mds)mindihangle;
        }
        real_ext GetMinDihAngle ( )
        {   return (real_ext)this->state.mindihangle;   }

        array3_mds * GetNbodyDecompPairForces()
        {   return this->alloc.Fij;    }
        array3_mds * GetNbodyDecompPairVectors()
        {   return this->alloc.Rij;    }

        void SetBox(matrix3_ext box, int epc)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            if (epc != 0)
                this->state.pcoupl = true;
            for (int i = 0; i < mds_ndim; i++ )
                for (int j = 0; j < mds_ndim; j++)
                    this->state.box[i][j] = (real_mds)box[i][j];
        }

        void SetMaxCluster ( int maxClust )
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.maxClust = maxClust;
        }
        int GetMaxCluster ( ) const
        {   return this->state.maxClust;  }

        void SetFileName ( const char *filename )
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_filename.assign(filename);
        }

        void SetTemperature(real_ext temperature)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.temperature = (real_mds)temperature;
        }

        real_ext GetTemperature(real_ext temperature)
        {
            return (real_ext)this->state.temperature;
        }

        void EnableCuda()
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.cuda = true;
        }

        void DisableDispersionCorrection()
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.nodispcor = true;
        }

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
        void DistributeInteraction(
                int nAtoms,
                const array3_ext *R,
                const array3_ext *F,
                const real_ext *phi,
                const real_ext *kappa,
                const int *atomIDs);
        void DistributeKinetic(
                real_ext mass,
                array3_ext x,
                array3_ext va,
                array3_ext vb,
                int atomID);

        StressGrid();
        ~StressGrid();

    private:
        state_t state;
        alloc_t alloc;

        // objects
        std::string m_filename;
        int m_max_threads;
        std::map<std::thread::id,int> m_thread_map;
        std::mutex m_mutex_state;

        void Clear();
        void DistributePairInteraction(
                const array3_mds & R1,
                const array3_mds & R2,
                const array3_mds & F,
                matrix3_mds * grid);

        void DistributeN3(
                const array3_mds & Ra,
                const array3_mds & Rb,
                const array3_mds & Rc,
                const array3_mds & Fa,
                const array3_mds & Fb,
                const array3_mds & Fc,
                Lapack * lapack,
                matrix3_mds * grid);
        void DistributeSettle(
                const array3_mds & Ra,
                const array3_mds & Rb,
                const array3_mds & Rc,
                const array3_mds & Fa,
                const array3_mds & Fb,
                const array3_mds & Fc,
                Lapack * lapack,
                matrix3_mds * stress_grid,
                matrix6_mds * elast_grid);
        void DistributeN4(
                const array3_mds & Ra,
                const array3_mds & Rb,
                const array3_mds & Rc,
                const array3_mds & Rd,
                const array3_mds & Fa,
                const array3_mds & Fb,
                const array3_mds & Fc,
                const array3_mds & Fd,
                Lapack * lapack,
                matrix3_mds * grid);
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
                Lapack * lapack,
                matrix3_mds * grid);
        void DistributeNBody(
                int nAtoms,
                const array3_ext *R,
                const array3_ext *F,
                Lapack * lapack,
                matrix3_mds * grid);
};
#endif//MDS_STRESSGRID_H
