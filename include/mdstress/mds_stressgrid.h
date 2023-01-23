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

// counters and runtime parameters
namespace mds {
    typedef struct {
        bool        nodispcor;
        bool        cuda;
        int         spatatom;
        int         fdecomp;
        int         contrib;
        real_mds    mindihangle;
        barray      periodic;
        iarray      gridCells;
        real_mds    gridSpacing;
        real_mds    temperature;
        int         nAtoms;
    } settings_t;

    typedef struct {
        bool        initialized;
        long        nCells;
        int         gridDims;
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
        matrix3_mds box;
        char        filename[128];
    } state_t;

    // allocations
    typedef struct {
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
            this->settings.nAtoms = n;
        }

        void SetPeriodicBoundaries(bool x, bool y, bool z, bool enforce);

        void SetNumberOfGridCellsX(int nx)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.gridCells[0] = nx;
        }
        void SetNumberOfGridCellsY(int ny)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.gridCells[1] = ny;
        }
        void SetNumberOfGridCellsZ(int nz)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.gridCells[2] = nz;
        }

        void SetSpacing(real_ext d)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.gridSpacing = (real_mds)d;
        }

        void SetForceDecomposition ( int fdecomp )
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.fdecomp = fdecomp;
        }

        void SetStressType ( int spatatom )
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.spatatom = spatatom;
        }

        void SetContribType ( int contrib )
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.contrib = contrib;
        }

        void SetMinDihAngle (real_ext mindihangle)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.mindihangle = (real_mds)mindihangle;
        }

        void SetBox(matrix3_ext box, int epc)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            if (epc != 0)
                this->state.pcoupl = true;
            for (int i = 0; i < mds_ndim; i++ )
                for (int j = 0; j < mds_ndim; j++)
                    this->state.box[i][j] = (real_mds)box[i][j];
        }

        void SetFileName ( const char *filename )
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_filename.assign(filename);
        }

        void SetTemperature(real_ext temperature)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.temperature = (real_mds)temperature;
        }

        real_ext GetTemperature(real_ext temperature)
        {
            return (real_ext)this->settings.temperature;
        }

        void EnableCuda()
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.cuda = true;
        }

        void DisableDispersionCorrection()
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.nodispcor = true;
        }

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

        // settings are public, since they may be read
        settings_t settings;
    private:
        state_t state;
        alloc_t alloc;

        // objects
        std::string m_filename;

        // threads
        int m_max_threads;
        std::map<std::thread::id,int> m_thread_map;
        std::mutex m_mutex_state;

        void Clear();
};
#endif//MDS_STRESSGRID_H
