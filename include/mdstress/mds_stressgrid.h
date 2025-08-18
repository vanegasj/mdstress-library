/*=========================================================================

  Module    : MDStress
  Authors   : A. Torres-Sanchez and J. M. Vanegas
  Modified  : B. Himberg and A. L. Lewis
  Purpose   : Compute the local stress from MD trajectories
  Date      : Aug-18-2025
  Version   :
  Changes   :

     http://mdstress.org

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

     Report any bugs to:
     juan.m.vanegas@gmail.com
=========================================================================*/

#ifndef MDS_STRESSGRID_H
#define MDS_STRESSGRID_H

#include "mds_common.h"
#include "mds_defines.h"
#include "mds_basicops.h"
#include "mds_cmenger.h"

class  mds::StressGrid
{
    public :
        StressGrid();
        ~StressGrid();

        void SetMaxThreads(int max_threads)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_max_threads = max_threads;
        }

        void SetThreadIDs(int thread_id)
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_thread_map[std::this_thread::get_id()] = thread_id;
        }

        void SetPeriodicBoundaries(bool x, bool y, bool z, bool enforce)
        { 
            std::lock_guard<std::mutex> lock(m_mutex_state);

            #ifdef CUSTRESS_ENABLE
            if (this->settings.cuda)
                custress_set_periodic(x, y, z, enforce);
            #endif//CUSTRESS_ENABLE

            this->state.periodic[0] = x;
            this->state.periodic[1] = y;
            this->state.periodic[2] = z;
            this->state.periodic[3] = enforce;
        }

        void SetGridNCells( int nx, int ny, int nz) {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->state.gridCells[0] = nx;
            this->state.gridCells[1] = ny;
            this->state.gridCells[2] = nz;
        }

        void SetGridSpacing(real_ext d) {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.gridSpacing = (real_mds)d;
        }
        
        void SetImpulseWidth(real_ext d) {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.impulseWidth= (real_mds)d;
        }

        void SetForceDecomposition ( int fdecomp ) {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.fdecomp = fdecomp;
        }

        void SetContribType ( int contrib ) {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.contrib = contrib;
        }

        void SetDebugPrint ( int debugprint ) {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.debugprint = debugprint;
        }

        void SetMinDihAngle (real_ext mindihangle) {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.mindihangle = (real_mds)mindihangle;
        }

        void SetBox(matrix3_ext box, int epc) {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            if (epc != 0)
                this->settings.pcoupl = true;
            for (int i = 0; i < mds_ndim; i++ )
                for (int j = 0; j < mds_ndim; j++)
                    this->state.box[i][j] = (real_mds)box[i][j];
        }

        void SetFileName ( const char *filename ) {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->m_filename.assign(filename);
        }

        void SetTemperature(real_ext temperature) {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.temperature = (real_mds)temperature;
        }

        // returns the last measured frame while setting the current frame
        int64_t SetFrameId(int64_t frameId, bool measure);

        void EnableCuda() {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.cuda = true;
        }

        void DisableDispersionCorrection()
        {
            std::lock_guard<std::mutex> lock(m_mutex_state);
            this->settings.nodispcor = true;
        }

        void Init( );

        void DistributeInteraction(
                int nAtoms,
                const array3_ext *R,
                const array3_ext *F,
                const real_ext *phi,
                const real_ext *kappa);

        void DistributeKinetic(
                real_ext mass,
                array3_ext x,
                array3_ext va,
                array3_ext vb);

        void UpdateBoxSpacings(matrix3_ext box);
        void DispersionCorrection(real_ext shift);
        void SumGrid( );
        
        // save stress state in line with gromacs checkpoints
        bool SaveCheckpoint(const char * filename, const char * temp_filename);
        bool LoadCheckpoint(const char * filename);
        
        void Reset( );
        void Write( );

        // settings are public, since they may be read
        // by the application itself
        settings_t settings;
    private:
        state_t state;
        alloc_t alloc;

        // filenames
        std::string m_filename;
        std::string m_filename_cpt;
        std::string m_filename_cpt_temp;

        // threads
        int m_max_threads;
        std::map<std::thread::id,int> m_thread_map;
        std::mutex m_mutex_state;

        void Clear();
};
#endif//MDS_STRESSGRID_H
