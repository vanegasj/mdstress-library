/*=========================================================================

  Module    : MDStress
  File      : mds_barrier.h
  Authors   : A. Torres-Sanchez and J. M. Vanegas
  Modified  : B. E. Himberg
  Purpose   : Compute the local stress from MD trajectories
  Date      : Jul 24 2020
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

#ifndef mds_barrier_h
#include "mds_common.h"

namespace mds {
    class barrier
    {
        public:
            barrier(const barrier&) = delete;
            barrier& operator=(const barrier&) = delete;
            explicit barrier(unsigned int count) :
                m_count(count), m_generation(0), m_count_reset_value(count)
            {
            }
            void count_down_and_wait()
            {
                unsigned int gen = m_generation.load(); if (--m_count == 0)
                {
                    if (m_generation.compare_exchange_weak(gen, gen + 1))
                        m_count = m_count_reset_value;
                    return;
                }
                while ((gen == m_generation) && (m_count != 0))
                    std::this_thread::yield();
            }
        private:
            std::atomic<unsigned int> m_count;
            std::atomic<unsigned int> m_generation;
            unsigned int m_count_reset_value;
    };
}
#endif // mds_barrier_h
