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

#ifndef mds_support_h
#define mds_support_h

#include "mds_common.h"
#include "mds_defines.h"
#include <vector>

namespace mds {
    void HarmonicPhiKappa(
            real_ext deltaR,
            real_ext k,
            real_ext& phi,
            real_ext& kappa);
    
    void BuckinghamPhiKappa(
            real_ext r,
            real_ext a,
            real_ext b,
            real_ext c,
            real_ext& phi,
            real_ext& kappa);
    
    void FourthPowerPhiKappa(
            real_ext k4,
            real_ext dist,
            real_ext dist0,
            real_ext& phi,
            real_ext& kappa);
    
    void MorsePhiKappa(
            real_ext expadeltaR,
            real_ext a,
            real_ext d,
            real_ext& phi,
            real_ext& kappa);
    
    void CubicBondPhiKappa(
            real_ext deltaR,
            real_ext k,
            real_ext kcubic,
            real_ext& phi,
            real_ext& kappa);
    
    void FENEPhiKappa(
            real_ext r,
            real_ext k,
            real_ext diffratio,
            real_ext& phi,
            real_ext& kappa);
    
    void HarmonicAnglePhiKappa(
            real_ext ab,
            real_ext bg,
            real_ext ag,
            real_ext costheta,
            real_ext deltatheta,
            real_ext k,
            array3_ext &phi,
            matrix3_ext &kappa);
    
    void HarmonicCosPhiKappa(
            real_ext ab,
            real_ext bg,
            real_ext ag,
            real_ext deltacos,
            real_ext k,
            array3_ext &phi,
            matrix3_ext &kappa);
    
    void UreyBradleyPhiKappa(
            real_ext ab,
            real_ext bg,
            real_ext ag,
            real_ext costheta,
            real_ext deltaRag,
            real_ext deltatheta,
            real_ext ktheta,
            real_ext kUB,
            array3_ext &phi,
            matrix3_ext &kappa);
    
    void BondBondCrossPhiKappa(
            real_ext k,
            real_ext deltarab,
            real_ext deltarbg,
            array3_ext &phi,
            matrix3_ext &kappa);
    
    void BondAngleCrossPhiKappa(
            real_ext k,
            real_ext deltarab,
            real_ext deltarbg,
            real_ext deltarag,
            array3_ext &phi,
            matrix3_ext &kappa);
    
    void QuarticAnglePhiKappa(
            real_ext ab,
            real_ext bg,
            real_ext ag,
            real_ext costheta,
            real_ext deltatheta,
            real_ext (&coeff)[5],
            array3_ext &phi,
            matrix3_ext &kappa);

    void CBTDihPhiKappa(
            real_ext ab,
            real_ext bg,
            real_ext ag,
            real_ext ae,
            real_ext be,
            real_ext ge,
            std::vector<real_ext>,
            array6_ext &phi,
            matrix6_ext &kappa);
}

#endif//mds_support_h
