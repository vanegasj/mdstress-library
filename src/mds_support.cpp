/*=========================================================================

  Module    : MDStress
  Authors   : A. Torres-Sanchez and J. M. Vanegas
  Modified  : B. Himberg and A. L. Lewis
  Purpose   : Compute the local stress from MD trajectories
  Dad::ate      : Aug-18-2025
  Version   :
  Changes   :

     http://mdstress.org

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

     Report any bugs to:
     juan.m.vanegas@gmail.com
=========================================================================*/

#include "mds_support.h"
#include "mds_basicops.h"
#include <map>
#include <functional>

//#include "./contrib/autodiff/forward/dual.hpp"
#include <autodiff/forward/dual.hpp>
namespace ad = autodiff;

#define iab 0
#define ibg 1
#define iag 2
#define iae 3
#define ibe 4
#define ige 5

using namespace mds;

// The multi-variable function for which derivatives are needed
ad::dual2nd CosTheta3(ad::dual2nd ab, ad::dual2nd bg, ad::dual2nd ag)
{
    return (ab*ab + bg*bg - ag*ag)/(2*ab*bg);
}

//autodiff Derivative of Cosine Function (creates derivative vector)
/*void ThreeBodyCosineD(real_mds rab, real_mds rbg, real_mds rag, mds::array3_mds &d_cos_array)
{
    ad::dual2nd ab = rab;
    ad::dual2nd bg = rbg;
    ad::dual2nd ag = rag;
    d_cos_array[iab] = ad::derivative(CosTheta3, ad::wrt(ab), ad::at(ab, bg, ag));
    d_cos_array[ibg] = ad::derivative(CosTheta3, ad::wrt(bg), ad::at(ab, bg, ag));
    d_cos_array[iag] = ad::derivative(CosTheta3, ad::wrt(ag), ad::at(ab, bg, ag));
}*/

//Derivative of Cosine Function (creates derivative vector)
/*void ThreeBodyCosineD(real_mds ab, real_mds bg, real_mds ag, mds::array3_mds &d_cos_array)
{
    real_mds numer, denom;

    //dab of costheta
    numer = (ab * ab) - (bg * bg) + (ag * ag);
    denom = realval_mds(2.0) * ab * ab * bg;

    d_cos_array[iab] = numer / denom;

    //dbg of costheta
    numer = -(ab * ab) + (bg * bg) + (ag * ag);
    denom = realval_mds(2.0) * ab * bg * bg;

    d_cos_array[ibg] = numer / denom;

    //dag of costheta
    numer = -ag;
    denom = ab * bg;

    d_cos_array[iag] = numer / denom;
}*/

//autodiff First and Second Derivative of Cosine Function (Creates derivative matrix(D2) and vector(D1))
void ThreeBodyCosineD2(real_mds rab, real_mds rbg, real_mds rag, mds::array3_mds &d_cos_array, mds::matrix3_mds &d2_cos_array)
{
    ad::dual2nd ab = rab;
    ad::dual2nd bg = rbg;
    ad::dual2nd ag = rag;

    auto [d0, dab, dabab] = ad::derivatives(CosTheta3, ad::wrt(ab, ab), ad::at(ab, bg, ag));
    d_cos_array[iab] = dab;
    d2_cos_array[iab][iab] = dabab;

    auto [d1, dbg, dbgab] = ad::derivatives(CosTheta3, ad::wrt(bg, ab), ad::at(ab, bg, ag));
    d_cos_array[ibg] = dbg;
    d2_cos_array[iab][ibg] = dbgab;
    d2_cos_array[ibg][iab] = dbgab;

    auto [d2, dag, dagab] = ad::derivatives(CosTheta3, ad::wrt(ag, ab), ad::at(ab, bg, ag));
    d_cos_array[iag] = dag;
    d2_cos_array[iab][iag] = dagab;
    d2_cos_array[iag][iab] = dagab;

    auto [d3, dbg2, dbgbg] = ad::derivatives(CosTheta3, ad::wrt(bg, bg), ad::at(ab, bg, ag));
    d2_cos_array[ibg][ibg] = dbgbg;

    auto [d4, dag2, dagbg] = ad::derivatives(CosTheta3, ad::wrt(ag, bg), ad::at(ab, bg, ag));
    d2_cos_array[iag][ibg] = dagbg;
    d2_cos_array[ibg][iag] = dagbg;

    auto [d5, dag3, dagag] = ad::derivatives(CosTheta3, ad::wrt(ag, ag), ad::at(ab, bg, ag));
    d2_cos_array[iag][iag] = dagag;
}

//Second Derivative of Cosine Function (Creates derivative matrix)
/*void ThreeBodyCosineD2(real_mds ab, real_mds bg, real_mds ag, matrix3_mds &d2_cos_array) {
    real_mds numer;
    real_mds denom;

    //d00 of costheta
    numer = (bg * bg) - (ag * ag);
    denom = (ab * ab * ab * bg);

    d2_cos_array[iab][iab] = numer / denom;

    // d01 and d10 of costheta

    numer = -((ab * ab) + (bg * bg) + (ag * ag));
    denom = realval_mds(2.0) * (ab * ab) * (bg * bg);

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
    numer = realval_mds(-1.0);
    denom = ab * bg;

    d2_cos_array[iag][iag] = numer / denom;
}*/

ad::dual2nd CosTheta4(ad::dual2nd ab, ad::dual2nd bg, ad::dual2nd ag, ad::dual2nd ae, ad::dual2nd be, ad::dual2nd ge)
{
    ad::dual2nd D_bge = -(bg + be + ge)*(bg + be - ge)*(ge + bg - be)*(be + ge - bg);
    ad::dual2nd D_bga = -(bg + ab + ag)*(bg + ab - ag)*(ag + bg - ab)*(ab + ag - bg);
    ad::dual2nd D_bg  = -bg*bg*bg*bg + (be*be + ab*ab + ge*ge + ag*ag - 2*ae*ae)*bg*bg + (be*be - ge*ge)*(ag*ag - ab*ab);
    return D_bg/(sqrt(D_bge * D_bga));
}

void FourBodyCosineD2(real_mds rab, real_mds rbg, real_mds rag, real_mds rae, real_mds rbe, real_mds rge, mds::array6_mds &d_cos_array, mds::matrix6_mds &d2_cos_array)
{
    ad::dual2nd ab = rab;
    ad::dual2nd bg = rbg;
    ad::dual2nd ag = rag;
    ad::dual2nd ae = rae;
    ad::dual2nd be = rbe;
    ad::dual2nd ge = rge;

    ad::dual2nd vars[6] = {ab, bg, ag, ae, be, ge};

    for(int i = 0; i < 6; i++)
    {
        for(int j = i; j < 6; j++)
        {
            auto derive = derivatives(CosTheta4, ad::wrt(vars[i], vars[j]), ad::at(ab, ag, ae, bg, be, ge));
            d2_cos_array[i][j] = derive[2];
            d2_cos_array[j][i] = derive[2];
            if(i == j)
                d_cos_array[i] = derive[1];
        }
    }
}

//First Derivative of Theta Function (Creates 1st derivative vector)
//Need Cosine Theta
//Need Derivative of Cosine Vector
void ThreeBodyThetaD(real_mds costheta, array3_mds &d_cos_array, array3_mds &d_theta_array) {
    real_mds scalefactor;

    scalefactor = realval_mds(-1.0) / sqrt(realval_mds(1.0) - (costheta * costheta));

    for (int i = 0; i < 3; i++) {
        d_theta_array[i] = scalefactor * d_cos_array[i];
    }
}

//Second Derivative of Theta Function (Creates 2nd Derivative Matix)
//Need Cosine Theta
//Need Derivative of Cosine Vector
//Need 2nd Derivative of Cosine Matrix
void ThreeBodyThetaD2(real_mds costheta, array3_mds d_cos_array, matrix3_mds &d2_cos_array, matrix3_mds &d2_theta_array) {
    real_mds scalefactor;
    real_mds sinthetasq;

    sinthetasq = realval_mds(1.0) - (costheta * costheta);

    scalefactor = realval_mds(1.0) / (sinthetasq * sqrt(sinthetasq));

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            d2_theta_array[i][j] = scalefactor * ((-sinthetasq) * d2_cos_array[i][j] - costheta * d_cos_array[i] * d_cos_array[j]);
        }
    }
}

void FourBodyThetaD(real_mds costheta, array6_mds &d_cos_array, array6_mds &d_theta_array) {
    real_mds scalefactor;

    scalefactor = realval_mds(-1.0) / sqrt(realval_mds(1.0) - (costheta * costheta));

    for (int i = 0; i < 6; i++) {
        d_theta_array[i] = scalefactor * d_cos_array[i];
    }
}

void FourBodyThetaD2(real_mds costheta, array6_mds &d_cos_array, matrix6_mds &d2_cos_array, matrix6_mds &d2_theta_array) {
    real_mds scalefactor;
    real_mds sinthetasq;

    sinthetasq = realval_mds(1.0) - (costheta * costheta);
    scalefactor = realval_mds(1.0) / (sinthetasq * sqrt(sinthetasq));

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            d2_theta_array[i][j] = scalefactor * ((-sinthetasq) * d2_cos_array[i][j] - costheta * d_cos_array[i] * d_cos_array[j]);
        }
    }
}

//Auxiliary Methods to Calculate the Phi and Kappa Terms for the Born Term
// For cosine derivative functions must be called before the theta derivatives. 
// derivative identifier dertype ab = 0, bg = 1, ag = 2

// ab|bg|ag = 0 | 1 | 2

// second derivative identifier derivative dertype2

//indices are flipped be careful!

/*
 * abab | bgab | agab    = 00 | 01 | 02
 * abbg | bgbg | agbg    = 10 | 11 | 12
 * abag | bgag | agag    = 20 | 21 | 22
 */

//List of 2-body Potential Auxilery Functions
//Calculate The Phi and Kappa for Harmonic Potential -> Implimented
void mds::HarmonicPhiKappa(real_ext deltaR, real_ext k, real_ext &phi, real_ext &kappa)
{
    phi = k * deltaR;
    kappa = k;
}

//Calculate the Phi and Kappa for Buckingham Potential -> Not Implimented, Nonbonded?
void mds::BuckinghamPhiKappa(real_ext r, real_ext a, real_ext b, real_ext c, real_ext &phi, real_ext &kappa)
{
    double exponent = -b * r;
    double rinv = 1 / r;
    double rinvsq = rinv * rinv;
    double rinvsix = rinvsq * rinvsq * rinvsq;

    phi = -a * b * exp(exponent) + (6 * c) * rinvsix * rinv;
    kappa = a * b * b * exp(exponent) - (42 * c) * rinvsix * rinvsq;
}

//Calculate the Phi and Kappa for the Fourth Power Potential -> Implimented g96bond
void mds::FourthPowerPhiKappa(real_ext k4, real_ext dist, real_ext dist0, real_ext &phi, real_ext &kappa)
{
    phi = k4 * dist * (dist * dist - dist0 * dist0);
    kappa = k4 * (3 * dist * dist - dist0 * dist0);
}

//Calculate the Phi and  Kappa for the Morse Potential -> Implimented
void mds::MorsePhiKappa(real_ext expadeltaR, real_ext a, real_ext d, real_ext &phi, real_ext &kappa)
{
    //expadeltaR = e^(-a*(r-r0))
    double coeffexp = 2 * a * d * expadeltaR;

    phi = coeffexp * (1 - expadeltaR);
    kappa = coeffexp * a * (2 * expadeltaR - 1);
}

//Calculate the Phi and Kappa for the Cubic Bond Potential -> Implimented
void mds::CubicBondPhiKappa(real_ext deltaR, real_ext k, real_ext kcubic, real_ext &phi, real_ext &kappa)
{
    phi = 2 * k * deltaR + 3 * k * kcubic * deltaR * deltaR;
    kappa = 2 * k + 6 * k * kcubic * deltaR;
}

//Calculate the Phi and Kappa for the FENE Potential -> Implimented
void mds::FENEPhiKappa(real_ext r, real_ext k, real_ext diffratio, real_ext& phi, real_ext& kappa)
{
    //diffratio = 1 - (r/r0)^2

    phi = k * r / diffratio;
    kappa = k * (2 - diffratio) / diffratio;
}

//List of 3-body Potential Auxiliary Functions
//Derivative Vector/Matrix form
// derivative identifier ab = 0, bg = 1, ag = 3

// ab|bg|ag = 0 | 1 | 2
/*
 * abab | bgab | agab    = 00 | 01 | 02
 * abbg | bgbg | agbg    = 10 | 11 | 12
 * abag | bgag | agag    = 20 | 21 | 22
 */

//Calculate the Phi and Kappa for the Harmonic Angle Potential -> Implimented
void mds::HarmonicAnglePhiKappa(real_ext ab_ext, real_ext bg_ext, real_ext ag_ext, real_ext costheta_ext, real_ext deltatheta_ext, real_ext k_ext, array3_ext &phi, matrix3_ext &kappa) 
{
    // convert to internal precision
    real_mds ab = (real_mds)ab_ext;
    real_mds bg = (real_mds)bg_ext;
    real_mds ag = (real_mds)ag_ext;
    real_mds costheta = (real_mds)costheta_ext;
    real_mds deltatheta = (real_mds)deltatheta_ext;
    real_mds k = (real_mds)k_ext;

    array3_mds d_cos_array;
    zeroarray3(d_cos_array);
    matrix3_mds d2_cos_array;
    zeromatrix3(d2_cos_array);
    array3_mds d_theta_array;
    zeroarray3(d_theta_array);
    matrix3_mds d2_theta_array;
    zeromatrix3(d2_theta_array);
    //ThreeBodyCosineD(ab, bg, ag, d_cos_array);
    ThreeBodyCosineD2(ab, bg, ag, d_cos_array, d2_cos_array);
    ThreeBodyThetaD(costheta, d_cos_array, d_theta_array);
    ThreeBodyThetaD2(costheta, d_cos_array, d2_cos_array, d2_theta_array);

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
void mds::HarmonicCosPhiKappa(real_ext ab_ext, real_ext bg_ext, real_ext ag_ext, real_ext deltacos_ext, real_ext k_ext, array3_ext &phi, matrix3_ext &kappa)
{
    // convert to internal precision
    real_mds ab = (real_mds)ab_ext;
    real_mds bg = (real_mds)bg_ext;
    real_mds ag = (real_mds)ag_ext;
    real_mds deltacos = (real_mds)deltacos_ext;
    real_mds k = (real_mds)k_ext;

    array3_mds d_cos_array;
    zeroarray3(d_cos_array);
    matrix3_mds d2_cos_array;
    zeromatrix3(d2_cos_array);
    //ThreeBodyCosineD(ab, bg, ag, d_cos_array);
    ThreeBodyCosineD2(ab, bg, ag, d_cos_array, d2_cos_array);

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
void mds::UreyBradleyPhiKappa(real_ext ab_ext, real_ext bg_ext, real_ext ag_ext, real_ext costheta_ext, real_ext deltaRag_ext, real_ext deltatheta_ext, real_ext ktheta_ext, real_ext kUB_ext, array3_ext &phi, matrix3_ext &kappa)
{
    // convert to internal precision
    real_mds ab = (real_mds)ab_ext;
    real_mds bg = (real_mds)bg_ext;
    real_mds ag = (real_mds)ag_ext;
    real_mds costheta = (real_mds)costheta_ext;
    real_mds deltaRag = (real_mds)deltaRag_ext;
    real_mds deltatheta = (real_mds)deltatheta_ext;
    real_mds ktheta = (real_mds)ktheta_ext;
    real_mds kUB = (real_mds)kUB_ext;

    array3_mds d_cos_array;
    zeroarray3(d_cos_array);

    matrix3_mds d2_cos_array;
    zeromatrix3(d2_cos_array);

    array3_mds d_theta_array;
    zeroarray3(d_theta_array);

    matrix3_mds d2_theta_array;
    zeromatrix3(d2_theta_array);

    //ThreeBodyCosineD(ab, bg, ag, d_cos_array);
    ThreeBodyCosineD2(ab, bg, ag, d_cos_array, d2_cos_array);
    ThreeBodyThetaD(costheta, d_cos_array, d_theta_array);
    ThreeBodyThetaD2(costheta, d_cos_array, d2_cos_array, d2_theta_array);

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
void mds::BondBondCrossPhiKappa(real_ext k, real_ext deltarab, real_ext deltarbg, array3_ext &phi, matrix3_ext &kappa)
{
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
void mds::BondAngleCrossPhiKappa(real_ext k, real_ext deltarab, real_ext deltarbg, real_ext deltarag, array3_ext &phi, matrix3_ext &kappa)
{
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
void mds::QuarticAnglePhiKappa(real_ext ab_ext, real_ext bg_ext, real_ext ag_ext, real_ext costheta_ext, real_ext deltatheta_ext, real_ext (&coeff)[5], array3_ext &phi, matrix3_ext &kappa)
{
    // convert to internal precision
    real_mds ab = (real_mds)ab_ext;
    real_mds bg = (real_mds)bg_ext;
    real_mds ag = (real_mds)ag_ext;
    real_mds costheta = (real_mds)costheta_ext;
    real_mds deltatheta = (real_mds)deltatheta_ext;

    array3_mds d_cos_array;
    zeroarray3(d_cos_array);

    matrix3_mds d2_cos_array;
    zeromatrix3(d2_cos_array);

    array3_mds d_theta_array;
    zeroarray3(d_theta_array);

    matrix3_mds d2_theta_array;
    zeromatrix3(d2_theta_array);

    //ThreeBodyCosineD(ab, bg, ag, d_cos_array);
    ThreeBodyCosineD2(ab, bg, ag, d_cos_array, d2_cos_array);
    ThreeBodyThetaD(costheta, d_cos_array, d_theta_array);
    ThreeBodyThetaD2(costheta, d_cos_array, d2_cos_array, d2_theta_array);


    real_mds deltathetasq = deltatheta * deltatheta;
    real_mds deltathetacube = deltathetasq * deltatheta;

    //Calculate the Finate Sums
    real_mds phiconst = (real_mds)coeff[1] + realval_mds(2.0) * (real_mds)coeff[2] * deltatheta + realval_mds(3.0) * (real_mds)coeff[3] * deltathetasq + realval_mds(4.0) * (real_mds)coeff[4] * deltathetacube;
    real_mds kappaconst = realval_mds(2.0) * (real_mds)coeff[2] + realval_mds(6.0) * (real_mds)coeff[3] * deltatheta + realval_mds(12.0) * (real_mds)coeff[4] * deltathetasq + realval_mds(20.0);

    //Calculate Phi Vector
    for (int i = 0; i < 3; i++) {
        phi[i] = (real_mds)(phiconst * d_theta_array[i]);
    }

    //Calculate Kappa Matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            kappa[i][j] = (real_mds)(phiconst * d2_theta_array[i][j] + kappaconst * d_theta_array[i] * d_theta_array[j]);
        }
    }
}

void FourBodyPhiKappa(real_ext rab, real_ext rbg, real_ext rag, real_ext rae, real_ext rbe, real_ext rge, std::vector<real_ext> par, array6_ext &phi, matrix6_ext &kappa,
                      std::function<ad::dual2nd(ad::dual2nd, ad::dual2nd, ad::dual2nd, ad::dual2nd, ad::dual2nd, ad::dual2nd, std::vector<real_ext>)> func)
{
    ad::dual2nd ab = rab;
    ad::dual2nd bg = rbg;
    ad::dual2nd ag = rag;
    ad::dual2nd ae = rae;
    ad::dual2nd be = rbe;
    ad::dual2nd ge = rge;

    auto [ad0, ad1, ad2] = derivatives(func, ad::wrt(ab, ab), ad::at(ab, bg, ag, ae, be, ge, par));
    phi[iab] = ad1;
    kappa[iab][iab] = ad2;
    auto [bd0, bd1, bd2] = derivatives(func, ad::wrt(ab, ag), ad::at(ab, bg, ag, ae, be, ge, par));
    kappa[iab][iag] = bd2;
    kappa[iag][iab] = bd2;
    auto [cd0, cd1, cd2] = derivatives(func, ad::wrt(ab, ae), ad::at(ab, bg, ag, ae, be, ge, par));
    kappa[iab][iae] = cd2;
    kappa[iae][iab] = cd2;
    auto [dd0, dd1, dd2] = derivatives(func, ad::wrt(ab, bg), ad::at(ab, bg, ag, ae, be, ge, par));
    kappa[iab][ibg] = dd2;
    kappa[ibg][iab] = dd2;
    auto [ed0, ed1, ed2] = derivatives(func, ad::wrt(ab, be), ad::at(ab, bg, ag, ae, be, ge, par));
    kappa[iab][ibe] = ed2;
    kappa[ibe][iab] = ed2;
    auto [fd0, fd1, fd2] = derivatives(func, ad::wrt(ab, ge), ad::at(ab, bg, ag, ae, be, ge, par));
    kappa[iab][ige] = fd2;
    kappa[ige][iab] = fd2;

    auto [gd0, gd1, gd2] = derivatives(func, ad::wrt(ag, ag), ad::at(ab, bg, ag, ae, be, ge, par));
    phi[iag] = gd1;
    kappa[iag][iag] = gd2;
    auto [hd0, hd1, hd2] = derivatives(func, ad::wrt(ag, ae), ad::at(ab, bg, ag, ae, be, ge, par));
    kappa[iag][iae] = hd2;
    kappa[iae][iag] = hd2;
    auto [id0, id1, id2] = derivatives(func, ad::wrt(ag, bg), ad::at(ab, bg, ag, ae, be, ge, par));
    kappa[iag][ibg] = id2;
    kappa[ibg][iag] = id2;
    auto [jd0, jd1, jd2] = derivatives(func, ad::wrt(ag, be), ad::at(ab, bg, ag, ae, be, ge, par));
    kappa[iag][ibe] = jd2;
    kappa[ibe][iag] = jd2;
    auto [kd0, kd1, kd2] = derivatives(func, ad::wrt(ag, ge), ad::at(ab, bg, ag, ae, be, ge, par));
    kappa[iag][ige] = kd2;
    kappa[ige][iag] = kd2;

    auto [ld0, ld1, ld2] = derivatives(func, ad::wrt(ae, ae), ad::at(ab, bg, ag, ae, be, ge, par));
    phi[iae] = ld1;
    kappa[iae][iae] = ld2;
    auto [md0, md1, md2] = derivatives(func, ad::wrt(ae, bg), ad::at(ab, bg, ag, ae, be, ge, par));
    kappa[iae][ibg] = md2;
    kappa[ibg][iae] = md2;
    auto [nd0, nd1, nd2] = derivatives(func, ad::wrt(ae, be), ad::at(ab, bg, ag, ae, be, ge, par));
    kappa[iae][ibe] = nd2;
    kappa[ibe][iae] = nd2;
    auto [od0, od1, od2] = derivatives(func, ad::wrt(ae, ge), ad::at(ab, bg, ag, ae, be, ge, par));
    kappa[iae][ige] = od2;
    kappa[ige][iae] = od2;

    auto [pd0, pd1, pd2] = derivatives(func, ad::wrt(bg, bg), ad::at(ab, bg, ag, ae, be, ge, par));
    phi[ibg] = pd1;
    kappa[ibg][ibg] = pd2;
    auto [qd0, qd1, qd2] = derivatives(func, ad::wrt(bg, be), ad::at(ab, bg, ag, ae, be, ge, par));
    kappa[ibg][ibe] = qd2;
    kappa[ibe][ibg] = qd2;
    auto [rd0, rd1, rd2] = derivatives(func, ad::wrt(bg, ge), ad::at(ab, bg, ag, ae, be, ge, par));
    kappa[ibg][ige] = rd2;
    kappa[ige][ibg] = rd2;

    auto [sd0, sd1, sd2] = derivatives(func, ad::wrt(be, be), ad::at(ab, bg, ag, ae, be, ge, par));
    phi[ibe] = sd1;
    kappa[ibe][ibe] = sd2;
    auto [td0, td1, td2] = derivatives(func, ad::wrt(be, ge), ad::at(ab, bg, ag, ae, be, ge, par));
    kappa[ibe][ige] = td2;
    kappa[ige][ibe] = td2;

    auto [ud0, ud1, ud2] = derivatives(func, ad::wrt(ge, ge), ad::at(ab, bg, ag, ae, be, ge, par));
    phi[ige] = ud1;
    kappa[ige][ige] = ud2;
}

ad::dual2nd CBTDihPot(ad::dual2nd ab, ad::dual2nd bg, ad::dual2nd ag, ad::dual2nd ae, ad::dual2nd be, ad::dual2nd ge, std::vector<real_ext> par)
{
    //variable testing
    /*
    auto [cosine_theta_ante, d1, b1] = derivatives(CosTheta3, ad::wrt(ab, ab), ad::at(ab, bg, ag));
    auto [cosine_theta_post, d2, b2] = derivatives(CosTheta3, ad::wrt(bg, bg), ad::at(bg, ge, be));
    auto [cosine_phi, d3, b3] = derivatives(CosTheta4, ad::wrt(ab, ab), ad::at(ab, bg, ag, ae, be, ge));
    double sine_theta_ante = sqrt(1 - cosine_theta_ante*cosine_theta_ante);
    double sine_theta_post = sqrt(1 - cosine_theta_post*cosine_theta_post);
    printf("AD sin_i-1 = %6.6f, cos_i-1 = %6.6f, sin_i = %6.6f, cos_i = %6.6f, cos_phi = %6.6f\n",
            sine_theta_ante, cosine_theta_ante, sine_theta_post, cosine_theta_post, cosine_phi);
    */

    ad::dual2nd k  = par[0];
    ad::dual2nd a0 = par[1];
    ad::dual2nd a1 = par[2];
    ad::dual2nd a2 = par[3];
    ad::dual2nd a3 = par[4];
    ad::dual2nd a4 = par[5];
    return k*pow(sqrt(1 - CosTheta3(ab, bg, ag)*CosTheta3(ab, bg, ag)), 3)*
                 pow(sqrt(1 - CosTheta3(bg, ge, be)*CosTheta3(bg, ge, be)), 3)*
                (a0 + a1*CosTheta4(ab, bg, ag, ae, be, ge) +
                          a2*pow(CosTheta4(ab, bg, ag, ae, be, ge),2) +
                          a3*pow(CosTheta4(ab, bg, ag, ae, be, ge),3) +
                          a4*pow(CosTheta4(ab, bg, ag, ae, be, ge),4));
}

//Calculate the Phi and Kappa for CBT - Combined Bending Torsion Dihedral
void mds::CBTDihPhiKappa(real_ext rab, real_ext rbg, real_ext rag, real_ext rae, real_ext rbe, real_ext rge, std::vector<real_ext> par, array6_ext &phi, matrix6_ext &kappa)
{
    FourBodyPhiKappa(rab, rbg, rag, rae, rbe, rge, par, phi, kappa, CBTDihPot);
}

// Equations derived from: https://manual.gromacs.org/documentation/current/reference-manual/functions/bonded-interactions.html

// Improper dihedrals: Harmonic type (Eq. 197)
ad::dual2nd ImpDihHarmPot(ad::dual2nd ab, ad::dual2nd bg, ad::dual2nd ag, ad::dual2nd ae, ad::dual2nd be, ad::dual2nd ge, std::vector<real_ext> par)
{
    ad::dual2nd k  = par[0];
    ad::dual2nd xi0 = par[1];

    ad::dual2nd xi = arccos(CosTheta4(ab, bg, ag, ae, be, ge));

    return 0.5*k*pow((xi - xi0), 2);
}

//Calculate the Phi and Kappa for Improper dihedrals: Harmonic type
void mds::ImpDihHarmPhiKappa(real_ext rab, real_ext rbg, real_ext rag, real_ext rae, real_ext rbe, real_ext rge, std::vector<real_ext> par, array6_ext &phi, matrix6_ext &kappa)
{
    FourBodyPhiKappa(rab, rbg, rag, rae, rbe, rge, par, phi, kappa, ImpDihHarmPot);
}


// Proper dihedrals: Periodic type (Eq. 198)
ad::dual2nd ProperDihPot(ad::dual2nd ab, ad::dual2nd bg, ad::dual2nd ag, ad::dual2nd ae, ad::dual2nd be, ad::dual2nd ge, std::vector<real_ext> par)
{
    ad::dual2nd phi = arccos(CosTheta4(ab, bg, ag, ae, be, ge));
    ad::dual2nd k = par[0];
    ad::dual2nd n = par[1];
    ad::dual2nd phis = par[2];
    
    return k*(1 + cos(n*phi - phis));
}

//Calculate the Phi and Kappa for Proper dihedrals: Periodic type
void mds::ProperDihPhiKappa(real_ext rab, real_ext rbg, real_ext rag, real_ext rae, real_ext rbe, real_ext rge, std::vector<real_ext> par, array6_ext &phi, matrix6_ext &kappa)
{
    FourBodyPhiKappa(rab, rbg, rag, rae, rbe, rge, par, phi, kappa, ProperDihPot);
}


// Proper dihedrals: Ryckaert-Bellemans (Eq. 199)
ad::dual2nd RyckBelleDihPot(ad::dual2nd ab, ad::dual2nd bg, ad::dual2nd ag, ad::dual2nd ae, ad::dual2nd be, ad::dual2nd ge, std::vector<real_ext> par)
{
    ad::dual2nd cosphi = CosTheta4(ab, bg, ag, ae, be, ge);
    ad::dual2nd c0 = par[0];
    ad::dual2nd c1 = par[1];
    ad::dual2nd c2 = par[2];
    ad::dual2nd c3 = par[3];
    ad::dual2nd c4 = par[4];
    ad::dual2nd c5 = par[5];

    // This equation normally uses Psi, which is equivalent to (Phi - 180 degrees)
    // Conversions between Psi and Phi can also be done by multiplying each cn by -1^n, as used below
    return  c0 + 
            -c1*cosphi +
            c2*pow(cosphi, 2) +
            -c3*pow(cosphi, 3) +
            c4*pow(cosphi, 4) +
            -c5*pow(cosphi, 5);
}

//Calculate the Phi and Kappa for Proper dihedrals: Periodic type
void mds::RyckBelleDihPhiKappa(real_ext rab, real_ext rbg, real_ext rag, real_ext rae, real_ext rbe, real_ext rge, std::vector<real_ext> par, array6_ext &phi, matrix6_ext &kappa)
{
    FourBodyPhiKappa(rab, rbg, rag, rae, rbe, rge, par, phi, kappa, RyckBelleDihPot);
}


// Proper dihedrals: Restricted torsion potential (Eq. 203)
ad::dual2nd RestrictTorPot(ad::dual2nd ab, ad::dual2nd bg, ad::dual2nd ag, ad::dual2nd ae, ad::dual2nd be, ad::dual2nd ge, std::vector<real_ext> par)
{
    ad::dual2nd cosphi = CosTheta4(ab, bg, ag, ae, be, ge);
    ad::dual2nd k = par[0];
    ad::dual2nd cosphi0 = par[1];

    return 0.5*k*((pow(cosphi - cosphi0), 2) / (1 - pow(cosphi, 2)));
}

//Calculate the Phi and Kappa for Proper dihedrals: Periodic type
void mds::RestrictTorPhiKappa(real_ext rab, real_ext rbg, real_ext rag, real_ext rae, real_ext rbe, real_ext rge, std::vector<real_ext> par, array6_ext &phi, matrix6_ext &kappa)
{
    FourBodyPhiKappa(rab, rbg, rag, rae, rbe, rge, par, phi, kappa, RestrictTorPot);
}