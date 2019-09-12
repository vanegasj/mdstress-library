#include "mds_stressgrid.h"
#include <random>

typedef double real_t;
#define realval(a) (a)

#define NUM_ATOMS 100
#define SEED 994373
#define LX realval(2.0)
#define LY realval(2.0)
#define LZ realval(2.0)
#define MCSTEPS 10000

static inline real_t gradV(const mds::darray & a, const mds::darray & b, mds::darray & fab)
{
    // aziz constants
    constexpr real_t rm    = realval(2.9673);
    constexpr real_t eps   = realval(10.8);
    constexpr real_t A     = realval(0.5448504e6);
    constexpr real_t alpha = realval(13.353384);
    constexpr real_t beta  = realval(0.0);
    constexpr real_t C6    = realval(1.3732412);
    constexpr real_t C8    = realval(0.4253785);
    constexpr real_t C10   = realval(0.178100);
    constexpr real_t D     = realval(1.241314);
    
    // get the separation
    mds::darray sep = {a[0]-b[0],a[1]-b[1],a[2]-b[2]};

    // put it in the box
    sep[0] = (sep[0] > LX) ? (sep[0] - LX) : (sep[0]);
    sep[1] = (sep[1] > LY) ? (sep[1] - LY) : (sep[1]);
    sep[2] = (sep[2] > LZ) ? (sep[2] - LZ) : (sep[2]);
    sep[0] = (sep[0] <= realval(0.0)) ? (sep[0] + LX) : (sep[0]);
    sep[1] = (sep[1] <= realval(0.0)) ? (sep[1] + LY) : (sep[1]);
    sep[2] = (sep[2] <= realval(0.0)) ? (sep[2] + LZ) : (sep[2]);

    // take shortest distance across periodic boundary
    sep[0] = (sep[0] < LX - sep[0]) ? (sep[0]) : (LX - sep[0]);
    sep[1] = (sep[1] < LY - sep[1]) ? (sep[1]) : (LY - sep[1]);
    sep[2] = (sep[2] < LZ - sep[2]) ? (sep[2]) : (LZ - sep[2]);

    // normalize r and calculate all powers
    real_t x = sqrt(sep[0]*sep[0]+sep[1]*sep[1]+sep[2]*sep[2]);
    real_t x2 = x*x;
    real_t x4 = x2*x2;
    real_t x6 = x4*x2;
    real_t x8 = x4*x4;
    real_t x10 = x6*x4;

    // F and gradVint2 are conditional
    real_t F = realval(1.0);
    real_t gradMag = realval(0.0);
    if (x <= D)
    {
        real_t xD = D/x-realval(1.0);
        F = exp(-xD*xD);
        gradMag = -realval(2.0)*xD*D*F*(C6/x6+C8/x8+C10/x10)/x2;
    }
    gradMag += A*(-alpha+realval(2.0)*beta*x)*exp(-alpha*x+beta*x2)
        + F*(realval(6.0)*C6/x6+realval(8.0)*C8/x8+realval(10.0)*C10/x10)/x;

    // scale the gradient
    gradMag *= eps*x;

    // store the gradient
    fab[0] += gradMag*sep[0];
    fab[1] += gradMag*sep[1];
    fab[2] += gradMag*sep[2];
    
    // return the potential
    return eps*(A*exp(-alpha*x+beta*x2)-(C6/x6+C8/x8+C10/x10)*F);
}

int main(int argc, const char ** argv)
{
    // prng for filling positions
    std::mt19937_64 gen(SEED);
    std::uniform_real_distribution<real_t> rand_real(realval(0.0), realval(1.0));

    // positions and forces of atoms
    mds::darray R[NUM_ATOMS] = {realval(0.0)};
    mds::darray F[NUM_ATOMS] = {realval(0.0)};

    // use random to fill positions
    for (int i = 0; i < NUM_ATOMS; ++i)
    {
        R[i][0] = LX*rand_real(gen);
        R[i][1] = LY*rand_real(gen);
        R[i][2] = LZ*rand_real(gen);
    }
    
    // calculate total force on each atom
    for (int i = 0; i < NUM_ATOMS; ++i)
    {
        // sum into Fi
        mds::darray Fi = {0.0};

        // first part
        for (int j = 0; j < i; ++j)
            gradV(R[i],R[j],Fi);
        // remainder
        for (int j = i+1; j < NUM_ATOMS; ++j)
            gradV(R[i],R[j],Fi);

        // store Fi
        F[i][0] = Fi[0];
        F[i][1] = Fi[1];
        F[i][2] = Fi[2];
    }
    
    // print some output
    printf("Old: Positions || Forces || Force\n");
    for (int i = 0; i < NUM_ATOMS; ++i)
    {
        printf("%02d: %15e, %15e, %15e", i+1, R[i][0], R[i][1], R[i][2]);
        printf(" || %15e, %15e, %15e", F[i][0], F[i][1], F[i][2]);
        printf(" || %15e\n", sqrt(F[i][0]*F[i][0]+F[i][1]*F[i][1]+F[i][2]*F[i][2]));
    }
    printf("\n");

    // relax the positions by seeking lowest potential configuration
    for (int i = 0; i < MCSTEPS; ++i)
    {
        // potentially touch each particle per mcstep
        for (int j = 0; j < NUM_ATOMS; ++j)
        {
            // choose a particle
            int k = gen() % NUM_ATOMS;

            // shift it randomly
            mds::darray Rk = {
                R[k][0] + LX*(rand_real(gen)-realval(0.5)),
                R[k][1] + LY*(rand_real(gen)-realval(0.5)),
                R[k][2] + LZ*(rand_real(gen)-realval(0.5))
            };

            // put in the box
            Rk[0] = (Rk[0] >= LX) ? Rk[0] - LX : Rk[0];
            Rk[1] = (Rk[1] >= LY) ? Rk[1] - LY : Rk[1];
            Rk[2] = (Rk[2] >= LZ) ? Rk[2] - LZ : Rk[2];
            Rk[0] = (Rk[0] < realval(0.0)) ? Rk[0] + LX : Rk[0];
            Rk[1] = (Rk[1] < realval(0.0)) ? Rk[1] + LY : Rk[1];
            Rk[2] = (Rk[2] < realval(0.0)) ? Rk[2] + LZ : Rk[2];

            // calculate new and old potential
            real_t pnew = realval(0.0);
            real_t pold = realval(0.0);
            mds::darray fnew = {realval(0.0)};
            mds::darray fold = {realval(0.0)};

            // sum potential and gradient
            for (int l = 0; l < k; ++l)
            {
                pold += gradV(R[k],R[l],fold);
                pnew += gradV( Rk ,R[l],fnew);
            }
            for (int l = k+1; l < NUM_ATOMS; ++l)
            {
                pold += gradV(R[k],R[l],fold);
                pnew += gradV( Rk ,R[l],fnew);
            }

            // accept if potential is less
            if (pnew < pold)
            {
                R[k][0] = Rk[0];
                R[k][1] = Rk[1];
                R[k][2] = Rk[2];
            }
        }
    }

    // calculate total force on each atom
    for (int i = 0; i < NUM_ATOMS; ++i)
    {
        // sum into Fi
        mds::darray Fi = {realval(0.0)};

        // first part
        for (int j = 0; j < i; ++j)
            gradV(R[i],R[j],Fi);
        // remainder
        for (int j = i+1; j < NUM_ATOMS; ++j)
            gradV(R[i],R[j],Fi);

        // store Fi
        F[i][0] = Fi[0];
        F[i][1] = Fi[1];
        F[i][2] = Fi[2];
    }

    // print some output
    printf("New: Positions || Forces || Force\n");
    for (int i = 0; i < NUM_ATOMS; ++i)
    {
        printf("%02d: %15e, %15e, %15e", i+1, R[i][0], R[i][1], R[i][2]);
        printf(" || %15e, %15e, %15e", F[i][0], F[i][1], F[i][2]);
        printf(" || %15e\n", sqrt(F[i][0]*F[i][0]+F[i][1]*F[i][1]+F[i][2]*F[i][2]));
    }
    printf("\n");

    // initialize mdstresslib
    mds::StressGrid mds;
    
    mds.SetContribType(mds_all);
    mds.SetStressType(mds_spat);
    mds.SetForceDecomposition(mds_ccfd);
    
    mds.SetFileName("output.bin");
    mds.SetMaxCluster(int(NUM_ATOMS*(NUM_ATOMS-1)/2 + 1));
    mds.SetNumberOfAtoms(NUM_ATOMS);

    mds::dmatrix box = {{0}};
    box[0][0] = LX;
    box[1][1] = LY;
    box[2][2] = LZ;
    mds.SetBox(box);

    mds.SetNumberOfGridCellsX(4);
    mds.SetNumberOfGridCellsY(4);
    mds.SetNumberOfGridCellsZ(4);

    // initialize mdstress
    mds.Init();
    mds.UpdateBoxSpacings(box);

    // perform stress analysis
    mds.DistributeInteraction(NUM_ATOMS,R,F,NULL);
    mds.SumGrid();
    mds.Write();

    return 0;
}
