/*=========================================================================

  Module    : MDStress
  File      : mds_stressgrid.cpp
  Authors   : A. Torres-Sanchez and J. M. Vanegas
  Modified  :
  Purpose   : Compute the local stress from MD trajectories
  Date      : 25/03/2015
  Version   :
  Changes   :

     http://www.lacan.upc.edu/LocalStressFromMD

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  

     Please, report any bug to either of us:
     torres.sanchez.a@gmail.com
     juan.m.vanegas@gmail.com
=========================================================================*/

#include "mds_stressgrid.h"

using namespace mds;

//Constructor
StressGrid::StressGrid()
{
    this->nx         = 0;
    this->ny         = 0;
    this->nz         = 0;
    this->maxClust   = mds_maxpart;
    this->gridsp[0]  = 0.0;
    this->gridsp[1]  = 0.0;
    this->gridsp[2]  = 0.0;
    this->spacing    = 0.0;
    
    this->spatatom = 0;
    this->fdecomp  = 0;
    this->contrib  = 0;
    
    this->ierr     = 0;

    this->nframes  = 0;
    this->nreset   = 0;
    
    for ( int i = 0; i < mds_ndim; i++ )
    {
        for ( int j = 0; j < mds_ndim; j++ )
        {
            this->box   [i][j]  = 0.0;
            this->sumbox[i][j]  = 0.0;
            this->invbox[i][j]  = 0.0;
        }
        this->gridsp[i] = 0.0;
    }

    this->invgridsp = 0.0;

    this->current_grid   = NULL;
    this->sum_grid       = NULL;
    this->lapack         = NULL;
    this->Amat           = NULL;
    this->AmatT          = NULL;
    this->bvec           = NULL;
    this->R_ij           = NULL;
}

//Destructor
StressGrid::~StressGrid()
{    
    this->Clear();
}

// Method to delete the preallocated member variables
void StressGrid::Clear()
{
    if (this->current_grid != NULL ) delete [] this->current_grid;
    if (this->sum_grid     != NULL ) delete [] this->sum_grid;
    if (this->Amat         != NULL ) delete [] this->Amat;
    if (this->AmatT        != NULL ) delete [] this->AmatT;
    if (this->bvec         != NULL ) delete [] this->bvec;
    if (this->R_ij         != NULL ) delete [] this->R_ij;
    if (this->lapack       != NULL ) delete    this->lapack;

    this->current_grid = NULL; 
    this->sum_grid     = NULL;
    this->Amat         = NULL;
    this->AmatT        = NULL;
    this->bvec         = NULL;
    this->R_ij         = NULL;
    this->lapack       = NULL;
}

// This function is provided to identify bad settings
int StressGrid::CheckSettings()
{
    if ( this->spatatom != mds_spat && this->spatatom != mds_atom )
    {
        std::cout << "ERROR::StressGrid: The stress type (SetStressType) should be either spatial (0) or atomic (1)\n";
        return 1;
    }
    if ( this->spatatom == mds_spat )
    {
        if ( this->nx < 0 || this->ny < 0 || this->nz < 0)
        {
            std::cout << "ERROR::StressGrid: The number of grid cells are negative: (nx,ny,nz)=" << " ( " <<this->nx << ", " << this->ny << ", " <<this->nz << " )\n"; 
            return 2;
        }
        
        if ( this->nx == 0 && this->ny == 0 && this->nz == 0)
        {
            if ( this->spacing < mds_eps )
            {
                std::cout << "ERROR::StressGrid: The local spacing is too small: " << this->spacing << "( < " << mds_eps << " )\n"; 
                return 3;
            }
            
            if ( iszeromatrix(this->box) )
            {
                std::cout << "ERROR::StressGrid: The initial MD box haven't been set or it's too small\n"; 
                return 4;
            }
        }
        if ( this->fdecomp < mds_ccfd || this->fdecomp > mds_gld )
        {
            std::cout << "ERROR::StressGrid: The stress type (SetForceDecomposition) should be: 0 (cCFD), 1 (nCFD), 2 (GLD) or 3 (GMC)\n";
            return 5;
        }
    }
    else
    {
        if ( this->nAtoms <= 0 )
        {
            std::cout << "ERROR::StressGrid: Number of atoms must > 0\n";
            return 6;
        }
    }
    if (  this->contrib < mds_sl || this->contrib > mds_cmp )
    {
        std::cout << "ERROR::StressGrid: Contribution label should be between " << mds_sl << " and " << mds_cmp << "\n";
        return 7;
    }

    if ( this->filename.size() == 0 )
    {
        std::cout << "ERROR::StressGrid: Filename for output is not set (SetFileName)" << mds_sl << " and " << mds_cmp << "\n";
        return 8;
    }
    
    if ( this->maxClust <= 1 )
    {
        std::cout << "ERROR:StressGrid: The maximum number of particles in a cluster is not valid:" << this->maxClust << "\n";
    }
    this->Clear();
    
    return 0;
}

//This function initialize the grid depending on the settings. If the settings are incorrect, 
//it throws an error
void StressGrid::Init()
{
    //First call checksettings to check if all parameters are OK
    this->ierr = this->CheckSettings();

    if ( !ierr )
    {
        //Sizes
        if (this->spatatom == mds_spat)
        {
            if(this->nx == 0)   this->nx = static_cast<int>(box[0][0]/this->spacing);
            if(this->ny == 0)   this->ny = static_cast<int>(box[1][1]/this->spacing);
            if(this->nz == 0)   this->nz = static_cast<int>(box[2][2]/this->spacing);
            
            this->ncells = this->nx*this->ny*this->nz;

            if(this->nx==0)  this->nx=1;
            if(this->ny==0)  this->ny=1;
            if(this->nz==0)  this->nz=1;
        }
        else
            this->ncells = this->nAtoms;

        //Give size to current and sum grid
        this->sum_grid     = new dmatrix [this->ncells];
        this->current_grid = new dmatrix [this->ncells];
        
        //Set all to zero
        for( int i=0; i<this->ncells; i++ )
        {
            zeromatrix ( this->sum_grid[i]     );
            zeromatrix ( this->current_grid[i] );
        }

        this->nframes = 0;
        
        //If the maximum cluster is larger than 5 we need special structures. Create them:
        if (this->maxClust > 5 )
        {
            int maxrows, maxcols;
            maxrows = mds_ndim*this->maxClust;
            maxcols = (this->maxClust*(this->maxClust-1))/2;
            
            this->Amat  = new double [maxrows*maxcols];
            
            if ( this->fdecomp == mds_ccfd )
                this->AmatT = new double [maxrows*maxcols]; 
            
            this->bvec  = new double [mds_max(maxrows,maxcols)];
            
            this->R_ij  = new darray [maxrows];
        }
        
        // Finally, create the lapack object to deal with linear solvers and projections
        this->lapack = new Lapack (mds_ndim*this->maxClust,(this->maxClust*(this->maxClust-1))/2);
        
        this->nframes = -1;
        this->Update();
    }
}

// This function updates the box, sum the current grid to sum_grid and sets current_grid
// to zero. It also computes the new spacings.
void StressGrid::Update ( )
{
    if ( !ierr )
    {
        for( int i=0; i<this->ncells; i++ )
            summatrix( this->sum_grid[i], this->current_grid[i], this->sum_grid[i] );

        // Update the inverse of the box
        inversematrix( this->box, this->invbox );

        this->gridsp[0] = this->box[0][0]/static_cast<double>(this->nx);
        this->gridsp[1] = this->box[1][1]/static_cast<double>(this->ny);
        this->gridsp[2] = this->box[2][2]/static_cast<double>(this->nz);
        
        this->invgridsp = 1.0/(this->gridsp[0]*this->gridsp[1]*this->gridsp[2]);
        
        summatrix( this->box, this->sumbox, this->sumbox );

        // Set the current grid to 0
        for( int i=0; i<this->ncells; i++ )
            zeromatrix ( this->current_grid[i] );
        
        this->nframes ++;
    }
}

//Set both sum_grid and current_grid to zero. Sum the number of resets (this is used for 
//printing files) and set the number of frames to zero
void StressGrid::Reset ( )
{
    if ( !ierr )
    {
        for( int i=0; i<this->ncells; i++ )
        {
            zeromatrix ( this->sum_grid[i]     );
            zeromatrix ( this->current_grid[i] );
        }
        this->nframes = 0;
        
        this->nreset ++;
    }
}


//Writes file with average stress to grid using the filename set by the user
void StressGrid::Write ( )
{
    if ( !ierr )
    {
        int                trash=1;
        dmatrix            avgbox;
        std::string        outname;
        std::ostringstream outnumber;
        FILE              *outfile;
        double             factor;
        
        outnumber << this->nreset;
        
        outname = this->filename + outnumber.str() + "." + mds_fileext;

        outfile = fopen(outname.c_str(), "wb" );
        
        fwrite(&trash, sizeof(int), 1, outfile);
        
        //Divide sumbox with respect to the number of frames to get the avg
        scalematrix( this->sumbox, 1.0/this->nframes, avgbox);

        fwrite(avgbox, sizeof(dmatrix), 1, outfile);

        fwrite(&this->nx, sizeof(this->nx), 1, outfile);
        fwrite(&this->ny, sizeof(this->ny), 1, outfile);
        fwrite(&this->nz, sizeof(this->nz), 1, outfile);

        factor = mds_units/this->nframes;
        
        for ( int i = 0; i < this->ncells; i++ )
            scalematrix(this->sum_grid[i], factor, this->sum_grid[i]);
        
        fwrite(this->sum_grid, sizeof(dmatrix), this->ncells, outfile);
        
        fclose(outfile);
        
    }
}

//----------------------------------------------------------------------------------------
// DistributeInteraction
//
// ROOT OF ALL EVIL
// This function reads the number of atoms, their
// respective positions and forces, and the atom IDs, and calls the functions in charge of distributing the stress
// on the grid depending on the local stress flags and the kind of interaction
// Requires:
// nAtoms  -> number of atoms of the contribution
// R       -> positions of the atoms
// F       -> forces on the atoms
// atomIDs -> labels of the atoms
void StressGrid::DistributeInteraction(int nAtoms, darraylist R, darraylist F, int *atomIDs = NULL)
{
    int    n;
    int    i,j;
    double temp;

    dmatrix stress;

    if ( nAtoms > this->maxClust )
    {
        std::cout << "ERROR::StressGrid: Distribute Interaction has been called with a number of atoms larger than the maximum cluster size previously set, nAtoms=" << nAtoms << " and maxClust=" << this->maxClust << "\n";
        this->ierr = 8;
    }
    
    if ( !ierr )
    {
        // If spatatom==mds_spat distribute the stress spatially following Noll's procedure
        if (this->spatatom == mds_spat)
        {
            // Depending on the number of atoms, call a different function
            switch (nAtoms)
            {

                case 2:
                    this->DistributePairInteraction( R[0], R[1], F[0] );
                    break;

                case 3:
                    this->DistributeN3( R[0], R[1], R[2],F[0], F[1], F[2] );
                    break;

                case -3:
                    this->DistributeSettle( R[0], R[1], R[2],F[0], F[1], F[2] );
                    break;

                case 4:
                    this->DistributeN4( R[0], R[1], R[2], R[3], F[0], F[1], F[2], F[3] );
                    break;

                case 5:
                    this->DistributeN5( R[0], R[1], R[2], R[3], R[4], F[0], F[1], F[2], F[3], F[4] );
                    break;

                default:
                    this->DistributeNBody( nAtoms, R, F );
                    break;
            }

        }
        // If spatatom==mds_atom, distributes the stress per atom using the atomic stress definition
        else if (this->spatatom == mds_atom)
        {
            // This is because SETTLE calls the function with nAtoms=-3
            if (nAtoms < 0) nAtoms = -nAtoms;

            if (atomIDs == NULL)
            {
                std::cout << "ERROR:: the atomIDs array is NULL. Cannot calculate the stress/atom.";
                return;
            }

            for (n = 0; n < nAtoms; n++ )
            {
                if ( atomIDs[n] >= this->nAtoms )
                {
                    std::cout << "ERROR:: the atom label" << atomIDs[n] << "is equal or larger than the total number of atoms" << this->nAtoms;
                    return;
                }
            }

            //Initialize the value of the (local) stress to 0
            for( i = 0; i< mds_ndim; i++ )
            {
                stress[i][i] = 0.0;
                for( j=i+1; j< mds_ndim; j++ )
                {
                    stress[i][j] = 0.0;
                    stress[j][i] = 0.0;
                }
            }


            //Calculate the stress
            for( n = 0; n < nAtoms; n++ )
            {
                for( i = 0; i< mds_ndim; i++ )
                {
                    temp = -(F[n][i] * R[n][i])/nAtoms;
                    stress[i][i] += temp;
                    for( j=i+1; j< mds_ndim; j++ )
                    {
                        temp = -(F[n][i] * R[n][j])/nAtoms;
                        stress[i][j] += temp;
                        stress[j][i] += temp;
                    }
                }
            }

            for ( n = 0; n < nAtoms; n++ )
            {
                AddAtomStressToGrid(atomIDs[n], stress);
            }
        }
    }

    return;
}

//----------------------------------------------------------------------------------------
// DistributePairInteraction
//
// Distributes interactions onto locals_grid (from the initial grid point to the last grid point)
// Requires:
// xi   -> position of particle I (A)
// xj   -> position of particle J (B)
// F    -> pairwise force
void StressGrid::DistributePairInteraction( darray xi, darray xj, darray F )
{
    int nDim = 3;

    double oldt, sumfactor;

    darray curpt, t, d_cgrid, diff;

    int  i, j, k, ii, jj, kk; //counters
    bool tof;                  //true or false

    iarray i1; //grid cell corresponding to particle I (A)
    iarray i2; //grid cell corresponding to particle J (B)
    iarray x;  //cell during spreding
    iarray xn; //next cell during spreading
    iarray c;  //director
    
    double gridp;
    
    dmatrix stress;

    int timesInLoop; //this is to avoid that the program gets stacked in an infinite loop

    //------------------------------------------------------------------------------------
    // Calculate the stress tensor

    // difference between xj and xi
    diffarray ( xj, xi, diff);

    stress[0][0] = F[0]*diff[0];
    stress[1][0] = F[1]*diff[0];
    stress[2][0] = F[2]*diff[0];
    stress[0][1] = F[0]*diff[1];
    stress[1][1] = F[1]*diff[1];
    stress[2][1] = F[2]*diff[1];
    stress[0][2] = F[0]*diff[2];
    stress[1][2] = F[1]*diff[2];
    stress[2][2] = F[2]*diff[2];

    
    //------------------------------------------------------------------------------------

    //------------------------------------------------------------------------------------
    //Distribute the stress

    // Calculate the grid coordinates (no pbc) for the extreme points
    this->GridCoord(xi, &i1[0], &i1[1], &i1[2]);
    this->GridCoord(xj, &i2[0], &i2[1], &i2[2]);

    // d_cgrid = vector from the center of the present cell to the initial point
    // c is a vector that guide the advance in each coordinate (+1 if it has to advance in this coordinate, -1 if it has to go back or 0 if it has to do nothing)
    for(i = 0; i < nDim; i++) 
    {
        d_cgrid[i] = xi[i]-(i1[i]+0.5)*this->gridsp[i];
        if(i2[i]>i1[i])
        {
            c[i] = 1;
        } 
        else if(i1[i]>i2[i])
        {
            c[i]=-1;
        } else
        {
            c[i] = 0;
        }
    }

    // First cross point with aplane (if there is at least one i.e. c[i] != 0)
    for(i = 0; i < nDim; i++ )
    {
        if(c[i]==1)
        {
            xn[i] = i1[i]+1;                    //label of the next cell is 1 step further than the previous in this direction
            gridp = xn[i] * this->gridsp[i];    //position of the cell
            t[i] = (xi[i]-gridp)/(xi[i]-xj[i]); //parametric time of crossing
        } 
        else if(c[i]==-1)
        {
            xn[i] = i1[i];                      //label of the next cell is the same in this direction
            gridp = xn[i] * this->gridsp[i];    //position of the cell
            t[i] = (xi[i]-gridp)/(xi[i]-xj[i]); //parametric time of crossing
        } 
        else 
            t[i]=1.1;                           //This sets the time larger than 1 because there is no crossing

        x[i] = i1[i];

    }
    
    oldt      = 0.0; //previous parametric time of crossing
    sumfactor = 0.0; //This is just to check that the contribution has been completely spread

    // While we don't reach the last point...

    tof = (c[0]*x[0]<c[0]*i2[0]+mds_eps)||(c[1]*x[1]<c[1]*i2[1]+mds_eps)||(c[2]*x[2]<c[2]*i2[2]+mds_eps);

    timesInLoop = 0;

    while(tof)
    {

        if(t[0]<t[1]+mds_eps && t[0]<t[2]+mds_eps)
        {
            // Distribute the contribution
            this->SpreadLineSource(diff,d_cgrid,oldt,t[0],x,stress,&sumfactor);
            // Move
            d_cgrid[0] -= c[0] * gridsp[0];
            oldt = t[0];
            
            x[0] += c[0];
            xn[0] += c[0];

            // Next cross point:
            gridp = xn[0] *gridsp[0];
            t[0] = (xi[0]-gridp)/(xi[0]-xj[0]);
        } 
        else if (t[1]<t[0]+mds_eps && t[1]<t[2]+mds_eps)
        {
            // Distribute the contribution
            this->SpreadLineSource(diff,d_cgrid,oldt,t[1],x,stress,&sumfactor);

            // Move
            d_cgrid[1] -= c[1] * gridsp[1];
            oldt = t[1];

            x[1]+=c[1];
            xn[1]+=c[1];

            // Next cross point:
            gridp = xn[1] * gridsp[1];
            t[1] = (xi[1]-gridp)/(xi[1]-xj[1]);

        } 
        else if (t[2]<t[0]+mds_eps && t[2]<t[1]+mds_eps)
        {
            // Distribute the contribution
            this->SpreadLineSource(diff,d_cgrid,oldt,t[2],x,stress,&sumfactor);

            // Move
            d_cgrid[2] -= c[2] * gridsp[2];
            oldt  = t[2];

            x[2]+=c[2];
            xn[2]+=c[2];

            // Next cross point:
            gridp = xn[2] * gridsp[2];
            t[2] = (xi[2]-gridp)/(xi[2]-xj[2]);
        }
        else
        {
            //printf("ERROR::gmx_spread_local_stress_on_grid: t=(%lf, %lf, %lf), i1=(%d, %d, %d), i2=(%d, %d, %d), x=(%d, %d, %d)\n",t[0], t[1], t[2], i1[0], i1[1], i1[2], i2[0], i2[1], i2[2], x[0], x[1], x[2]);
            return;
        }

        tof = (c[0]*x[0]<c[0]*i2[0])||(c[1]*x[1]<c[1]*i2[1])||(c[2]*x[2]<c[2]*i2[2]);
        timesInLoop ++;
        
        if(timesInLoop > 10000000) //To avoid infinite loops
        {
            return;
        }
    }

    // Distribute the last contribution

    this->SpreadLineSource(diff,d_cgrid,oldt,1,x,stress,&sumfactor);

    //------------------------------------------------------------------------------------
}

//----------------------------------------------------------------------------------------
// DistributeKinetic
//
// Distributes kinetic contributions onto the grid
// Requires:
// mass       -> mass of the particle
// x          -> position of the atom
// va         -> velocity of the particle at time t for Vel-verlet, or t-dt/2 for leapfrog integrators
// vb         -> velocity of the particle at t+dt/2 (for leapfrog integrators only)
// atomID     -> ID of the atom
//
// For leapfrog integrators we know va(t-dt/2) and vb(t+dt/2), but we want the contribution at v(t) which we don't know.
// So we take the average kinetic contribution from the velocities at each half step -m*(va(t-dt/2)^2 + vb(t+dt/2)^2)/2
// Warning! this is not the same as simply taking the average of the half-step velocities, which would be incorrect.
//
// For velocity-verlet integrators we know v at the same time step, t, as the positions so the contribution is -m*va(t)*va(t)

void StressGrid::DistributeKinetic(double mass, darray x, darray va, darray vb = NULL, int atomID = -1)
{

    dmatrix stress;

    if (ierr != 0)
    {
        if (vb == NULL)
        {
            stress[0][0] = -mass*va[0]*va[0];
            stress[0][1] = -mass*va[0]*va[1];
            stress[0][2] = -mass*va[0]*va[2];
            stress[1][0] = -mass*va[1]*va[0];
            stress[1][1] = -mass*va[1]*va[1];
            stress[1][2] = -mass*va[1]*va[2];
            stress[2][0] = -mass*va[2]*va[0];
            stress[2][1] = -mass*va[2]*va[1];
            stress[2][2] = -mass*va[2]*va[2];
        }
        else
        {
            stress[0][0] = -mass*(va[0]*va[0] + vb[0]*vb[0])/2;
            stress[0][1] = -mass*(va[0]*va[1] + vb[0]*vb[1])/2;
            stress[0][2] = -mass*(va[0]*va[2] + vb[0]*vb[2])/2;
            stress[1][0] = -mass*(va[1]*va[0] + vb[1]*vb[0])/2;
            stress[1][1] = -mass*(va[1]*va[1] + vb[1]*vb[1])/2;
            stress[1][2] = -mass*(va[1]*va[2] + vb[1]*vb[2])/2;
            stress[2][0] = -mass*(va[2]*va[0] + vb[2]*vb[0])/2;
            stress[2][1] = -mass*(va[2]*va[1] + vb[2]*vb[1])/2;
            stress[2][2] = -mass*(va[2]*va[2] + vb[2]*vb[2])/2;
        }
        // If spatatom==mds_spat distribute the stress spatially following Noll's procedure
        if (this->spatatom == mds_spat)
        {
            SpreadPointSource(x, stress);
        }
        else if (this->spatatom == mds_atom)
        {
            if (atomID == -1)
            {
                std::cout << "ERROR:: Unknown atomID for kinetic contribution. Cannot calculate the stress/atom.";
                return;
            }
            else
            {
                AddAtomStressToGrid(atomID,stress);
            }
        }
    }
}

//----------------------------------------------------------------------------------------
// SpreadLineSource
//
// Distributes "line sources" onto a grid point (asuming trilinear weighting functions!) 
// Requires:
// a      -> position of particle A
// b      -> position of particle B
// t1     -> starting parametric time (the line segment is parametrized from 0 to 1 from A to B)
// t2     -> ending  parametric time 
// x      -> position of the grid point (integers) where we are spreading the contribution
// stress -> stress to be spread
// sumfactor -> (this is just for monitoring) part of the line source that has already been spread
void StressGrid::SpreadLineSource(darray a, darray b, double t1, double t2, iarray x, dmatrix stress, double *sumfactor)
{
    int i,j,k;
    int ii,jj,kk;
    int gridcell;
    double factor;
    double dummy1,dummy2,dummy3,dummy4,dummy5,dummy6,dummy7,dummy8,dummy9,dummy10,dummy11,dummy12;
    double t12,t22,t13,t23;

    dmatrix partial_stress;

    t12 = t1*t1;
    t13 = t12*t1;
    t22 = t2*t2;
    t23 = t22*t2;

    ii=x[0]; jj=x[1]; kk=x[2];

    dummy1 = -2*a[0]*a[1]*a[2]*t12*t12;
    dummy2 =  2*a[0]*a[1]*a[2]*t22*t22;

    for(i=1;i>=-1;i-=2)
    {
        ii+=i;
        dummy3 = 2*b[0]+i*this->gridsp[0];
        dummy7 = a[1]*a[2]*dummy3;

        for(j=1;j>=-1;j-=2)
        {
            jj+=j;
            dummy4 = 2*b[1]+j*this->gridsp[1];
            dummy6 = dummy3*dummy4;
            dummy8 = a[0]*a[2]*dummy4;
            dummy10= a[2]*dummy3*dummy4;

            for(k=1;k>=-1;k-=2)
            {
                kk+=k;
                dummy5 = 2*b[2]+k*this->gridsp[2];
                dummy9 = a[1]*a[0]*dummy5;
                dummy11= a[1]*dummy3*dummy5;
                dummy12= a[0]*dummy4*dummy5;
                factor = i*j*k*0.125*this->invgridsp*this->invgridsp*(dummy1+dummy2+(t2-t1)*dummy6*dummy5+1.333333333333*(t23-t13)
                            *(dummy7+dummy8+dummy9)+(t22-t12)*(dummy10+dummy11+dummy12));

                *sumfactor=*sumfactor+factor;
                
                scalematrix(stress,factor,partial_stress);
    
                gridcell = modulo(ii,nx)*nz*ny+modulo(jj,ny)*nz+modulo(kk,nz);
                
                this->AddAtomStressToGrid (gridcell, partial_stress);
            }
        }
    }

}


//----------------------------------------------------------------------------------------
// SpreadPointSource
//
// Distributes "point sources" onto the grid
// Requires:
// x      -> source point 
// stress -> stress to be distributed
void StressGrid::SpreadPointSource( darray pt, dmatrix stress )
{

    // Spreads the velocity in one point

    int i, j, k, ii, jj, kk, iii, jjj, kkk, nx, ny, nz;
    dmatrix part_stress;
    double factor, invgridsp, dummy1, dummy2;
    int gridcell;
    
    // Get the coordinates of the point in the grid
    this->GridCoord(pt,&ii,&jj,&kk);
    iii=ii;
    jjj=jj;
    kkk=kk;

    // Spread it
    for(i=1;i>=-1;i-=2)
    {
        iii+=i;
        dummy1 = i * invgridsp * invgridsp * (pt[0]-(ii+0.5*(1-i))*gridsp[0]);
        for(j=1;j>=-1;j-=2)
        {
            jjj+=j;
            dummy2 = dummy1 * j * (pt[1]-(jj+0.5*(1-j))*gridsp[1]);
            for(k=1;k>=-1;k-=2)
            {
                kkk+=k;
                factor = dummy2 * k * (pt[2]-(kk+0.5*(1-k))*gridsp[2]);
                scalematrix(stress,factor,part_stress);
                gridcell = modulo(iii,nx)*nz*ny+modulo(jjj,ny)*nz+modulo(kkk,nz);
                this->AddAtomStressToGrid( gridcell,part_stress );
            }
        }
    }
}

//----------------------------------------------------------------------------------------
//SPECIFIC DECOMPOSITIONS FOR 3,4 AND 5 AND N PARTICLES


// Decompose 3-body potentials (angles)
void StressGrid::DistributeN3( darray Ra, darray Rb, darray Rc, darray Fa, darray Fb, darray Fc )
{

    //Counter
    int i;

    //************************************************************************************
    // UNIT vectors between particles
    darray AB, AC, BC;

    // Distances
    double normAB,normAC,normBC;

    // (Covariant) Central Force decomposition
    double lab, lac, lbc;

    darray Fij;
    //************************************************************************************

    //Dimension and number of particles
    int nDim = 3;
    int nPart = 3;

    //Number of rows and columns
    int nRow = mds_nrow3, nCol = mds_ncol3, nRHS = 1;

    // Matrix of the system (9 equations x 3 unknowns)
    double M[mds_nrow3*mds_ncol3];
    // Vector, we want to solve M*x = b
    double b[mds_nrow3], s[mds_ncol3];

    // If the force decomposition is cCFD or CFD
    if(this->fdecomp == mds_ccfd || this->fdecomp == mds_ncfd)
    {
        diffarray(Rb, Ra, AB);
        diffarray(Rc, Ra, AC);
        diffarray(Rc, Rb, BC);

        normAB=normarray(AB);
        normAC=normarray(AC);
        normBC=normarray(BC);

        for ( i = 0; i < nCol*nRow; i++ )
        {
            M[i] = 0.0;
        }

        //Force on particle 1:
        M[nRow*0+0] = AB[0]; M[nRow*1+0] = AC[0];
        M[nRow*0+1] = AB[1]; M[nRow*1+1] = AC[1];
        M[nRow*0+2] = AB[2]; M[nRow*1+2] = AC[2];
        b[0] = Fa[0]; b[1] = Fa[1]; b[2] = Fa[2];

        //Force on particle 2:
        M[nRow*0+3] = -AB[0]; M[nRow*2+3] = BC[0];
        M[nRow*0+4] = -AB[1]; M[nRow*2+4] = BC[1];
        M[nRow*0+5] = -AB[2]; M[nRow*2+5] = BC[2];
        b[3] = Fb[0]; b[4] = Fb[1]; b[5] = Fb[2];

        //Force on particle 3:
        M[nRow*1+6] = -AC[0]; M[nRow*2+6] = -BC[0];
        M[nRow*1+7] = -AC[1]; M[nRow*2+7] = -BC[1];
        M[nRow*1+8] = -AC[2]; M[nRow*2+8] = -BC[2];
        b[6] = Fc[0]; b[7] = Fc[1]; b[8] = Fc[2];
       
        if ( this->lapack->SolveMinNorm(nRow, nCol, M, b) )
        {
            std::cout << "ERROR::StressGrid: LAPACK solver fails\n";
            this->ierr = 10;
        }

        // Sum the 3 contributions to the stress
        lab = b[0];
        lac = b[1];
        lbc = b[2];

        Fij[0] = lab * AB[0]; Fij[1] = lab * AB[1]; Fij[2] = lab * AB[2];
        this->DistributePairInteraction(Ra, Rb, Fij);
        Fij[0] = lac * AC[0]; Fij[1] = lac * AC[1]; Fij[2] = lac * AC[2];
        this->DistributePairInteraction(Ra, Rc, Fij);
        Fij[0] = lbc * BC[0]; Fij[1] = lbc * BC[1]; Fij[2] = lbc * BC[2];
        this->DistributePairInteraction(Rb, Rc, Fij);

    }
    else if (this->fdecomp == mds_gld)
    {
        Fij[0] = (Fa[0]-Fb[0])/3.0; Fij[1] = (Fa[1]-Fb[1])/3.0; Fij[2] = (Fa[2]-Fb[2])/3.0;
        this->DistributePairInteraction(Ra, Rb, Fij);
        Fij[0] = (Fa[0]-Fc[0])/3.0; Fij[1] = (Fa[1]-Fc[1])/3.0; Fij[2] = (Fa[2]-Fc[2])/3.0;
        this->DistributePairInteraction(Ra, Rc, Fij);
        Fij[0] = (Fb[0]-Fc[0])/3.0; Fij[1] = (Fb[1]-Fc[1])/3.0; Fij[2] = (Fb[2]-Fc[2])/3.0;
        this->DistributePairInteraction(Rb, Rc, Fij);
    }
    
}

// Decompose Settle
void StressGrid::DistributeSettle( darray Ra, darray Rb, darray Rc, darray Fa, darray Fb, darray Fc )
{
    //Counter
    int i;

    //************************************************************************************
    // UNIT vectors between particles
    darray AB, AC, BC;

    // Distances
    double normAB,normAC,normBC;

    // (Covariant) Central Force decomposition
    double lab, lac, lbc;

    darray Fij;
    //************************************************************************************

    //Dimension and number of particles
    int nDim = 3;
    int nPart = 3;

    //Number of rows and columns
    int nRow = mds_nrow3, nCol = mds_ncol3, nRHS = 1;

    // Matrix of the system (12 equations x 6 unknowns)
    double M[mds_nrow3*mds_ncol3];
    // Vector, we want to solve M*x = b
    double b[mds_nrow3], s[mds_ncol3];

    if (this->fdecomp == mds_ccfd || this->fdecomp == mds_ncfd || this->fdecomp == mds_gld )
    {
        diffarray(Rb, Ra, AB);
        diffarray(Rc, Ra, AC);
        diffarray(Rc, Rb, BC);

        normAB=normarray(AB);
        normAC=normarray(AC);
        normBC=normarray(BC);

        for ( i = 0; i < nCol*nRow; i++ )
        {
            M[i] = 0.0;
        }

        //Force on particle 1:
        M[nRow*0+0] = AB[0]; M[nRow*1+0] = AC[0];
        M[nRow*0+1] = AB[1]; M[nRow*1+1] = AC[1];
        M[nRow*0+2] = AB[2]; M[nRow*1+2] = AC[2];
        b[0] = Fa[0]; b[1] = Fa[1]; b[2] = Fa[2];

        //Force on particle 2:
        M[nRow*0+3] = -AB[0]; M[nRow*2+3] = BC[0];
        M[nRow*0+4] = -AB[1]; M[nRow*2+4] = BC[1];
        M[nRow*0+5] = -AB[2]; M[nRow*2+5] = BC[2];
        b[3] = Fb[0]; b[4] = Fb[1]; b[5] = Fb[2];

        //Force on particle 3:
        M[nRow*1+6] = -AC[0]; M[nRow*2+6] = -BC[0];
        M[nRow*1+7] = -AC[1]; M[nRow*2+7] = -BC[1];
        M[nRow*1+8] = -AC[2]; M[nRow*2+8] = -BC[2];
        b[6] = Fc[0]; b[7] = Fc[1]; b[8] = Fc[2];

        if ( this->lapack->SolveMinNorm(nRow, nCol, M, b) )
        {
            std::cout << "ERROR::StressGrid: LAPACK solver fails\n";
            this->ierr = 10;
        }

        // Sum the 3 contributions to the stress
        lab = b[0];
        lac = b[1];
        lbc = b[2];

        Fij[0] = lab * AB[0]; Fij[1] = lab * AB[1]; Fij[2] = lab * AB[2];
        this->DistributePairInteraction( Ra, Rb, Fij);
        Fij[0] = lac * AC[0]; Fij[1] = lac * AC[1]; Fij[2] = lac * AC[2];
        this->DistributePairInteraction( Ra, Rc, Fij );
        Fij[0] = lbc * BC[0]; Fij[1] = lbc * BC[1]; Fij[2] = lbc * BC[2];
        this->DistributePairInteraction( Rb, Rc, Fij );

    }
}

// Decompose 4-body potentials (dihedrals)
void StressGrid::DistributeN4( darray Ra, darray Rb, darray Rc, darray Rd, darray Fa, darray Fb, darray Fc, darray Fd )
{
    //Counter
    int i;

    //************************************************************************************
    // UNIT vectors between particles
    darray AB, AC, AD, BC, BD, CD;

    // Distances
    double normAB,normAC,normAD,normBC,normBD,normCD;

    // (Covariant) Central Force decomposition
    double lab, lac, lad, lbc, lbd, lcd;

    darray Fij;
    //************************************************************************************

    //Dimension and number of particles
    int nDim = 3;
    int nPart = 4;

    //Number of rows and columns
    int nRow = mds_nrow4, nCol = mds_ncol4, nRHS = 1;

    // Matrix of the system (12 equations x 6 unknowns)
    double M[mds_nrow4*mds_ncol4];
    // Vector, we want to solve M*x = b
    double b[mds_nrow4], s[mds_ncol4];

    // If the force decomposition is cCFD or CFD
    if(this->fdecomp == mds_ccfd || this->fdecomp == mds_ncfd)
    {
        diffarray(Rb, Ra, AB);
        diffarray(Rc, Ra, AC);
        diffarray(Rd, Ra, AD);
        diffarray(Rc, Rb, BC);
        diffarray(Rd, Rb, BD);
        diffarray(Rd, Rc, CD);

        normAB=normarray(AB);
        normAC=normarray(AC);
        normAD=normarray(AD);
        normBC=normarray(BC);
        normBD=normarray(BD);
        normCD=normarray(CD);

        for ( i = 0; i < nCol*nRow; i++ )
        {
            M[i] = 0.0;
        }
        //Force on particle 1:
        M[nRow*0+0] = AB[0]; M[nRow*1+0] = AC[0]; M[nRow*2+0] = AD[0];
        M[nRow*0+1] = AB[1]; M[nRow*1+1] = AC[1]; M[nRow*2+1] = AD[1];
        M[nRow*0+2] = AB[2]; M[nRow*1+2] = AC[2]; M[nRow*2+2] = AD[2];
        b[0] = Fa[0]; b[1] = Fa[1]; b[2] = Fa[2];

        //Force on particle 2:
        M[nRow*0+3] = -AB[0]; M[nRow*3+3] = BC[0]; M[nRow*4+3] = BD[0];
        M[nRow*0+4] = -AB[1]; M[nRow*3+4] = BC[1]; M[nRow*4+4] = BD[1];
        M[nRow*0+5] = -AB[2]; M[nRow*3+5] = BC[2]; M[nRow*4+5] = BD[2];
        b[3] = Fb[0]; b[4] = Fb[1]; b[5] = Fb[2];

        //Force on particle 3:
        M[nRow*1+6] = -AC[0]; M[nRow*3+6] = -BC[0]; M[nRow*5+6] = CD[0];
        M[nRow*1+7] = -AC[1]; M[nRow*3+7] = -BC[1]; M[nRow*5+7] = CD[1];
        M[nRow*1+8] = -AC[2]; M[nRow*3+8] = -BC[2]; M[nRow*5+8] = CD[2];
        b[6] = Fc[0]; b[7] = Fc[1]; b[8] = Fc[2];

        //Force on particle 4:
        M[nRow*2+9]  = -AD[0]; M[nRow*4+9] = -BD[0]; M[nRow*5+9] = -CD[0];
        M[nRow*2+10] = -AD[1]; M[nRow*4+10] = -BD[1]; M[nRow*5+10] = -CD[1];
        M[nRow*2+11] = -AD[2]; M[nRow*4+11] = -BD[2]; M[nRow*5+11] = -CD[2];
        b[9] = Fd[0]; b[10] = Fd[1]; b[11] = Fd[2];

        if ( this->lapack->SolveMinNorm(nRow, nCol, M, b) )
        {
            std::cout << "ERROR::StressGrid: LAPACK solver fails\n";
            this->ierr = 10;
        }

        // Sum the 6 contributions to the stress
        
        lab = b[0];
        lac = b[1];
        lad = b[2];
        lbc = b[3];
        lbd = b[4];
        lcd = b[5];

        Fij[0] = lab * AB[0]; Fij[1] = lab * AB[1]; Fij[2] = lab * AB[2];
        this->DistributePairInteraction(Ra, Rb, Fij);
        Fij[0] = lac * AC[0]; Fij[1] = lac * AC[1]; Fij[2] = lac * AC[2];
        this->DistributePairInteraction(Ra, Rc, Fij);
        Fij[0] = lad * AD[0]; Fij[1] = lad * AD[1]; Fij[2] = lad * AD[2];
        this->DistributePairInteraction(Ra, Rd, Fij);
        Fij[0] = lbc * BC[0]; Fij[1] = lbc * BC[1]; Fij[2] = lbc * BC[2];
        this->DistributePairInteraction(Rb, Rc, Fij);
        Fij[0] = lbd * BD[0]; Fij[1] = lbd * BD[1]; Fij[2] = lbd * BD[2];
        this->DistributePairInteraction(Rb, Rd, Fij);
        Fij[0] = lcd * CD[0]; Fij[1] = lcd * CD[1]; Fij[2] = lcd * CD[2];
        this->DistributePairInteraction(Rc, Rd, Fij);

    }
    else if (this->fdecomp == mds_gld)
    {
        Fij[0] = (Fa[0]-Fb[0])/4.0; Fij[1] = (Fa[1]-Fb[1])/4.0; Fij[2] = (Fa[2]-Fb[2])/4.0;
        this->DistributePairInteraction(Ra, Rb, Fij);
        Fij[0] = (Fa[0]-Fc[0])/4.0; Fij[1] = (Fa[1]-Fc[1])/4.0; Fij[2] = (Fa[2]-Fc[2])/4.0;
        this->DistributePairInteraction(Ra, Rc, Fij);
        Fij[0] = (Fa[0]-Fd[0])/4.0; Fij[1] = (Fa[1]-Fd[1])/4.0; Fij[2] = (Fa[2]-Fd[2])/4.0;
        this->DistributePairInteraction(Ra, Rd, Fij);
        Fij[0] = (Fb[0]-Fc[0])/4.0; Fij[1] = (Fb[1]-Fc[1])/4.0; Fij[2] = (Fb[2]-Fc[2])/4.0;
        this->DistributePairInteraction(Rb, Rc, Fij);
        Fij[0] = (Fb[0]-Fd[0])/4.0; Fij[1] = (Fb[1]-Fd[1])/4.0; Fij[2] = (Fb[2]-Fd[2])/4.0;
        this->DistributePairInteraction(Rb, Rd, Fij);
        Fij[0] = (Fc[0]-Fd[0])/4.0; Fij[1] = (Fc[1]-Fd[1])/4.0; Fij[2] = (Fc[2]-Fd[2])/4.0;
        this->DistributePairInteraction(Rc, Rd, Fij);
    }
    
}


// Decompose 5-body potentials (CMAP)
void StressGrid::DistributeN5(darray Ra, darray Rb, darray Rc, darray Rd, darray Re, darray Fa, darray Fb, darray Fc, darray Fd, darray Fe)
{
    //Counter
    int i;

    //************************************************************************************
    // UNIT vectors between particles
    darray AB, AC, AD, AE, BC, BD, BE, CD, CE, DE;

    // Distances
    double normAB,normAC,normAD,normAE,normBC,normBD,normBE,normCD,normCE,normDE;

    // (Covariant) Central Force decomposition
    double lab, lac, lad, lae, lbc, lbd, lbe, lcd, lce, lde;

    darray Fij;
    //************************************************************************************

    //Dimension and number of particles
    int nDim = 3;
    int nPart = 5;

    //Number of rows and columns
    int nRow = mds_nrow5, nCol = mds_ncol5, nRHS = 1;

    // Matrix of the system (15 equations x 10 unknowns)
    double M[mds_nrow5*mds_ncol5];
    // Vector, we want to solve M*x = b
    double b[mds_nrow5], s[mds_ncol5], CaleyMengerNormal[mds_ncol5];
    // Scalar product of the Normal and the initial CFD
    double prod;

    // If the force decomposition is cCFD or CFD
    if(this->fdecomp == mds_ccfd || this->fdecomp == mds_ncfd)
    {
        diffarray(Rb, Ra, AB);
        diffarray(Rc, Ra, AC);
        diffarray(Rd, Ra, AD);
        diffarray(Re, Ra, AE);
        diffarray(Rc, Rb, BC);
        diffarray(Rd, Rb, BD);
        diffarray(Re, Rb, BE);
        diffarray(Rd, Rc, CD);
        diffarray(Re, Rc, CE);
        diffarray(Re, Rd, DE);

        normAB=normarray(AB);
        normAC=normarray(AC);
        normAD=normarray(AD);
        normAE=normarray(AE);
        normBC=normarray(BC);
        normBD=normarray(BD);
        normBE=normarray(BE);
        normCD=normarray(CD);
        normCE=normarray(CE);
        normDE=normarray(DE);

        for(i = 0; i < 3; i++)
        {
            if(normAB > mds_eps)
                AB[i]/=normAB;
            if(normAC > mds_eps)
                AC[i]/=normAC;
            if(normAD > mds_eps)
                AD[i]/=normAD;
            if(normAE > mds_eps)
                AE[i]/=normAE;
            if(normBC > mds_eps)
                BC[i]/=normBC;
            if(normBD > mds_eps)
                BD[i]/=normBD;
            if(normBE > mds_eps)
                BE[i]/=normBE;
            if(normCD > mds_eps)
                CD[i]/=normCD;
            if(normCE > mds_eps)
                CE[i]/=normCE;
            if(normDE > mds_eps)
                DE[i]/=normDE;
        }

        for ( i = 0; i < nCol*nRow; i++ )
        {
            M[i] = 0.0;
        }

        //Force on particle 1:
        M[nRow*0+0] = AB[0]; M[nRow*1+0] = AC[0]; M[nRow*2+0] = AD[0]; M[nRow*3+0] = AE[0];
        M[nRow*0+1] = AB[1]; M[nRow*1+1] = AC[1]; M[nRow*2+1] = AD[1]; M[nRow*3+1] = AE[1];
        M[nRow*0+2] = AB[2]; M[nRow*1+2] = AC[2]; M[nRow*2+2] = AD[2]; M[nRow*3+2] = AE[2];
        b[0] = Fa[0]; b[1] = Fa[1]; b[2] = Fa[2];

        //Force on particle 2:
        M[nRow*0+3] = -AB[0]; M[nRow*4+3] = BC[0]; M[nRow*5+3] = BD[0]; M[nRow*6+3] = BE[0];
        M[nRow*0+4] = -AB[1]; M[nRow*4+4] = BC[1]; M[nRow*5+4] = BD[1]; M[nRow*6+4] = BE[1];
        M[nRow*0+5] = -AB[2]; M[nRow*4+5] = BC[2]; M[nRow*5+5] = BD[2]; M[nRow*6+5] = BE[2];
        b[3] = Fb[0]; b[4] = Fb[1]; b[5] = Fb[2];

        //Force on particle 3:
        M[nRow*1+6] = -AC[0]; M[nRow*4+6] = -BC[0]; M[nRow*7+6] = CD[0]; M[nRow*8+6] = CE[0];
        M[nRow*1+7] = -AC[1]; M[nRow*4+7] = -BC[1]; M[nRow*7+7] = CD[1]; M[nRow*8+7] = CE[1];
        M[nRow*1+8] = -AC[2]; M[nRow*4+8] = -BC[2]; M[nRow*7+8] = CD[2]; M[nRow*8+8] = CE[2];
        b[6] = Fc[0]; b[7] = Fc[1]; b[8] = Fc[2];

        //Force on particle 4:
        M[nRow*2+9]  = -AD[0]; M[nRow*5+9]  = -BD[0]; M[nRow*7+9]  = -CD[0]; M[nRow*9+9]  = DE[0];
        M[nRow*2+10] = -AD[1]; M[nRow*5+10] = -BD[1]; M[nRow*7+10] = -CD[1]; M[nRow*9+10] = DE[1];
        M[nRow*2+11] = -AD[2]; M[nRow*5+11] = -BD[2]; M[nRow*7+11] = -CD[2]; M[nRow*9+11] = DE[2];
        b[9] = Fd[0]; b[10] = Fd[1]; b[11] = Fd[2];

        //Force on particle 5:
        M[nRow*3+12] = -AE[0]; M[nRow*6+12] = -BE[0]; M[nRow*8+12] = -CE[0]; M[nRow*9+12] = -DE[0];
        M[nRow*3+13] = -AE[1]; M[nRow*6+13] = -BE[1]; M[nRow*8+13] = -CE[1]; M[nRow*9+13] = -DE[1];
        M[nRow*3+14] = -AE[2]; M[nRow*6+14] = -BE[2]; M[nRow*8+14] = -CE[2]; M[nRow*9+14] = -DE[2];
        b[12] = Fe[0]; b[13] = Fe[1]; b[14] = Fe[2];
        
        if ( this->lapack->SolveMinNorm(nRow, nCol, M, b) )
        {
            std::cout << "ERROR::StressGrid: LAPACK solver fails\n";
            this->ierr = 10;
        }

        //If cCFD project the least squares CFD to the shape space
        if(this->fdecomp == mds_ccfd)
        {
            //Calculate the normal to the Shape Space
            ShapeSpace5Normal(normAB,normAC,normAD,normAE,normBC,normBD,normBE,normCD,normCE,normDE,CaleyMengerNormal);
            
            //Covariant derivative:
            prod = 0.0;
            for ( i = 0; i < nCol; i++ )
            {
                prod += b[i]*CaleyMengerNormal[i];
            }

            for ( i = 0; i < nCol; i++ )
            {
                b[i] = b[i] - prod * CaleyMengerNormal[i];
            }
        }

        // Sum the 10 contributions to the stress
        lab = b[0];
        lac = b[1];
        lad = b[2];
        lae = b[3];
        lbc = b[4];
        lbd = b[5];
        lbe = b[6];
        lcd = b[7];
        lce = b[8];
        lde = b[9];

        Fij[0] = lab * AB[0]; Fij[1] = lab * AB[1]; Fij[2] = lab * AB[2];
        this->DistributePairInteraction(Ra, Rb, Fij);
        Fij[0] = lac * AC[0]; Fij[1] = lac * AC[1]; Fij[2] = lac * AC[2];
        this->DistributePairInteraction(Ra, Rc, Fij);
        Fij[0] = lad * AD[0]; Fij[1] = lad * AD[1]; Fij[2] = lad * AD[2];
        this->DistributePairInteraction(Ra, Rd, Fij);
        Fij[0] = lae * AE[0]; Fij[1] = lae * AE[1]; Fij[2] = lae * AE[2];
        this->DistributePairInteraction(Ra, Re, Fij);
        Fij[0] = lbc * BC[0]; Fij[1] = lbc * BC[1]; Fij[2] = lbc * BC[2];
        this->DistributePairInteraction(Rb, Rc, Fij);
        Fij[0] = lbd * BD[0]; Fij[1] = lbd * BD[1]; Fij[2] = lbd * BD[2];
        this->DistributePairInteraction(Rb, Rd, Fij);
        Fij[0] = lbe * BE[0]; Fij[1] = lbe * BE[1]; Fij[2] = lbe * BE[2];
        this->DistributePairInteraction(Rb, Re, Fij);
        Fij[0] = lcd * CD[0]; Fij[1] = lcd * CD[1]; Fij[2] = lcd * CD[2];
        this->DistributePairInteraction(Rc, Rd, Fij);
        Fij[0] = lce * CE[0]; Fij[1] = lce * CE[1]; Fij[2] = lce * CE[2];
        this->DistributePairInteraction(Rc, Re, Fij);
        Fij[0] = lde * DE[0]; Fij[1] = lde * DE[1]; Fij[2] = lde * DE[2];
        this->DistributePairInteraction(Rd, Re, Fij);

    }
    else if(this->fdecomp == mds_gld)
    {
        Fij[0] = (Fa[0]-Fb[0])/5.0; Fij[1] = (Fa[1]-Fb[1])/5.0; Fij[2] = (Fa[2]-Fb[2])/5.0;
        this->DistributePairInteraction(Ra, Rb, Fij);
        Fij[0] = (Fa[0]-Fc[0])/5.0; Fij[1] = (Fa[1]-Fc[1])/5.0; Fij[2] = (Fa[2]-Fc[2])/5.0;
        this->DistributePairInteraction(Ra, Rc, Fij);
        Fij[0] = (Fa[0]-Fd[0])/5.0; Fij[1] = (Fa[1]-Fd[1])/5.0; Fij[2] = (Fa[2]-Fd[2])/5.0;
        this->DistributePairInteraction(Ra, Rd, Fij);
        Fij[0] = (Fa[0]-Fe[0])/5.0; Fij[1] = (Fa[1]-Fe[1])/5.0; Fij[2] = (Fa[2]-Fe[2])/5.0;
        this->DistributePairInteraction(Ra, Re, Fij);
        Fij[0] = (Fb[0]-Fc[0])/5.0; Fij[1] = (Fb[1]-Fc[1])/5.0; Fij[2] = (Fb[2]-Fc[2])/5.0;
        this->DistributePairInteraction(Rb, Rc, Fij);
        Fij[0] = (Fb[0]-Fd[0])/5.0; Fij[1] = (Fb[1]-Fd[1])/5.0; Fij[2] = (Fb[2]-Fd[2])/5.0;
        this->DistributePairInteraction(Rb, Rd, Fij);
        Fij[0] = (Fb[0]-Fe[0])/5.0; Fij[1] = (Fb[1]-Fe[1])/5.0; Fij[2] = (Fb[2]-Fe[2])/5.0;
        this->DistributePairInteraction(Rb, Re, Fij);
        Fij[0] = (Fc[0]-Fd[0])/5.0; Fij[1] = (Fc[1]-Fd[1])/5.0; Fij[2] = (Fc[2]-Fd[2])/5.0;
        this->DistributePairInteraction(Rc, Rd, Fij);
        Fij[0] = (Fc[0]-Fe[0])/5.0; Fij[1] = (Fc[1]-Fe[1])/5.0; Fij[2] = (Fc[2]-Fe[2])/5.0;
        this->DistributePairInteraction(Rc, Re, Fij);
        Fij[0] = (Fd[0]-Fe[0])/5.0; Fij[1] = (Fd[1]-Fe[1])/5.0; Fij[2] = (Fd[2]-Fe[2])/5.0;
        this->DistributePairInteraction(Rd, Re, Fij);
    }
}

// General function to decompose N-body potentials (it can be used to compute higher order terms coming from EAM for instance)
void StressGrid::DistributeNBody ( int nPart, darraylist R, darraylist F )
{
    int i,j,k, iD, jD, n;
 
    darray F_ij;
    
    // If the force decomposition is cCFD or CFD
    if(this->fdecomp == mds_ccfd || this->fdecomp == mds_ncfd)
    {
    
        //Number of rows and columns
        int nRow;
        int nCol;

        nRow = 3 * nPart;
        nCol = (nPart * (nPart - 1)) / 2;
                
        n = 0;
        for ( i = 0; i < nPart; i++ )
        {
            for ( j = i+1; j < nPart; j++ )
            {
                diffarray(R[j], R[i], this->R_ij[n]);
                scalearray(this->R_ij[n],1.0/normarray(this->R_ij[n]),this->R_ij[n]);
                n++;
            }
        }
        
        for ( i = 0; i < nCol*nRow; i++ )
        {
            this->Amat [i] = 0.0;
            this->AmatT[i] = 0.0;
        }
        
        n = 0;
        for ( i = 0; i < nPart; i++ )
        {
            iD = mds_ndim * i;
            for ( j = i+1; j < nPart; j++ )
            {
                jD = mds_ndim * j;
                for ( k = 0; k < mds_ndim; k++ )
                {
                    this->Amat [nRow*n+(iD+k)] =  this->R_ij[n][k];
                    this->Amat [nRow*n+(jD+k)] = -this->R_ij[n][k];
                    this->AmatT[(iD+k)*nCol+n] =  this->R_ij[n][k];
                    this->AmatT[(jD+k)*nCol+n] = -this->R_ij[n][k];
                }
                n++;
            }
            
            scalearray(F[i], 1.0, &this->bvec[iD]);
        }
                      
        if ( this->lapack->SolveMinNorm(nRow, nCol, this->Amat, this->bvec))
        {
            std::cout << "ERROR::StressGrid: LAPACK solver fails\n";
            this->ierr = 10;
        }
        
        if(this->fdecomp == mds_ccfd)
            this->lapack->QQTb( nCol, nRow, nCol, nRow-6, this->AmatT, this->bvec );
            
        n = 0;
        for ( i = 0; i < nPart; i++ )
        {
            for ( j = i+1; j < nPart; j++ )
            {
                scalearray(this->R_ij[n],this->bvec[n], F_ij);
                this->DistributePairInteraction(R[i], R[j], F_ij);
                n++;
            }
        }
    }
    else if(this->fdecomp == mds_gld)
    {

        for ( i = 0; i < nPart; i++ )
        {

            for ( j = i+1; j < nPart; j++ )
            {

                diffarray(F[i], F[j], F_ij );
                scalearray(F_ij, 1.0/static_cast<double>(nPart), F_ij);

                this->DistributePairInteraction(R[i], R[j], F_ij);
            }
        }
    }
    
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
//AUXILIARY FUNCTIONS

//Sum the stress to the current_grid
void StressGrid::AddAtomStressToGrid(int n, dmatrix stress)
{
    summatrix (this->current_grid[n],stress,this->current_grid[n]);
    return;
}

// Finds the indices on the grid for a given set of coordinates
void StressGrid::GridCoord( darray pt, int *i, int *j, int *k )
{

    int nx,ny,nz;
    double rxx,ryx,ryy,rzx,rzy,rzz;

    /*Looks up indices for grid:
    fractional indices are invbox * coordinate;
    grid indices are then nx*f_ind[0], etc.
    */
    rxx = this->invbox[0][0];
    ryx = this->invbox[1][0];
    ryy = this->invbox[1][1];
    rzx = this->invbox[2][0];
    rzy = this->invbox[2][1];
    rzz = this->invbox[2][2];

    nx = this->nx;
    ny = this->ny;
    nz = this->nz;

    *i = static_cast<int>(nx * pt[0] * rxx);
    *j = static_cast<int>(ny * pt[1] * ryy);
    *k = static_cast<int>(nz * pt[2] * rzz);

    if(pt[0] < 0) *i =*i-1;
    if(pt[1] < 0) *j =*j-1;
    if(pt[2] < 0) *k =*k-1;
}
//----------------------------------------------------------------------------------------
