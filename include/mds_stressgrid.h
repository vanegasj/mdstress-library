/*=========================================================================

  Module    : MDStress
  File      : mds_stressgrid.h
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

#ifndef __stressgrid_h
#define __stressgrid_h

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

#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <sstream>

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
        
        /** Set/Get number of atoms: */
        //@{
        void SetNumberOfAtoms(int n)
        {   this->nAtoms = n;    }
        int  GetNumberOfAtoms( )
        {   return this->nAtoms; }
        //@}
        
        /** Set/Get number of grid cells in each direction: */
        //@{
        void SetNumberOfGridCellsX(int nx)
        {   this->nx = nx;  }
        void SetNumberOfGridCellsY(int ny)
        {   this->ny = ny;  }
        void SetNumberOfGridCellsZ(int nz)
        {   this->nz = nz;  }
        int  GetNumberOfGridCellsX( )
        {   return this->nx; }
        int  GetNumberOfGridCellsY( )
        {   return this->ny; }
        int  GetNumberOfGridCellsZ( )
        {   return this->nz; }
        //@}
        
        /** Set/Get spacing in each direction: */
        //@{
        void SetSpacing(double d)
        {   this->spacing = d;  }
        double  GetSpacingX( )
        {   return this->gridsp[0]; }
        double  GetSpacingY( )
        {   return this->gridsp[1]; }
        double  GetSpacingZ( )
        {   return this->gridsp[2]; }
        //@}        
        
         /**Set/Get force decomposition */
        //@{
        void SetForceDecomposition ( int fdecomp )
        {   this->fdecomp = fdecomp;    }
        int GetForceDecomposition ( void ) const
        {   return this->fdecomp;   }
        //@}
        
        
         /**Set/Get stress type */
        //@{
        void SetStressType ( int spatatom )
        {   this->spatatom = spatatom;    }
        int GetStressType ( void ) const
        {   return this->spatatom;   }
        //@}
        
        /**Set box: */
        void SetBox(dmatrix box)
        {   
            for (int i = 0; i < mds_ndim; i++ )
                for (int j = 0; j < mds_ndim; j++)
                    this->box[i][j] = box[i][j];
        }

        /**Set the maximum cluster size (by default mds_maxpart) */
        void SetMaxCluster (int maxClust )
        {   this->maxClust = maxClust;  }
        int GetMaxCluster ( ) const
        {   return this->maxClust;  }
        
        /**Set name of the files (extension .mds and numbered by resets) */
        void SetFileName ( const char *filename )
        {   this->filename.assign(filename); }
        
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
         * 10: Lapack failed */
        int GetError ( )
        {   return this->ierr;  }
        
        /** This function initialize the grid depending on the settings. If the settings are incorrect, 
         * it throws an error */
        void Init   ( );
        
        /** This function updates the box, sum the current grid to sum_grid and sets current_grid
         * to zero. It also computes the new spacings */
        void Update ( );
        
        /** Set both sum_grid and current_grid to zero. Sum the number of resets (this is used for 
         * printing files) and set the number of frames to zero */
        void Reset  ( );
        
        /**Writes file with average stress to grid using the filename set by the user*/
        void Write  ( );
        
        /**Writes and resets*/
        void WriteAndReset ( )
        {   this->Write();  this->Reset();  }

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
        void DistributeInteraction ( int nAtoms, darraylist R, darraylist F, int *atomIDs );
        
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
        void DistributeKinetic   ( double mass, darray x, darray va, darray vb, int atomID  );
        //@}
        
        /** Constructor */
        StressGrid( );

        /** Destructor */
        ~StressGrid();
        
    private:
        /** @name Inputs*/
        //@{
        int           nAtoms;         ///< Number of atoms
        int           nx;             ///< Number of grid cells in the x direction
        int           ny;             ///< Number of grid cells in the y direction
        int           nz;             ///< Number of grid cells in the z direction
        long          ncells;         ///< Total number of cells in the calculation
        int           maxClust;
        double        spacing;        ///< spacing requested for the grid
        dmatrix       box;            ///< Actual box
        int           spatatom;       ///< enSpat or enAtom
        int           fdecomp;        ///< which force decomposition
        int           contrib;        ///< which contribution
        std::string   filename;       ///< body of the filename where the stress is stored
        //@}
    
        /** @name Outputs*/
        //@{
        Lapack    *lapack;       ///< mds_lapack: solves underdetermined/overdetermined systems of equations and projects solution onto shape space
        double    *Amat;         ///< matrix for linear systems (for systems with more than 5 particles)
        double    *AmatT;        ///< transpose of the matrix (used for projecting solution onto the shape space)
        double    *bvec;         ///< vector for solving the linear system    
        darraylist R_ij;         ///< distance vectors
        int        ierr;         ///< error type: 0, 1, etc (See GetError() function)
        int        nframes;      ///< Number of frames
        int        nreset;       ///< Number of resets (for writing files)
        dmatrix    sumbox;       ///< Average box
        dmatrix    invbox;       ///< Inverse of the box
        darray     gridsp;       ///< grid spacing
        double     invgridsp;    ///< inverse of grid spacing
        dmatrix   *current_grid; ///< Grid (either nx*ny*nz or nAtoms)
        dmatrix   *sum_grid;     ///< Sum Grid
        //@}

        /** Method to delete the preallocated member variables */
        void Clear();

        /** This function is provided to avoid misleadings parameters, or to identify bad settings */
        int CheckSettings();
        
        /** DistributePairInteraction
         *
         * Distributes interactions onto locals_grid (from the initial grid point to the last grid point)
         * Requires:
         * R1   -> position of particle I (A)
         * R2   -> position of particle J (B)
         * F    -> pairwise force */
        void DistributePairInteraction     ( darray R1, darray R2, darray F );
        
        /** Decompose 3-body potentials (angles)*/
        void DistributeN3                  ( darray Ra, darray Rb, darray Rc, darray Fa, darray Fb, darray Fc );
        
        /** Decompose Settle */
        void DistributeSettle              ( darray Ra, darray Rb, darray Rc, darray Fa, darray Fb, darray Fc );
        
        /** Decompose 4-body potentials (dihedrals) */
        void DistributeN4                  ( darray Ra, darray Rb, darray Rc, darray Rd, darray Fa, darray Fb, darray Fc, darray Fd );
        
        /** Decompose 5-body potentials (CMAP) */
        void DistributeN5                  ( darray Ra, darray Rb, darray Rc, darray Rd, darray Re, darray Fa, darray Fb, darray Fc, darray Fd, darray Fe);
        
        /** General function to decompose N-body potentials (it can be used to compute higher order terms coming from EAM for instance) */
        void DistributeNBody               ( int nPart, darraylist R, darraylist F );
        
        /** SpreadPointSource
         * Distributes "point sources" onto the grid
         * Requires:
         * x      -> source point 
         * stress -> stress to be distributed */
        void SpreadPointSource         (darray x, dmatrix stress);
        
        /** SpreadLineSource
        * Distributes "line sources" onto a grid point (asuming trilinear weighting functions!) 
        * Requires:
        * a      -> position of particle A
        * b      -> position of particle B
        * t1     -> starting parametric time (the line segment is parametrized from 0 to 1 from A to B)
        * t2     -> ending  parametric time 
        * x      -> position of the grid point (integers) where we are spreading the contribution
        * stress -> stress to be spread
        * sumfactor -> (this is just for monitoring) part of the line source that has already been spread */
        void SpreadLineSource          (darray a, darray b, double t1, double t2, iarray x, dmatrix stress, double *sumfactor);
        
        //AUXILIARY FUNCTIONS
        
        /** Sum the stress to the current_grid */
        void AddAtomStressToGrid       (int i,  dmatrix stress);

        /** Finds the indices on the grid for a given set of coordinates */
        void GridCoord(darray pt, int *i, int *j, int *k);

};
#endif // __stressgrid_h
