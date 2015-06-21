/*=========================================================================

  Module    : MDStress
  File      : mds_boostpython.cpp
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

#include <boost/python.hpp>
#include <mds_stressgrid.h>


using namespace boost::python;


/** \class StressGridPython
* \brief This class is used as a wrapper for python. It converts python inputs 
* to C++ inputs */
class mds::StressGridPython
{
    public:
    void Init ( )
    {   this->stressgrid.Init(); 
        if (this->stressgrid.GetError() == 0)
        {
            this->maxClust = this->stressgrid.GetMaxCluster();
            this->atomIDs  = new int    [this->maxClust];
            this->R        = new darray [this->maxClust];
            this->F        = new darray [this->maxClust];
        }
    }
    void Update ( )
    {   this->stressgrid.Update(); }
    void Reset ( )
    {   this->stressgrid.Reset(); }
    void Write ( )
    {   this->stressgrid.Write(); }
    void WriteAndReset ( )
    {   this->stressgrid.WriteAndReset(); }
    
    void SetFilename ( const char *c)
    {   this->stressgrid.SetFileName( c );   }
    
    void SetNumberOfAtoms ( int nAtoms )
    {   this->stressgrid.SetNumberOfAtoms( nAtoms ); }
    int GetNumberOfAtoms ( )
    {   return this->stressgrid.GetNumberOfAtoms(); }
    
    void SetMaxCluster (int maxClust )
    {   this->stressgrid.SetMaxCluster(maxClust);  }
    int GetMaxCluster ( ) const
    {   return this->maxClust;  }
    
    void SetBox ( boost::python::list  box )
    {
        dmatrix box_;

        for (int i = 0; i < 3; i ++ )
        {
            for (int j = 0; j < 3; j ++ )
            {
                box_[i][j] = extract<double>(box[i][j]);
            }
        }
        stressgrid.SetBox( box_ );
    }
    
    void SetNumberOfGridCellsX ( int nx )
    {   stressgrid.SetNumberOfGridCellsX( nx ); }
    void SetNumberOfGridCellsY ( int ny )
    {   stressgrid.SetNumberOfGridCellsY( ny ); }
    void SetNumberOfGridCellsZ ( int nz )
    {   stressgrid.SetNumberOfGridCellsZ( nz ); }
    int GetNumberOfGridCellsX ( )
    {   return stressgrid.GetNumberOfGridCellsX(); }
    int GetNumberOfGridCellsY ( )
    {   return stressgrid.GetNumberOfGridCellsY(); }
    int GetNumberOfGridCellsZ ( )
    {   return stressgrid.GetNumberOfGridCellsZ(); }
    
    void SetSpacing ( double spacing )
    {   stressgrid.SetSpacing( spacing );   }
    int GetSpacingX ( )
    {   return stressgrid.GetSpacingX(); }
    int GetSpacingY ( )
    {   return stressgrid.GetSpacingY(); }
    int GetSpacingZ ( )
    {   return stressgrid.GetSpacingZ(); }
    
    void SetForceDecomposition ( int fdecomp )
    {   stressgrid.SetForceDecomposition( fdecomp );   }
    int  GetForceDecomposition ( )
    {   return stressgrid.GetForceDecomposition( );   }

    void SetStressType ( int sttype )
    {   stressgrid.SetStressType( sttype ); }
    int  GetStressType ( )
    {   return stressgrid.GetStressType( );   }

    
    void DistributeInteraction( int nAtoms, boost::python::list R, boost::python::list F, boost::python::list atomIDs = boost::python::list())
    {
        if ( nAtoms > this->maxClust )
        {
            std::cout << "ERROR::StressGridPython: Distribute Interaction has been called with a number of atoms larger than the maximum cluster size previously set, nAtoms=" << nAtoms << " and maxClust=" << this->maxClust << "\n";
            return;
        }

        for (int i = 0; i < nAtoms; i ++ )
        {
            for (int j = 0; j < mds_ndim; j ++ )
            {
                this->R[i][j] = extract<double>(R[i][j]);
                this->F[i][j] = extract<double>(F[i][j]);
            }
        }

        if ( atomIDs == boost::python::list() )
        {
            this->stressgrid.DistributeInteraction( nAtoms, this->R, this->F, NULL);
        }
        else
        {
            this->stressgrid.DistributeInteraction( nAtoms, this->R, this->F, this->atomIDs);
        }
    }

    void DistributeKinetic( double mass, boost::python::list x, boost::python::list va, boost::python::list vb = boost::python::list(), int atomID = -1)
    {
        darray x_;
        darray va_,vb_;

        if ( vb == boost::python::list() )
        {
            for (int j = 0; j < mds_ndim; j ++ )
            {
                x_[j]  = extract<double>(x[j]);
                va_[j] = extract<double>(va[j]);
            }
            stressgrid.DistributeKinetic( mass, x_, va_, NULL, atomID);
        }
        else
        {
            for (int j = 0; j < mds_ndim; j ++ )
            {
                x_[j]  = extract<double>(x[j]);
                va_[j] = extract<double>(va[j]);
                vb_[j] = extract<double>(vb[j]);
            }
            stressgrid.DistributeKinetic( mass, x_, va_, vb_, atomID);
        }
    }

    private:
        
    StressGrid stressgrid;
    darraylist R,F;
    int       *atomIDs;
    int        maxClust;

};

// Boost python module to load StressGrid from python
BOOST_PYTHON_MODULE(libmdstresspy)
{        
    
    class_<mds::StressGridPython>("StressGrid")
    .def("distributeinteraction", &mds::StressGridPython::DistributeInteraction, (boost::python::arg("nAtoms"),boost::python::arg("R"), boost::python::arg("F"),boost::python::arg("atomid")=boost::python::list()))
    .def("distributekinetic",     &mds::StressGridPython::DistributeKinetic, (boost::python::arg("mass"), boost::python::arg("x"), boost::python::arg("va"), boost::python::arg("vb") = boost::python::list(), boost::python::arg("atomid")))
    .def("init",         &mds::StressGridPython::Init)
    .def("update",       &mds::StressGridPython::Update)
    .def("reset",        &mds::StressGridPython::Reset)
    .def("write",        &mds::StressGridPython::Write)
    .def("writeandreset",&mds::StressGridPython::WriteAndReset)
    .def("filename",     &mds::StressGridPython::SetFilename)
    
    .def("setnatoms",    &mds::StressGridPython::SetNumberOfAtoms)
    .def("getnatoms",    &mds::StressGridPython::GetNumberOfAtoms)
    
    .def("setmaxclust",  &mds::StressGridPython::SetMaxCluster)
    .def("getmaxclust",  &mds::StressGridPython::GetMaxCluster)
    
    .def("setncellsx",   &mds::StressGridPython::SetNumberOfGridCellsX)
    .def("setncellsy",   &mds::StressGridPython::SetNumberOfGridCellsY)
    .def("setncellsz",   &mds::StressGridPython::SetNumberOfGridCellsZ)
    .def("getncellsx",   &mds::StressGridPython::GetNumberOfGridCellsX)
    .def("getncellsy",   &mds::StressGridPython::GetNumberOfGridCellsY)
    .def("getncellsz",   &mds::StressGridPython::GetNumberOfGridCellsZ)
    
    .def("setspacing",   &mds::StressGridPython::SetSpacing)
    .def("getspacingx",  &mds::StressGridPython::GetSpacingX)
    .def("getspacingy",  &mds::StressGridPython::GetSpacingY)
    .def("getspacingz",  &mds::StressGridPython::GetSpacingZ)

    .def("setfdecomp",   &mds::StressGridPython::SetForceDecomposition)
    .def("getfdecomp",   &mds::StressGridPython::GetForceDecomposition)

    .def("setsttype",    &mds::StressGridPython::SetStressType)
    .def("getsttype",    &mds::StressGridPython::GetStressType)
        
    .def("setbox",       &mds::StressGridPython::SetBox)
     
    ;
}
