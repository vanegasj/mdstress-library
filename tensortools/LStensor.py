#=========================================================================
#
#  Module    : LOCAL STRESS FROM GROMACS TRAJECTORIES
#  File      : LStensor.py
#  Authors   : A. Torres-Sanchez and J. M. Vanegas
#  Modified  :
#  Purpose   : Compute the local stress from precomputed trajectories in GROMACS
#  Date      : 25/03/2015
#  Version   :
#  Changes   :
#
#     http://www.lacan.upc.edu/LocalStressFromMD
#
#     This software is distributed WITHOUT ANY WARRANTY; without even
#     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#     PURPOSE.
#
#     Please, report any bug to either of us:
#     juan.m.vanegas@gmail.com
#     torres.sanchez.a@gmail.com
#=========================================================================
#
# References:
#
# Regarding this program:
# [1] Manual (parent folder/manual)
# [2] J. M. Vanegas; A. Torres-Sanchez; M. Arroyo; J. Chem. Theor. Comput. 10 (2), 691-702 (2014)
# [3] O. H. S. Ollila; H.J. Risselada; M. Louhivouri; E. Lindahl; I. Vattulainen; S.J. Marrink; Phys. Rev Lett. 102, 078101 (2009)
#
# General IKN framework and Central Force Decomposition
# [4] E. B. Tadmor; R. E. Miller; Modeling Materials: Continuum, Atomistic and Multiscale Techniques, Cambridge University Press (2011)
# [5] N. C. Admal; E. B. Tadmor; J. Elast. 100, 63 (2010)
#
# Covariant Central Force Decomposition
# [6] A. Torres-Sanchez; J. M. Vanegas; M. Arroyo; Submitted to PRL (2015)
# [7] A. Torres-Sanchez; J. M. Vanegas; M. Arroyo; In preparation (2015)
#
# Goetz-Lipowsky Decomposition
# [8] R. Goetz; R. Lipowsky; J. Chem. Phys. 108, 7397 (1998)
#
# Decomposition on geometric centers
# [9] H. Heinz; W. Paul; K. Binder; Phys. Rev. E. 72 066704 (2005)

try:
    import numpy as np
except:
    print("Numpy not installed or not in python path. Please install it or add it to the python path.")
    exit(10)
try:
    from scipy.ndimage.filters import gaussian_filter
    from scipy.io import netcdf
except:
    print("Scipy not installed or not in python path.  Please install it or add it to the python path.")
    exit(10)

try:
    from prody import parsePDB
    from prody import writePDB
    prody = 0
except:
    prody = 1

import struct



class LStensor:
    '''
    Local Stress calculations. Master class for post-processing data from GROMACS_LS package.

    This class allows loading data produced by the GROMACS_LS package on a rectangular grid or on the atoms of a given system
    '''

    def __init__(self, order):
        ''' Init, sets the basic variables of the class'''
        self.__sizeofint__   = 4
        self.__sizeofdouble__ = 8

        self.order = order
        self.dsize = 3**order

        self.verbose   = False

        self.atoms     = None
        self.data_atom = None
        self.invariants_atom = None
        self.nAtom     = 0
        self.atStr     = False
        self.atPos     = None
        self.a_units   = 'None'

        self.gridFlag = False
        self.data_grid = None
        self.invariants_grid = None
        self.box = np.zeros([3,3])
        self.nx = self.ny = self.nz = 0
        self.dx = self.dy = self.dz = 0
        self.g_units='None'

        self.sym  = None
        self.asym = None
        self.div  = None

        self.dim = ['x','y','z']

    def verbose(self):
        self.verbose = True

    def quiet(self):
        self.verbose = False

    ####################################################################################################################
    # GRID:

    def g_set (self,nx,ny,nz,box=None):
        '''
        Sets the grid of the tensor
        '''
        self.gridFlag = True

        if (box != None):
            self.box = box

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.dx = self.box[0,0]/self.nx
        self.dy = self.box[1,1]/self.ny
        self.dz = self.box[2,2]/self.nz

        return

    def g_get (self):
        '''
        Returns the the grid data: nx, ny, nz and box
        '''
        return self.nx,self.ny,self.nz,self.box

    def g_setdata (self, data):
        '''
        Sets the data in the grid
        (the grid needs to have been set prior to calling this function)
        '''

        if (self.gridFlag == False):
            print("ERROR: LStensor: g_setdata: grid is not defined")
            return 1

        if (np.shape(data)[0]!=self.nx*self.ny*self.nz):
            print("ERROR: LStensor: g_setdata: shape of data does not match the grid size {0}", format(np.shape(data)[0]), format(self.nx*self.ny*self.nz))
            return 1

        self.data_grid = data

        return 0


    def g_loaddata (self, files, bAvg = True):
        '''
        Loads data (and grid) from a set of files. The bAvg flag indicates whether to return the average or the sum of the data from
        each file
        '''
        # Read first input file
        self.data_grid,self.box,self.nx,self.ny,self.nz = self.__readGridbin__(files[0])

        # Read subsequent input files if necessary
        nfiles = len(files)

        # Average or sum data from the rest of the files
        for i in range(1, nfiles):
            tdata,tbox,tx,ty,tz = self.__readGridbin__(files[i])
            self.box += tbox
            self.data_grid += tdata

        self.box /= nfiles

        self.dx = self.box[0,0]/self.nx
        self.dy = self.box[1,1]/self.ny
        self.dz = self.box[2,2]/self.nz

        if (bAvg):
            self.data_grid /= nfiles

        self.gridFlag = True
        
        return 0

    def g_rescale (self,value):
        '''
        Rescale the data
        '''
        self.data_grid *= value

    def g_slice(self, iD, number):
        '''
        Extracts a slice of the 3D grid
        '''
        self.data_grid = np.reshape(self.data_grid,[self.nx, self.ny, self.nz, self.dsize])

        if(iD == 'x'):
            self.data_grid = self.data_grid[number,:,:,:]
            self.nx = 1
            self.dx = self.box[0,0]/self.nx
        elif(iD == 'y'):
            self.data_grid = self.data_grid[:,number,:,:]
            self.ny = 1
            self.dy = self.box[1,1]/self.ny
        elif(iD == 'z'):
            self.data_grid = self.data_grid[:,:,number,:]
            self.nz = 1
            self.dz = self.box[2,2]/self.nz

        self.data_grid = np.reshape(self.data_grid,[self.nx*self.ny*self.nz, self.dsize])

        return

    def g_intout (self, iD, avg = True, red = False):
        '''
        Integrates out one dimension. avg indicates whether to average the data in the dimension or to sum it
        red indicates whether to integrate data or just sum it.
        '''
        if (self.gridFlag == False):
            print("ERROR: LStensor: g_intout: grid is not defined")
            return 1
        
        if(self.verbose):
            print("Integrating out dimension {0}...".format(iD))
        # Integrate out the dimension selected

        self.data_grid = np.reshape(self.data_grid,[self.nx, self.ny, self.nz, self.dsize])

        if(iD == 'x'):
            self.data_grid = np.sum(self.data_grid, axis = 0)
            if(avg):
                self.data_grid /= self.nx
            if(red):
                self.data_grid*=self.dx
            self.nx = 1
            self.dx = self.box[0,0]/self.nx
        elif(iD == 'y'):
            self.data_grid = np.sum(self.data_grid, axis = 1)
            if(avg):
                self.data_grid /= self.ny
            if(red):
                self.data_grid*=self.dy
            self.ny = 1
            self.dy = self.box[1,1]/self.ny
        elif(iD == 'z'):
            self.data_grid = np.sum(self.data_grid, axis = 2)
            if(avg):
                self.data_grid /= self.nz
            if(red):
                self.data_grid*=self.dz
            self.nz = 1
            self.dz = self.box[2,2]/self.nz
        else:
            print("ERROR: LStensor: The iD {0} is different from 'x', 'y' or 'z'",format(iD))
            return

        self.data_grid = np.reshape(self.data_grid,[self.nx*self.ny*self.nz, self.dsize])
        
        if(self.verbose):
            print("DONE!\n")
            
        return

    def g_prof (self, iD, avg = True):
        '''
        Calculates the profile along a given direction
        '''
        if(self.verbose):
            print("Getting profile along dimension {0}***".format(iD))
        for i in self.dim:
            if(i != iD):
                self.g_intout(i,avg)
        if(self.verbose):       
            print("***DONE!\n")

    def g_savebin (self, outputfile):
        '''
        Stores the result in a binary file
        '''
        if (self.gridFlag == False):
            print("ERROR: LStensor: g_savebin: grid is not defined")
            return 1
        
        if(self.verbose):
            print("Writing data on grid in binary format to {0}...".format(outputfile))

        fp = open(outputfile, 'wb')
        fp.write(struct.pack('i',1))

        # Write the box vectors
        fp.write(struct.pack('9d',*(self.box.ravel())))

        # Grid size
        fp.write(struct.pack('i', self.nx))
        fp.write(struct.pack('i', self.ny))
        fp.write(struct.pack('i', self.nz))
        ntot = self.nx*self.ny*self.nz


        fp.write(struct.pack(str(self.dsize*ntot)+'d',*(self.data_grid.ravel())))
        fp.close()

        if(self.verbose):
            print("DONE!\n".format(outputfile))

        return

    def g_savenc  (self, outputfile):
        '''
        Store data in NETCDF file format
        '''
        
        if (self.gridFlag == False):
            print("ERROR: LStensor: g_savenc: grid is not defined")
            return 1
        if (self.order > 2 ):
            print("ERROR: function g_savenc cannot write tensors of order larger than 2")
            return 1
        
        if (self.verbose):
            print("Writing data in NETCDF format to {0}...".format(outputfile))



        fp = netcdf.NetCDFFile(outputfile,'w')

        # Define some global attributes including xyz_origin and xyz_step (in Angstroms) for correct units in other programs (i.e. UCSF Chimera)
        fp.xyz_origin = np.zeros(3)
        fp.xyz_step = np.array([10.0*self.dx,10.0*self.dy,10.0*self.dz])
        fp.grid_size = [self.nx,self.ny,self.nz]
        fp.box_size = np.array([self.box[0,0],self.box[1,1],self.box[2,2]])

        # Define the dimensions and set the units
        fp.createDimension('x',self.nx)
        x = fp.createVariable('x', 'd', ('x',))
        x[:] = np.linspace(0.0,self.box[0,0],num=self.nx,endpoint=False)
        x.units = 'nm'
        fp.createDimension('y',self.ny)
        y = fp.createVariable('y', 'd', ('y',))
        y[:] = np.linspace(0.0,self.box[1,1],num=self.ny,endpoint=False)
        y.units = 'nm'
        fp.createDimension('z',self.nz)
        z = fp.createVariable('z', 'd', ('z',))
        z[:] = np.linspace(0.0,self.box[2,2],num=self.nz,endpoint=False)
        z.units = 'nm'

        # Store the data array
        if  (self.order == 0):
            data_array       = fp.createVariable('Density', 'd', ('z','y','x'))
            data_array.units = self.g_units
            data_array[:]    = np.transpose(self.data_grid[:].reshape((self.nx,self.ny,self.nz)))
        elif(self.order == 1):
            labels = ['vx','vy','vz']
            data_array = [None]*3
            for i in range(3):
                data_array[i]       = fp.createVariable(labels[i], 'd', ('z','y','x'))
                data_array[i].units = self.g_units
                data_array[i][:]    = np.transpose(self.data_grid[:,i][:].reshape((self.nx,self.ny,self.nz)))
        elif(self.order == 2):
            labels = ['Sxx','Sxy','Sxz','Syx','Syy','Syz','Szx','Szy','Szz']
            data_array = [None]*9
            for i in range(9):
                data_array[i]       = fp.createVariable(labels[i], 'd', ('z','y','x'))
                data_array[i].units = self.g_units
                data_array[i][:]    = np.transpose(self.data_grid[:,i][:].reshape((self.nx,self.ny,self.nz)))

        fp.close

        if (self.verbose):
            print("DONE!\n")

        return 0

    def g_saveinvnc  (self, outputfile):
        '''
        Store invariants in NETCDF format
        '''
        
        if (self.gridFlag == False):
            print("ERROR: LStensor: g_saveinvnc: grid is not defined")
            return 1
        if (self.order == 0 ):
            print("ERROR: function g_saveinvnc does not write 0-order tensors. Use g_savenc instead")
            return 1
        if (self.order > 2 ):
            print("ERROR: function g_saveinvnc cannot write tensors of order larger than 2")
            return 1
        
        if (self.verbose):
            print("Writing invariants on grid in NETCDF format to {0}...".format(outputfile))
            

        
        fp = netcdf.NetCDFFile(outputfile,'w')

        # Define some global attributes including xyz_origin and xyz_step (in Angstroms) for correct units in other programs (i.e. UCSF Chimera)
        fp.xyz_origin = np.zeros(3)
        fp.xyz_step = np.array([10.0*self.dx,10.0*self.dy,10.0*self.dz])
        fp.grid_size = [self.nx,self.ny,self.nz]
        fp.box_size = np.array([self.box[0,0],self.box[1,1],self.box[2,2]])

        # Define the dimensions and set the units
        fp.createDimension('x',self.nx)
        x = fp.createVariable('x', 'd', ('x',))
        x[:] = np.linspace(0.0,self.box[0,0],num=self.nx)
        x.units = 'nm'
        fp.createDimension('y',self.ny)
        y = fp.createVariable('y', 'd', ('y',))
        y[:] = np.linspace(0.0,self.box[1,1],num=self.ny)
        y.units = 'nm'
        fp.createDimension('z',self.nz)
        z = fp.createVariable('z', 'd', ('z',))
        z[:] = np.linspace(0.0,self.box[2,2],num=self.nz)
        z.units = 'nm'

        # Store the data array
        if(self.order == 1):
            labels = ['norm']
            data_array = [None]*1
            for i in range(1):
                data_array[i]       = fp.createVariable(labels[i], 'd', ('z','y','x'))
                data_array[i].units = self.g_units
                data_array[i][:]    = np.transpose(self.invariants_grid[:,i][:].reshape((self.nx,self.ny,self.nz)))
        if(self.order == 2):
            labels = ['S0','S1','S2']
            data_array = [None]*3
            for i in range(3):
                data_array[i]       = fp.createVariable(labels[i], 'd', ('z','y','x'))
                data_array[i].units = self.g_units
                data_array[i][:]    = np.transpose(self.invariants_grid[:,i][:].reshape((self.nx,self.ny,self.nz)))

        fp.close

        if (self.verbose):
            print("DONE!\n")

        return 0

    def g_savetxt (self, outputfile):
        '''
        Write data to txt file
        '''

        fp = open(outputfile, 'w')

        #fp.write("# LStensor of order {0}\n# ".format(self.order))
        if (self.order == 0):
            fp.write("# Density in units of kg/m^3, charge/m^3, or counts/m^3\n# ")
        elif (self.order == 1):
            fp.write("# Vector field (LStensor of order {0})\n# ".format(self.order))
        elif (self.order == 2):
            fp.write("# Stress in units of bar (10^5 Pa)\n# ")

        if (self.nx != 1):
            fp.write("x\t")
        if (self.ny != 1):
            fp.write("y\t")
        if (self.nz != 1):
            fp.write("z\t")
        if (self.order == 0):
            fp.write

        if (self.order == 0):
            fp.write("Density\n")
        elif (self.order == 1):
            fp.write("Vx Vy Vz\n")
        elif (self.order == 2):
            fp.write("Sxx\tSxy\tSxz\tSyx\tSyy\tSyz\tSzx\tSzy\tSzz\n")

        if(self.verbose):
            print("Writing data on grid in txt format to {0}...".format(outputfile))

        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if(self.nx > 1):
                         fp.write(str(i*self.dx)+'\t')
                    if(self.ny > 1):
                         fp.write(str(j*self.dy)+'\t')
                    if(self.nz > 1):
                         fp.write(str(k*self.dz)+'\t')
                    for d in range(self.dsize):
                        fp.write(str(self.data_grid[i*self.ny*self.nz+j*self.nz+k,d])+'\t')
                    fp.write('\n')

        fp.close()

        if(self.verbose):
            print("DONE!\n")

        return

    ####################################################################################################################
    # ATOM:


    def a_rescale (self,value):
        '''
        Rescale data per atom
        '''
        self.data_atom *= value

    def a_loaddata (self, files, bAvg = True):
        '''
        Load data per atom from files
        '''

        # Read first input file
        self.data_atom,self.box,self.nAtom = self.__readAtombin__(files[0])

        # Read subsequent input files if necessary
        nfiles = len(files)

        # Average or sum data from the rest of the files
        for i in range(1, nfiles):
            tdata, tbox, nAtom = self.__readAtombin__(files[i])
            self.box += tbox
            self.data_atom += tdata

        self.box /= nfiles

        if (bAvg):
            self.data_atom /= nfiles

        return 0

    def a_loadstr (self, file):
        '''
        Load structure to write PDBs with data or transform data per atom to a grid. It requires the python package prody.
        '''

        if(prody == 1):
            print("ERROR: LStensor: a_loadstr: Trying to set an atomic structure but ProDy is not installed or not in python path. You can find it at http://prody.csb.pitt.edu/index.html.\n" )
            return 1

        self.atoms = parsePDB(file)

        # save positions in nm
        self.atPos = self.atoms.getCoords()/10.0
        self.nAtom = len(self.atPos)

        self.atStr = True

        return 0


    def a_2grid (self, gridsp):
        '''
        Transform data per atom to data on grid
        '''
        
        if (self.atStr == False):
            print("ERROR: LStensor: a_2grid: atom structure undefined\n")
            return 1
        
        #By default generate a grid with a spacing of 0.1 nm
        self.nx = max(int(self.box[0][0]/gridsp[0]), 1)
        self.dx = self.box[0][0]/self.nx
        self.ny = max(int(self.box[1][1]/gridsp[1]), 1)
        self.dy = self.box[1][1]/self.ny
        self.nz = max(int(self.box[2][2]/gridsp[2]), 1)
        self.dz = self.box[2][2]/self.nz
        self.gridFlag = True
        
        data_ = np.zeros([self.nx*self.ny*self.nz,self.dsize])

        if(self.verbose):
            print("Transforming atomic data into grid data...")
        for j in range(self.nAtom):

            jx = int(self.atPos[j,0]/self.dx)
            jy = int(self.atPos[j,1]/self.dy)
            jz = int(self.atPos[j,2]/self.dz)

            data_[jx*self.ny*self.nz+jy*self.nz+jz,:] += self.data_atom[j,:]

        if(self.verbose):
            print("DONE!\n")

        self.data_grid = np.reshape(data_,[self.nx*self.ny*self.nz,self.dsize])
        self.g_units = self.a_units+'/nm^3'
        self.data_grid *= self.nx*self.ny*self.nz/(self.box[0,0]*self.box[1,1]*self.box[2,2])

        return

    def a_savebin(self,outputfile):
        '''
        Save a binary file with the data per atom
        '''

        if(self.verbose):
            print("Writing data on atoms in binary format to {0}...".format(outputfile))

        fp = open(outputfile, 'wb')
        fp.write(struct.pack('i',1))

        # Write the box vectors
        fp.write(struct.pack('9d',*(self.box.ravel())))

        # Grid size
        fp.write(struct.pack('i', self.nAtom))

        ntot = self.nAtom

        fp.write(struct.pack(str(self.dsize*ntot)+'d',*(self.data_grid.ravel())))
        fp.close()

        if(self.verbose):
            print("DONE!\n".format(outputfile))

        return

    def a_savepdb(self, outputfile):
        '''
        Save pdb with the data per atom
        '''
        
        if (self.atStr == False):
            print("ERROR: LStensor: a_savepdb: atom structure undefined\n")
            return 1
        
        if(prody == 1):
            print("ERROR: LStensor: a_savepdb: Trying to  write a structure but ProDy not installed or not in python path. You can find it on http://prody.csb.pitt.edu/index.html.\n" )
            return 1

        outname, outext = outputfile.split('.')
        print('Natoms = {0} and dsize = {1}'.format(self.nAtom,self.dsize))
        for i in range(self.dsize):
            oname = outname+'_b{0}.'.format(i) +outext
            writePDB(oname,self.atoms,beta = self.data_atom[:,i])

        return

    def a_savetxt (self, outputfile):
        '''
        Write stress per atom data to txt file
        '''
        
        if (self.atStr == False):
            print("ERROR: LStensor: a_savetxt: atom structure undefined\n")
            return 1
        
        fp = open(outputfile, 'w')

        fp.write("# Stress per atom data \n")
        fp.write("# Atom No.\tSxx\tSxy\tSxz\tSyx\tSyy\tSyz\tSzx\tSzy\tSzz\n")

        if(self.verbose):
            print("Writing data on grid in txt format to {0}...".format(outputfile))

        for i in range(self.nAtom):
            fp.write(str(i)+'\t')
            for d in range(self.dsize):
                fp.write(str(self.data_atom[i,d])+'\t')
            fp.write('\n')

        fp.close()

        if(self.verbose):
            print("DONE!\n")

        return

    def a_saveinvpdb(self, outputfile):
        '''
        Save invariants in a pdb
        '''

        if(prody == 1):
            print("ERROR: LStensor: a_saveinvpdb: Trying to  write a structure but ProDy not installed or not in python path. You can find it on http://prody.csb.pitt.edu/index.html.\n" )
            return 1

        outname, outext = outputfile.split('.')

        for i in range(3):
            oname  = outname+'_b{0}.'.format(i) +outext
            writePDB(oname,self.atoms,beta = self.invariants_atom[:,i])

        return

    ####################################################################################################################
    # INTERNAL LSFIELDS (SYM, ASYM, DIVERGENCE):

    def g_symasym(self):
        
        if (self.gridFlag == False):
            print("ERROR: LStensor: g_symasym: grid is not defined")
            return 1
        
        if(self.order != 2):
            print("LStensor: gsymasym: The function g_symasym is only applicable to 2nd order tensors\n")
            return

        # Create the symmetric and antisymmetric tensors
        data_sym =  np.zeros([self.nx*self.ny*self.nz,9])
        data_asym = np.zeros([self.nx*self.ny*self.nz,9])

        if(self.verbose):
            print("Splitting data on grid into its symmetric and antisymmetric components...")
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):

                    n = i*self.ny*self.nz+j*self.nz+k

                    # First the diagonal
                    data_sym [n,0] = self.data_grid[n,0]
                    data_asym[n,0] = 0.0
                    data_sym [n,4] = self.data_grid[n,4]
                    data_asym[n,4] = 0.0
                    data_sym [n,8] = self.data_grid[n,8]
                    data_asym[n,8] = 0.0

                    # Symmetric
                    data_sym [n,1] = data_sym [n,3] = (self.data_grid[n,1]+self.data_grid[n,3])/2.0
                    data_sym [n,2] = data_sym [n,6] = (self.data_grid[n,2]+self.data_grid[n,6])/2.0
                    data_sym [n,5] = data_sym [n,7] = (self.data_grid[n,5]+self.data_grid[n,7])/2.0

                    #Antisymmetric
                    data_asym[n,1] = (self.data_grid[n,1]-self.data_grid[n,3])/2.0
                    data_asym[n,3] = -data_asym[n,1]

                    data_asym[n,2] =  (self.data_grid[n,2]-self.data_grid[n,6])/2.0
                    data_asym[n,6] = -data_asym[n,2]
                    data_asym[n,5] =  (self.data_grid[n,5]-self.data_grid[n,7])/2.0
                    data_asym[n,7] = -data_asym[n,5]

        self.sym = LStensor(self.order)
        grid = self.g_get()
        self.sym.g_set(grid[0], grid[1], grid[2], grid[3])
        self.sym.g_setdata(data_sym)
        self.sym.g_units = self.g_units
        self.sym.verbose = self.verbose
            
            
        self.asym = LStensor(self.order)
        grid = self.g_get()
        self.asym.g_set(grid[0], grid[1], grid[2], grid[3])
        self.asym.g_setdata(data_asym)
        self.asym.g_units = self.g_units
        self.asym.verbose = self.verbose

        if(self.verbose):
            print("DONE!\n")
            
        return

    ####################################################################################################################
    # GAUSSIAN FILTERS ON GRID:

    def g_gfilt (self, sigma, truncate=4):

        if (self.gridFlag == False):
            print("ERROR: LStensor: g_gfilt: grid is not defined")
            return 1
        
        if (self.verbose):
            print("Performing gaussian filter using a sigma of {0}...".format(sigma))
            print("\tWARNING!!!! Check your memory usage while running\n\tA large grid size may exhaust your system's memory")

        data_  = np.reshape(self.data_grid,(self.nx, self.ny, self.nz, self.dsize))

        for i in range(self.dsize):
            data_[:,:,:,i] = gaussian_filter(data_[:,:,:,i], sigma, mode='wrap', truncate = truncate, order = [0,0,0])

        data_ = np.reshape(data_,(self.nx*self.ny*self.nz,self.dsize))

        self.data_grid = data_

        if (self.verbose):
            print("DONE!\n")

        return

    # DIVERGENCE (GAUSSIAN FILTER) -> returns a LSfield with the divergence on the grid
    def g_div (self, sigma, truncate = 4):

        if (self.order == 0):
            print("ERROR: LStensor: g_div: cannot get the divergence of a 0-th order tensor")
            return None

        if (self.verbose):
            print("Performing convolution with the gradient Gaussian filter using a sigma of {0} to get the divergence of the field...".format(sigma))
            print("\tWARNING!!!! Check your memory usage while running\n\tA large grid size may exhaust your system's memory")

        data_  = np.zeros([self.nx, self.ny, self.nz, self.dsize//3])
        self.data_grid = np.reshape(self.data_grid,(self.nx, self.ny, self.nz, self.dsize))


        for i in range(self.dsize//3):
            data_[:,:,:,i]  = -gaussian_filter(self.data_grid[:,:,:,i*3+0], sigma, mode='wrap', truncate = truncate, order = [1,0,0])
            data_[:,:,:,i] -=  gaussian_filter(self.data_grid[:,:,:,i*3+1], sigma, mode='wrap', truncate = truncate, order = [0,1,0])
            data_[:,:,:,i] -=  gaussian_filter(self.data_grid[:,:,:,i*3+2], sigma, mode='wrap', truncate = truncate, order = [0,0,1])

        data_ = np.reshape(data_,(self.nx*self.ny*self.nz,self.dsize//3))
        self.data_grid = np.reshape(self.data_grid,(self.nx * self.ny* self.nz, self.dsize))

        self.div = LStensor(self.order-1)
        grid = self.g_get()
        self.div.g_set(grid[0], grid[1], grid[2], grid[3])
        self.div.g_setdata(data_)
        self.div.g_units = self.g_units+'/nm'

        self.div.verbose = self.verbose
        
        if (self.verbose):
            print("DONE!\n")
        return

    ####################################################################################################################
    # INVARIANTS:

    def g_inv(self):

        '''
            Get invariants of the field depending on the order in a grid
        '''

        if (self.order == 1):
            self.invariants_grid = np.zeros([len(self.data_grid),1])
            self.invariants_grid[:,0] = np.sqrt(self.data_grid[:,0]**2 + self.data_grid[:,1]**2)


        if (self.order == 2):
            self.invariants_grid = np.zeros([len(self.data_grid),3])

            # Trace (/3.0)
            self.invariants_grid[:,0] = (self.data_grid[:,0] + self.data_grid[:,4] + self.data_grid[:,8])/3.0

            self.invariants_grid[:,1] = np.sqrt(3*(self.data_grid[:,0]-self.data_grid[:,4])**2 + (self.data_grid[:,4]-self.data_grid[:,8])**2 + (self.data_grid[:,8]-self.data_grid[:,0])**2 + 6.0*(self.data_grid[:,1]**2 + self.data_grid[:,2]**2 + self.data_grid[:,5]**2))

            for i in range(self.nx*self.ny*self.nz):
                self.invariants_grid[i,2] = np.linalg.det(self.data_grid[i,:].reshape(3,3))
                #self.invariants_grid[i,2] = np.linalg.det(self.data_grid[i,[0,1,3,4]].reshape(2,2))

    def a_inv(self):

        '''
            Get invariants of the field depending on the order per atom
        '''

        if (self.order == 2):
            self.invariants_atom = np.zeros([len(self.data_atom),3])

            # Trace (/3.0)
            #self.invariants_grid[:,0] = (self.data_atom[:,0] + self.data_atom[:,4] + self.data_atom[:,8])/3.0
            self.invariants_atom[:,0] = (self.data_atom[:,0] + self.data_atom[:,4])/2.0

            #self.invariants_grid[:,1] = np.sqrt(3*(self.data_atom[:,0]-self.data_atom[:,4])**2 + (self.data_atom[:,4]-self.data_atom[:,8])**2 + (self.data_atom[:,8]-self.data_atom[:,0])**2 + 6.0*(self.data_atom[:,1]**2 + self.data_atom[:,2]**2 + self.data_atom[:,5]**2))
            self.invariants_atom[:,1] = np.sqrt((self.data_atom[:,0]**2+self.data_atom[:,4]**2 - self.data_atom[:,0]*self.data_atom[:,4]+ 3.0*(self.data_atom[:,1]**2 + self.data_atom[:,2]**2)))

            for i in range(self.nAtom):
                #self.invariants_grid[i,2] = np.linalg.det(self.data_atom[i,:].reshape(3,3))
                self.invariants_atom[i,2] = np.linalg.det(self.data_atom[i,[0,1,3,4]].reshape(2,2))

    ####################################################################################################################
    # LOAD DATA (PRIVATE):

    def __readGridbin__(self,inputfile):

        fp = open(inputfile, 'rb')
        bdouble = struct.unpack('i', fp.read(1*self.__sizeofint__))[0]
        box = np.array(struct.unpack('9d', fp.read(9*self.__sizeofdouble__)))
        box = np.reshape(box, (3,3))

        # Read the number of grid points in the box
        nx = struct.unpack('i', fp.read(1*self.__sizeofint__))[0]
        ny = struct.unpack('i', fp.read(1*self.__sizeofint__))[0]
        nz = struct.unpack('i', fp.read(1*self.__sizeofint__))[0]

        if (self.verbose):
            print("Reading the input binary file {0}...".format(inputfile))
            print("\tGrid size = ({0},{1},{2})".format(nx,ny,nz))

        ntot = nx*ny*nz
        data = np.array(struct.unpack(str(self.dsize*ntot)+'d', fp.read(self.dsize*ntot*self.__sizeofdouble__)))

        if(self.verbose):
            print("DONE!\n")

        fp.close()

        data = data.reshape((ntot,self.dsize))

        return data,box,nx,ny,nz

    def __readAtombin__(self,inputfile):

        fp = open(inputfile, 'rb')
        bdouble = struct.unpack('i', fp.read(1*self.__sizeofint__))[0]
        box = np.array(struct.unpack('9d', fp.read(9*self.__sizeofdouble__)))
        box = np.reshape(box, (3,3))

        # Read the number of grid points in the box
        nAtom = struct.unpack('i', fp.read(1*self.__sizeofint__))[0]

        if (self.verbose):
            print("Reading the input binary file {0}...".format(inputfile))
            print("\tNumber of atoms = {0}".format(nAtom))

        ntot = nAtom

        data = np.array(struct.unpack(str(self.dsize*ntot)+'d', fp.read(self.dsize*ntot*self.__sizeofdouble__)))

        if(self.verbose):
            print("DONE!\n")

        fp.close()

        data = data.reshape((ntot,self.dsize))

        return data,box,nAtom

    ####################################################################################################################
