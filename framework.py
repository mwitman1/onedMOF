#! /usr/local/bin/python

# NOTE runs w/packages pycifrw (import CifFile)
from optparse import OptionParser
import inspect
import numpy as np
import math
import os
import CifFile

class Framework(object):

    def __init__(self, filename):
        """
        Initialize framework related data

        Store relevant lattice parms, replications required, etc
        """
        self.name = filename
        self.cifname = filename[:-4]

        # Store lattice params
        self.a = 0
        self.b = 0
        self.c = 0

        # Store angles
        self.alpha = 0
        self.beta = 0
        self.gamma = 0

        # store cartesian unit cell vectors
        self.v_a = np.zeros(3)
        self.v_b = np.zeros(3)
        self.v_c = np.zeros(3)

        # store space conversion matrices
        self.to_cartesian = np.zeros((3,3))
        self.to_fractional = np.zeros((3,3))

        # store atom descriptors
        self.label = None           # C1 , C2 , H3 , N4 , etc...
        self.description = None     # C_a, C_b, H_a, H_b, etc ...   
        self.atmtype = None         # C  , C  , H  , N  , etc...

        # Atom location data
        self.rx = None
        self.ry = None
        self.rz = None
        self.ra = None
        self.rb = None
        self.rc = None

        self.load_cif(filename)

    # Functions for coordinate transformations


    def set_params(self):
        """
        Set lattice parameters and compute transformation matrices for 1x1x1 unit cell
        """
        self.alpha = self.alpha*2*np.pi/360
        self.beta = self.beta*2*np.pi/360
        self.gamma = self.gamma*2*np.pi/360

        self.compute_UC_matrix()

    def compute_UC_matrix(self):
        """
        Compute transformation matrices for ixjxk simulation box
        """
        # Compute transformation matrix
        tempd=(np.cos(self.alpha)-np.cos(self.gamma)*np.cos(self.beta))/np.sin(self.gamma)
        self.to_cartesian[0,0] = self.a;
        self.to_cartesian[0,1] = self.b*np.cos(self.gamma)
        self.to_cartesian[0,2] = self.c*np.cos(self.beta)
        self.to_cartesian[1,0] = 0.0
        self.to_cartesian[1,1] = self.b*np.sin(self.gamma)
        self.to_cartesian[1,2] = self.c*tempd
        self.to_cartesian[2,0] = 0.0
        self.to_cartesian[2,1] = 0.0
        self.to_cartesian[2,2] = self.c*np.sqrt(1.0-np.power(np.cos(self.beta), 2.0)-np.power(tempd,2.0))


        # compute inverse matrix and lattice vectors
        self.to_fractional = np.linalg.inv(self.to_cartesian)
        self.v_a = np.array((self.a, 
                             0.0, 
                             0.0))
        self.v_b = np.array((self.b*np.cos(self.gamma), 
                             self.b*np.sin(self.gamma), 
                             0.0))
        self.v_c = np.array((self.c*np.cos(self.beta), 
                             self.c*tempd, 
                             self.c*np.sqrt(1.0-np.power(np.cos(self.beta), 2.0)-np.power(tempd,2.0))))

    def update_UC_matrix(self, factor, directions):
    
        to_cartesian = np.copy(self.to_cartesian)

        for ind in directions:
            if(ind == 0):
                to_cartesian[0,0] *= factor
            elif(ind == 1):
                to_cartesian[0,1] *= factor
                to_cartesian[1,1] *= factor
            elif(ind == 2):
                to_cartesian[0,2] *= factor
                to_cartesian[1,2] *= factor
                to_cartesian[2,2] *= factor

        return to_cartesian

    def transform_abc(self, pts):
        """
        Transform points from ixjxk abc to xyz
        """
        if(np.shape(pts)[0] != 3):
            raise ValueError
        else:
            #print pts
            return np.dot(self.to_cartesian,pts)

    def transform_xyz(self, pts):
        """
        Transform points from xyz to ixjxk abc
        """
        if(np.shape(pts)[0] != 3):
            raise ValueError
        else:
            return np.dot(self.to_fractional,pts)

    def modUC(self, num):
        """
        Retrun any fractional coordinate back into the unit cell
        """
        if(hasattr(num,'__iter__')):
            for i in range(len(num)):
                if(num[i] < 0.0):
                    num[i] = 1+math.fmod(num[i], 1.0)
                else:
                    num[i] = math.fmod(num[i], 1.0)

            return num

        else:
            if(num < 0.0):
                num = math.fmod((num*(-1)), 1.0)
            else:
                num = math.fmod(num, 1.0)

    def modGroupUC(self, num):
        for i in range(3):
            for j in range(len(num[0])):
                if(num[i,j] < 0.0):
                    num[i,j] = 1+math.fmod(num[i,j], 1.0)
                else:
                    num[i,j] = math.fmod(num[i,j], 1.0)
        return num



    # Functions for CIF file parsing
    def cif_error(self, string):
        print("ERROR! Cif of <" + self.name + ".cif> has no data for: " + string)
        exit()

    def load_cif(self, filename):
        cwd = os.getcwd()

        self.cf = CifFile.ReadCif(cwd + '/' + filename)

        # should work as long as given CIF file only contains one data set 
        # TODO will need to figure out how to handle exceptions)

        if len(self.cf.keys()) > 1:
            raise ValueError("Error! A CIF file with more than one data loop was encountered. \
                              Can't handle this yet...\nExiting...")
        else:
            self.cfdata = str(self.cf.keys()[0])


        # Import all necessary data from CIF to this instance of Grid
        # Lattice info
        try:
            self.a = float(self.cf[self.cfdata]["_cell_length_a"])
        except:
            self.cif_error("_cell_length_a")
        try:
            self.b = float(self.cf[self.cfdata]["_cell_length_b"])
        except:
            self.cif_error("_cell_length_b")
        try:
            self.c = float(self.cf[self.cfdata]["_cell_length_c"])
        except:
            self.cif_error("_cell_length_c")
        try:
            self.alpha = float(self.cf[self.cfdata]["_cell_angle_alpha"])
        except:
            self.cif_error("_cell_angle_alpha")
        try:
            self.beta = float(self.cf[self.cfdata]["_cell_angle_beta"])
        except:
            self.cif_error("_cell_angle_beta")
        try:
            self.gamma = float(self.cf[self.cfdata]["_cell_angle_gamma"])
        except:
            self.cif_error("_cell_angle_gamma")

        # Label info
        try:
            self.label = np.array(self.cf[self.cfdata]["_atom_site_label"])
        except:
            self.cif_error("_atom_site_label")
        try:
            self.atmtype = np.array(self.cf[self.cfdata]["_atom_site_type_symbol"])
        except:
            self.cif_error("_atom_site_type_symbol")
        try:
            self.description = np.array(self.cf[self.cfdata]["_atom_site_description"])
        except:
            pass
            #self.cif_error("_atom_site_description")


        # Position info
        try:
            self.ra = np.array([float(self.cf[self.cfdata]["_atom_site_fract_x"][i])
                                for i in range(len(self.cf[self.cfdata]["_atom_site_fract_x"]))])
        except:
            self.cif_error("_atom_site_fract_x")
        try:
            self.rb = np.array([float(self.cf[self.cfdata]["_atom_site_fract_y"][i])
                                for i in range(len(self.cf[self.cfdata]["_atom_site_fract_y"]))])
        except:
            self.cif_error("_atom_site_fract_y")
        try:
            self.rc = np.array([float(self.cf[self.cfdata]["_atom_site_fract_z"][i])
                                for i in range(len(self.cf[self.cfdata]["_atom_site_fract_z"]))])
        except:
            self.cif_error("_atom_site_fract_z")

        self.set_params() 

        convert_abc_to_xyz = self.transform_abc(np.array([self.ra, self.rb, self.rc]))
        self.rx = np.array(convert_abc_to_xyz[0])
        self.ry = np.array(convert_abc_to_xyz[1])
        self.rz = np.array(convert_abc_to_xyz[2])

        #print(self.ra)
        #print(self.rb)
        #print(self.rc)
        #print(self.rx)
        #print(self.ry)
        #print(self.rz)

    def reconstruct_cif(self, a, b, c, ra, rb, rc, label, atmtype, name_append = "test"):

        #print(self.cf[self.cfdata].keys())
        #print(type(self.cf[self.cfdata]))
        #for elem in inspect.getmembers(self.cf[self.cfdata], predicate=inspect.ismethod):
        #    print(elem)
        #print()
        #print(type(self.cf))
        #for elem in inspect.getmembers(self.cf, predicate=inspect.ismethod):
        #    print(elem)
        self.cf[self.cfdata].RemoveItem('_cell_length_a') 
        self.cf[self.cfdata].AddItem('_cell_length_a', a) 
        self.cf[self.cfdata].RemoveItem('_cell_length_b') 
        self.cf[self.cfdata].AddItem('_cell_length_b', b) 
        self.cf[self.cfdata].RemoveItem('_cell_length_c') 
        self.cf[self.cfdata].AddItem('_cell_length_c', c) 
        self.cf[self.cfdata].ChangeItemOrder('_cell_length_c', 0) 
        self.cf[self.cfdata].ChangeItemOrder('_cell_length_b', 0) 
        self.cf[self.cfdata].ChangeItemOrder('_cell_length_a', 0) 
        #self.cf[self.cfdata].AddToLoop('_cell_length_a', {'_cell_length_a': [a,b,c,self.alpha,self.gamma,self.beta]})
        #self.cf.AddCifItem({"_cell_length_a":a})
        #self.cf[self.cfdata]["_cell_length_b"] = float(b)
        #self.cf[self.cfdata]["_cell_length_c"] = float(c)

        # if the proposed ligand is smaller than the original, the total # of atoms in final cif will be less
        num_to_delete = len(self.cf[self.cfdata]["_atom_site_label"]) - len(label)
        for i in range(num_to_delete):
            self.cf[self.cfdata]["_atom_site_label"].pop()
            self.cf[self.cfdata]["_atom_site_type_symbol"].pop()
            self.cf[self.cfdata]["_atom_site_fract_x"].pop() 
            self.cf[self.cfdata]["_atom_site_fract_y"].pop()
            self.cf[self.cfdata]["_atom_site_fract_z"].pop()


        for i in range(len(label)):

            # if we need to add atoms to the cif
            if(i >= len(self.cf[self.cfdata]["_atom_site_label"])):
                self.cf[self.cfdata]["_atom_site_label"].append(label[i])
                self.cf[self.cfdata]["_atom_site_type_symbol"].append(atmtype[i])
                self.cf[self.cfdata]["_atom_site_fract_x"].append(ra[i])
                self.cf[self.cfdata]["_atom_site_fract_y"].append(rb[i])
                self.cf[self.cfdata]["_atom_site_fract_z"].append(rc[i])
            else:
                self.cf[self.cfdata]["_atom_site_label"][i] = label[i]
                self.cf[self.cfdata]["_atom_site_type_symbol"][i] = atmtype[i]
                self.cf[self.cfdata]["_atom_site_fract_x"][i] = ra[i] 
                self.cf[self.cfdata]["_atom_site_fract_y"][i] = rb[i]
                self.cf[self.cfdata]["_atom_site_fract_z"][i] = rc[i]

                



            # if we need to delete atoms from the cif bc the new ligand was smaller


            
        #self.cf[self.cfdata].AddToLoop('_atom_site_label', {'_atom_site_label': label})
        #self.cf[self.cfdata].AddToLoop('_atom_site_atmtype', {'_atom_site_atmtyp': atmtype})
        #self.cf[self.cfdata].AddToLoop('_atom_site_fract_x', {'_atom_site_fract_x': ra})
        #self.cf[self.cfdata].AddToLoop('_atom_site_fract_y', {'_atom_site_fract_y': rb})
        #self.cf[self.cfdata].AddToLoop('_atom_site_fract_z', {'_atom_site_fract_z': rc})


        outfile = open(self.cifname + '-' + name_append + '.cif', "w")
        outfile.write(self.cf.WriteOut())
        outfile.close()
        

