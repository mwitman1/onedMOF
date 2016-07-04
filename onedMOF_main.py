#! /usr/bin/env python

# local modules
# from SVD_transform import transform_data as svd
from molecule import Molecule
from framework import Framework
import transformations as trans

# specialty modules
# from deap import creator, base, tools, algorithms

# standard modules
import sys
import numpy as np
import math
import scipy.optimize
import itertools
from copy import deepcopy

np.set_printoptions(precision=4)

def parse_input():

    inputfile = open("onedMOF.input", "r")
    lines = inputfile.readlines()

    reading_rods = False
    reading_cxns = False

    for i in range(len(lines)):
        parsed = lines[i].strip().split()

        if(parsed[0] == "Framework"):
            framework_name = parsed[1]
            continue
        elif(parsed[0] == "Molecule"):
            molecule_name = parsed[1]
            continue
        elif(parsed[0] == "Dimensionality"):
            dimensionality = int(parsed[1])
            continue

        if(parsed[0] == "Rods"):
            num_rods = int(parsed[1])
            reading_rods = True
            rods = [[] for i in range(num_rods)]
            rod_centers = [[[],[],[]] for i in range(num_rods)]
            #print(rods)
            continue

        elif(parsed[0] == "Connections"):
            num_cxns = int(parsed[1])
            reading_rods = False
            reading_cxns = True
            cxns = [[] for i in range(num_cxns)]
            connect_to_rod = [[] for i in range(num_cxns)]
            #print(cxns)
            continue


        if(reading_rods):
            if(parsed[0] == "Rod"):
                curr_rod_index = int(parsed[1])
                rod_centers[curr_rod_index][0] = float(parsed[2])
                rod_centers[curr_rod_index][1] = float(parsed[3])
                rod_centers[curr_rod_index][2] = float(parsed[4])
                #print(curr_rod_index)
            else:
                #print(curr_rod_index)
                rods[curr_rod_index].append(int(parsed[1]))

        elif(reading_cxns):
            if(parsed[0] == "Connection"):
                curr_cxn_index = int(parsed[1])
            else:
                cxns[curr_cxn_index].append(int(parsed[1]))

                valid_connect = False
                for i in range(len(rods)):
                    if(int(parsed[1]) in rods[i]):
                        connect_to_rod[curr_cxn_index].append(i)
                        valid_connect = True
                        break

                if(not valid_connect):
                    raise ValueError("Error! Connection atom index (%s) does not match any index on the specified rods")

        

    return framework_name, molecule_name, dimensionality, rods, rod_centers, cxns, connect_to_rod

            


class Assembly(object):

    def __init__(self, framework_name, molecule_name, dimensionality, 
                 rods, rod_centers, cxns, connect_to_rod):

        # Initializations of necessary data structs
        self.frame = Framework(framework_name)
        self.mol = Molecule(molecule_name)
        self.oned_direct = dimensionality
        self.twod_direct = []
        for i in range(0,3):
            if(i != self.oned_direct):
                self.twod_direct.append(i)
        # NOTE may need to increase this depending on whether 
        self.oned_imgs = [0,1,2,3]
        self.rods = rods
        self.rod_centers = rod_centers
        self.cxns = cxns
        self.connect_to_rod = connect_to_rod
    
        # compute necessary quantities for optimization from inputs
        self.get_rod_coords()
        self.get_cxn_coords()
        self.get_cxn_extent()
        self.get_cxn_perms()
        self.prepare_opt_vars()
        self.print_initialization()

        # print initialization of optimization variables
        #self.get_starting_SVD_guess()
        print("\n\nOptimization starting guess:")
        self.write_results()



        # run optimization
        self.run_optimization_stochastic()
        #self.run_optimization_GA()

        # output results
        print("\n\nOptimization results:")
        self.write_results()

        # construct and write new unit cell to file
        self.construct_final_UC_SVD()


    def get_rod_coords(self):
        # NOTE for now we have to identify rod centers in input file
        self.rod_centers_abc = [] 
        self.rod_centers_xyz = [] 

        self.rod_coords_abc = []
        self.rod_coords_xyz = []
        self.rod_atmtype = []

        self.rod_ref_abc = []
        self.rod_ref_xyz = []

        self.rod_disp_abc = []
        self.rod_disp_xyz = []


        for i in range(len(rods)):
            # NOTE for now we have to identify rod centers in input file
            self.rod_centers_abc.append(np.array(self.rod_centers[i]))
            self.rod_centers_xyz.append(np.dot(self.frame.to_cartesian, np.array(self.rod_centers[i])))

            this_rod_abc = np.zeros((3,len(self.rods[i])))
            this_rod_xyz = np.zeros((3,len(self.rods[i])))
            this_rod_atmtype = np.empty((len(self.rods[i])),dtype='|S2')
            
            this_rod_ref_abc = np.zeros((3,len(rods[i])))
            this_rod_ref_xyz = np.zeros((3,len(rods[i])))

            this_rod_disp_abc = np.zeros((3,len(rods[i])))
            this_rod_disp_xyz = np.zeros((3,len(rods[i])))


            for j in range(len(rods[i])):
                this_rod_abc[0,j] = self.frame.ra[rods[i][j]]
                this_rod_abc[1,j] = self.frame.rb[rods[i][j]]
                this_rod_abc[2,j] = self.frame.rc[rods[i][j]]

                this_rod_xyz[0,j] = self.frame.rx[rods[i][j]]
                this_rod_xyz[1,j] = self.frame.ry[rods[i][j]]
                this_rod_xyz[2,j] = self.frame.rz[rods[i][j]]
                this_rod_atmtype[j] = str(self.frame.atmtype[rods[i][j]])

                this_rod_ref_abc[0,j] = self.rod_centers_abc[i][0]
                this_rod_ref_abc[1,j] = self.rod_centers_abc[i][1]
                this_rod_ref_abc[2,j] = self.rod_centers_abc[i][2]

                # we have the added complication of rods straddling the PB
                # in this case our ref pt must be defined as center coord that is the closest
                # periodic image, otherwise it won't work
                if(math.fmod(self.rod_centers_abc[i][self.twod_direct[0]],1.0) == 0.0):
                    if(this_rod_abc[self.twod_direct[0],j] < 0.5):
                        this_rod_ref_abc[self.twod_direct[0], j] = 0.0
                    else:
                        this_rod_ref_abc[self.twod_direct[0], j] = 1.0

                if(math.fmod(self.rod_centers_abc[i][self.twod_direct[1]],1.0) == 0.0):
                    if(this_rod_abc[self.twod_direct[1],j] < 0.5):
                        this_rod_ref_abc[self.twod_direct[1], j] = 0.0
                    else:
                        this_rod_ref_abc[self.twod_direct[1], j] = 1.0

                this_rod_ref_abc[self.oned_direct, j] = this_rod_abc[self.oned_direct, j]
               
                this_rod_ref_xyz = np.dot(self.frame.to_cartesian, this_rod_ref_abc) 
                    
            
                #this_triclinic_shift = self.compute_triclinic_xyz_shift(this_cxn_abc[self.oned_direct,j])
                #this_cxn_disp_xyz[:,j] = this_cxn_xyz[:,j] - this_triclinic_shift
                this_rod_disp_xyz[:,j] = this_rod_xyz[:,j] - this_rod_ref_xyz[:,j]

                #this_cxn_disp_xyz[self.twod_direct[0],j] += this_triclinic_shift[self.twod_direct[0]] 
                #this_cxn_disp_xyz[self.twod_direct[1],j] += this_triclinic_shift[self.twod_direct[1]] 


            self.rod_coords_abc.append(this_rod_abc)
            self.rod_coords_xyz.append(this_rod_xyz)
            self.rod_atmtype.append(this_rod_atmtype)

            self.rod_ref_abc.append(this_rod_ref_abc)
            self.rod_ref_xyz.append(this_rod_ref_xyz)

            self.rod_disp_xyz.append(this_rod_disp_xyz)




    def get_cxn_coords(self):
        self.cxn_abc = []
        self.cxn_xyz = []

        # center of coordinates of all cxn pts for a given cxn
        self.cxn_center_abc = []
        self.cxn_center_xyz = []

        # abc center of the rod this cxn pt is in
        self.cxn_ref_abc = []
        self.cxn_ref_xyz = []

        # vector displacement from the center of the rod this cxn pt is in
        # ALWAYS a constant in optimization
        self.cxn_disp_abc = []
        self.cxn_disp_xyz = []

        for i in range(len(cxns)):
            this_cxn_abc = np.zeros((3,len(cxns[i])))
            this_cxn_xyz = np.zeros((3,len(cxns[i])))

            this_cxn_ref_abc = np.zeros((3,len(cxns[i])))
            this_cxn_ref_xyz = np.zeros((3,len(cxns[i])))

            this_cxn_disp_abc = np.zeros((3,len(cxns[i])))
            this_cxn_disp_xyz = np.zeros((3,len(cxns[i])))
        
            for j in range(len(cxns[i])):

                rod_ind = self.connect_to_rod[i][j]                

                this_cxn_abc[0,j] = self.frame.ra[cxns[i][j]]
                this_cxn_abc[1,j] = self.frame.rb[cxns[i][j]]
                this_cxn_abc[2,j] = self.frame.rc[cxns[i][j]]

                this_cxn_xyz[0,j] = self.frame.rx[cxns[i][j]]
                this_cxn_xyz[1,j] = self.frame.ry[cxns[i][j]]
                this_cxn_xyz[2,j] = self.frame.rz[cxns[i][j]]

                this_cxn_ref_abc[0,j] = self.rod_centers_abc[rod_ind][0]
                this_cxn_ref_abc[1,j] = self.rod_centers_abc[rod_ind][1]
                this_cxn_ref_abc[2,j] = self.rod_centers_abc[rod_ind][2]


                # we have the added complication of rods straddling the PB
                # in this case our ref pt must be defined as center coord that is the closest
                # periodic image, otherwise it won't work
                if(math.fmod(self.rod_centers_abc[rod_ind][self.twod_direct[0]],1.0) == 0.0):
                    if(this_cxn_abc[self.twod_direct[0],j] < 0.5):
                        this_cxn_ref_abc[self.twod_direct[0], j] = 0.0
                    else:
                        this_cxn_ref_abc[self.twod_direct[0], j] = 1.0

                if(math.fmod(self.rod_centers_abc[rod_ind][self.twod_direct[1]],1.0) == 0.0):
                    if(this_cxn_abc[self.twod_direct[1],j] < 0.5):
                        this_cxn_ref_abc[self.twod_direct[1], j] = 0.0
                    else:
                        this_cxn_ref_abc[self.twod_direct[1], j] = 1.0

                this_cxn_ref_abc[self.oned_direct, j] = this_cxn_abc[self.oned_direct, j]
               
                this_cxn_ref_xyz = np.dot(self.frame.to_cartesian, this_cxn_ref_abc) 
                    
            
                #this_triclinic_shift = self.compute_triclinic_xyz_shift(this_cxn_abc[self.oned_direct,j])
                #this_cxn_disp_xyz[:,j] = this_cxn_xyz[:,j] - this_triclinic_shift
                this_cxn_disp_xyz[:,j] = this_cxn_xyz[:,j] - this_cxn_ref_xyz[:,j]

                #this_cxn_disp_xyz[self.twod_direct[0],j] += this_triclinic_shift[self.twod_direct[0]] 
                #this_cxn_disp_xyz[self.twod_direct[1],j] += this_triclinic_shift[self.twod_direct[1]] 

            self.cxn_abc.append(this_cxn_abc)
            self.cxn_xyz.append(this_cxn_xyz)

            self.cxn_center_xyz.append(np.array([np.average(this_cxn_xyz[0,:]),
                                                 np.average(this_cxn_xyz[1,:]),
                                                 np.average(this_cxn_xyz[2,:])]))

            self.cxn_ref_abc.append(this_cxn_ref_abc)
            self.cxn_ref_xyz.append(this_cxn_ref_xyz)

            self.cxn_disp_xyz.append(this_cxn_disp_xyz)


    def get_cxn_extent(self):
        # NOTE only looks at first cxn group
        self.max_extent = 0
        #for i in range(len(self.cxn_xyz)):
        for j in range(np.shape(self.cxn_xyz)[0]):
            this_extent = np.linalg.norm(self.cxn_xyz[0][0:3,0]-self.cxn_xyz[0][0:3,j])
            if(this_extent > self.max_extent):
                self.max_extent = float(this_extent)

    def get_cxn_perms(self):
        """
        We will have to do SVD across all possible periodic images of the cxn groups
        Therefore we will need to test nCr (cxn group size * num_imgs) choose (mol cxn size) 
        """

        self.cxn_perms = []

        for i in range(len(self.cxns)):
            self.cxn_perms.append(list(itertools.combinations([j for j in range(len(self.cxns[i])*len(self.oned_imgs))], 
                                                              len(self.mol.cxns))))
            #print(len(self.cxn_perms[i]))
            #print(self.cxn_perms[i])

    def get_cxn_images(self):
        """
        For each cxn pt in a group, we need n periodic images to do the SVD transform on
        """

        self.cxn_images = []
        
        for i in range(len(self.cxns)):
            for j in range(len(self.oned_imgs)):
                pass
            

    def compute_triclinic_xyz_shift(self, oned_abc_coord):
        """
        Triclinic shift refers to the shift in twod_direct coordinates that is induced
        by a non-90 deg angle in the direction of oned_direct

        This shift must be accounted for because we are going to optimize rod positions
        in the oned_direct, which means that triclinic shift will change throughout the optimization
        """

        shift_vec = np.array([0.0,0.0,0.0])

        shift_vec[self.oned_direct] = oned_abc_coord


        shift_vec = np.dot(self.frame.to_cartesian, shift_vec)
        #print("%f %s" % (oned_abc_coord, str(shift_vec)))
        return shift_vec
        
        

    def print_initialization(self):
        print("Preparing optimization...")
        print("Framework: %s\nMolecule: %s\nDirection of 1D: %s\nDirections of expansion: %s" %
              (self.frame.name, self.mol.name, self.oned_direct, self.twod_direct))

        print("\n\n\nRod groups:")
        for i in range(len(rods)):
            print("Rod: %d -> abc: %s -> xyz: %s" % (i, str(self.rod_centers_abc[i]), str(self.rod_centers_xyz[i])))

        print("\n\n\nConnection groups:")
        for i in range(len(cxns)):
            print("\n\nGroup: %d" % (i))
            print("Has centroid of -> xyz: %s" % (self.cxn_center_xyz[i]))
            for j in range(len(cxns[i])):
                print("\nAtom %i -> rod %d -> abc: %s -> xyz: %s" % 
                      (self.cxns[i][j], self.connect_to_rod[i][j], str(self.cxn_abc[i][:,j]),
                                                                          str(self.cxn_xyz[i][:,j])))

                print("Maps to ref pt of abc: %s -> xyz: %s" % (self.cxn_ref_abc[i][:,j],
                                                                self.cxn_ref_xyz[i][:,j]))
                print("Displacement from ref pt: %s" % (self.cxn_disp_xyz[i][:,j]))



        print("\n\n\nMolecule info:")
        print("Has centroid of -> xyz: %s" % (self.mol.center))
        for j in range(np.shape(self.mol.cxns)[1]):
            print("Conn %i -> xyz: %s" % (j, str(self.mol.cxns[:,j])))
            
        print("Permutations btwn mol cxns and rod cxns:")
        for it in self.mol.permutations:
            print(it)

        #print(np.dot(self.frame.to_cartesian, np.array([0.547573,0.911527,0.531949])))
        #print(np.dot(self.frame.to_cartesian, np.array([0.0,0.0,2.0])))
        #print(self.compute_triclinic_xyz_shift(2.0))


    def shift_group_in_oned(self):
        """
        This fcn is important bc we need to inspect different integer images in the oned_direct
        sometimes when evaluating fits
        """
        pass

    def prepare_opt_vars(self):
        # the scaling factor to increase the lattice params in the non-1D direction
        # the initial guess is the (max extent of the cxn group)/(max extent of the new ligand)
        self.opt_vec = [self.mol.max_extent/self.max_extent]

        # For now just constrain so that we don't go too far away from the sol'n
        self.opt_bounds = [(0.8,3.0)]
        # the list of shifts that each rod must undergo to achieve optimal framework
        for i in range(len(self.rods)):
            self.opt_vec += [0.0] 
            self.opt_bounds += [(0.0,1.0)]


        # option 1: just optimize the orientation and translation of each molecule to its cxn group
        #for i in range(len(self.cxns)):
        #    # euler1, euler2, euler3, tx, ty, tz
        #    self.opt_vec += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #    self.opt_bounds += [(None,None), (None,None), (None,None), (None,None), (None,None), (None,None)]


        #    for j in range(len(self.cxns[i])):
        #        # the variable that tells us what periodic image of each connection pt to look at
        #        self.opt_vec += [0.0]
        #        self.opt_bounds += [(None,None)]

        #        if(i==1 and (j == 0 or j == 3)):
        #            self.opt_vec[len(self.opt_vec)-1] = 1.0

        #        if(i==2 and (j == 0 or j == 3)):
        #            self.opt_vec[len(self.opt_vec)-1] = 1.0

        # option 2: use a SVD to obtain the optimal orientation and translation
        # this would be much faster but is it even possible to do with PBC??

        # an attribute that lets us store the best permutation for the SVD
        # that provides the optimal fit between mol cxn and rod cxn
        self.opt_perms = [() for i in range(len(self.cxns))]
        self.opt_mol_fit = [() for i in range(len(self.cxns))]

 

    def orient_cxn_and_translate(self, xuse, index = None):

        oriented = []
        if(index == None):
            for i in range(len(self.cxns)):
                conn_start_ind = 1 + len(self.rods) + i*6 + i*len(self.cxns[i])
                rot_from_angles = trans.compose_matrix(angles = xuse[conn_start_ind:conn_start_ind+3])[0:4,0:4]
                #print(rot_from_angles)
                rot_from_angles[0,3] = xuse[conn_start_ind + 3]
                rot_from_angles[1,3] = xuse[conn_start_ind + 4]
                rot_from_angles[2,3] = xuse[conn_start_ind + 5]
                oriented.append(np.dot(rot_from_angles, self.mol.cxns))

        else:
            conn_start_ind = 1 + len(self.rods) + index*6
            rot_from_angles = trans.compose_matrix(angles = xuse[conn_start_ind:conn_start_ind+3])[0:4,0:4]
            #print(rot_from_angles)
            rot_from_angles[0,3] = xuse[conn_start_ind + 3]
            rot_from_angles[1,3] = xuse[conn_start_ind + 4]
            rot_from_angles[2,3] = xuse[conn_start_ind + 5]
            oriented.append(np.dot(rot_from_angles, self.mol.cxns))

        return oriented
        
    def orient_molecule_and_translate(self, xuse, index = None):

        oriented = []
        if(index == None):
            for i in range(len(self.mol.molecule)):
                # start index is F, {dC}, euler angles and tx,y,z, then the C image of each cxn pt
                conn_start_ind = 1 + len(self.rods) + i*6 + i*len(self.cxns[i]) 
                rot_from_angles = trans.compose_matrix(angles = xuse[conn_start_ind:conn_start_ind+3])[0:4,0:4]
                #print(rot_from_angles)
                rot_from_angles[0,3] = xuse[conn_start_ind + 3]
                rot_from_angles[1,3] = xuse[conn_start_ind + 4]
                rot_from_angles[2,3] = xuse[conn_start_ind + 5]
                oriented.append(np.dot(rot_from_angles, self.mol.molecule))

        else:
            conn_start_ind = 1 + len(self.rods) + index*6
            rot_from_angles = trans.compose_matrix(angles = xuse[conn_start_ind:conn_start_ind+3])[0:4,0:4]
            #print(rot_from_angles)
            rot_from_angles[0,3] = xuse[conn_start_ind + 3]
            rot_from_angles[1,3] = xuse[conn_start_ind + 4]
            rot_from_angles[2,3] = xuse[conn_start_ind + 5]
            oriented.append(np.dot(rot_from_angles, self.mol.molecule))

        return oriented


    def get_starting_SVD_guess(self):
        """
        Get the affine transformation matrix that best matches the moelcules cxns
        to each rod cxn by SVD

        We do a Euclidean (rigid) transform
        """

        print("\n\nCalculating intial affine transformations:")
        print("\n\nM = affine transformation for best fit of mol cxn -> rod cxn:")
        for i in range(len(self.cxns)):
            print("\nMol1 -> Rod cxn: %d" % (i))
            a = self.mol.cxns[0:3,:]
            b = self.cxn_xyz[i]

            M = trans.affine_matrix_from_points(a, b, shear = False, scale = False, usesvd = True)

            alpha, beta, gamma = trans.euler_from_matrix(M)
            translations = M[0:3,3]

            conn_start_ind = 1 + len(self.rods) + i*6 + i*len(self.cxns[i])
            print(len(self.cxns[i]))
            print(conn_start_ind)
            self.opt_vec[conn_start_ind+0] = alpha  
            self.opt_vec[conn_start_ind+1] = beta 
            self.opt_vec[conn_start_ind+2] = gamma 
            self.opt_vec[conn_start_ind+3] = translations[0] 
            self.opt_vec[conn_start_ind+4] = translations[1]
            self.opt_vec[conn_start_ind+5] = translations[2]
            print(M)
            
    def evaluate_RMSE_from_SVD(self, src, dest):
        """
        Get the affine transformation matrix that best matches the moelcules cxns
        to each rod cxn by SVD

        We do a Euclidean (rigid) transform
        """

        # M = trans.affine_matrix_from_points(src, dest, shear = False, scale = False, usesvd = True)
        M = trans.superimposition_matrix(src, dest)

        fit = np.dot(M,src)

        rmse = 0
        for i in range(np.shape(fit)[1]):
            rmse += np.linalg.norm(fit[:3,i]-dest[:3,i])

        return M, rmse, fit 


    def construct_curr_UC_SVD(self, xuse):
        """
        Creates the current representation of the unit cell so that we can
        evaluate how well the components are embedded in 3 space

        NOTE: we may have to break this optimization into several pieces:
            (1) determine embedding by fitting one molecule
            (2) only optimize the remaining molecule orientations/translations with the fixed
                embedding variables (F, {dCs})
        """

        # Steps to reconstruct unit cell
        # 1: stasrt with opt_vec[0] (F) and opt_vec[1:n_rods] (set of dCs)
        # 2: recompute cxn points based on F and dCs

        # recompute UC matrix transformation based on current scale factor
        to_cartesian = self.frame.update_UC_matrix(xuse[0], self.twod_direct)
        #to_fractional = np.linalg.inv(to_cartesian)
        #print(to_cartesian)

        # get the current oriented and translated ligands
        # oriented = self.orient_cxn_and_translate(xuse)


        # get final connection pt coords on molecule
        #for i in range(len(oriented)):
        #    oriented[i][0:3,:] = np.dot(to_fractional, oriented[i][0:3,:])
        #    oriented[i][0:3,:] = self.frame.modGroupUC(oriented[i][0:3,:])
        #    oriented[i][0:3,:] = np.dot(to_cartesian, oriented[i][0:3,:])


        #final_xyz = []
        #new_abc = np.copy(self.cxn_ref_abc)
        #print(new_abc)
        #print(np.shape(new_abc))
        #print(self.cxn_ref_abc)
        #print(np.shape(self.cxn_ref_abc))


        totalRMSE = 0.0

        # do this routine for  each cxn group
        for i in range(len(self.cxns)):
        #    print("\n\nCxn group %d:" % (i))
            # for each cxn group, we need the original cxn coords and all their possible oned imgs
            this_new_abc = np.zeros((3,np.shape(self.cxn_ref_abc[i])[1] * len(self.oned_imgs)))

            #print(this_new_abc)
            # loop over each cxn atom
            for j in range(len(self.cxns[i])):
                # loop over each 1D img of that cxn atom
                for k in range(len(self.oned_imgs)):
                    # cxns[i]1  cxns[i]2
                    # print(k + j*len(self.oned_imgs))
                    #print(self.cxn_ref_abc[i][:,j]) 
                    this_new_abc[:, k + j*len(self.oned_imgs)] = self.cxn_ref_abc[i][:,j]
                    # shift ref pt based on curr val of rod shift
                    # shift ref pt as well based on the periodic image indicated in the opt_Vec
                    # print(xuse[1 + self.connect_to_rod[i][j]])
                    #print(xuse[1 + self.connect_to_rod[i][j]]+self.oned_imgs[k])
                    this_new_abc[self.oned_direct, k + j*len(self.oned_imgs)] += xuse[1 + self.connect_to_rod[i][j]]+self.oned_imgs[k]

            #print(this_new_abc)

            # get the new xyz of all points
            
            this_new_xyz = np.dot(to_cartesian, this_new_abc)
            #print(this_new_xyz)
            # apply shift (this is the fixed relative positions in rod constraint)
            # non-trivial if we have non perpendicular oned_direct, but we took care of this
            # in self.get_cxn_coords()
            
            # loop over each cxn atom
            for j in range(len(self.cxns[i])):
                # loop over each 1D img of that cxn atom
                for k in range(len(self.oned_imgs)):
                    #print(this_new_xyz[:, k + j*len(self.oned_imgs)])
                    #print(self.cxn_disp_xyz[i][:,j])
                    this_new_xyz[:, k + j*len(self.oned_imgs)] += self.cxn_disp_xyz[i][:,j]

            #print(this_new_xyz)

            this_opt_perm_err = 100000000.0
            for perm in self.cxn_perms[i]:
                #print(perm)
                #print(self.mol.cxns)
                #print(this_new_xyz[:, perm])
                M, rmse, fit = self.evaluate_RMSE_from_SVD(self.mol.cxns, this_new_xyz[:, perm])
                #print(rmse)

                if(rmse < this_opt_perm_err):
                    this_opt_perm_err = rmse
                    self.opt_perms[i] = perm
                    self.opt_mol_fit[i] = M

                if(rmse < 0.1):
                    # This is a stopping criteria for the rmse
                    # break our loop over permutations if we've found the answer
                    break
            
            

            
            totalRMSE += this_opt_perm_err
            # if(i == 0):
                #print(xuse)
                #print(self.mol.cxns)
                #print(this_new_xyz)
                #print(np.dot(to_fractional,this_new_xyz))
                ##print(this_new_xyz[:,[self.opt_perms[0]]])
                ##print(this_new_xyz[:,self.cxn_perms[0][0]])
                #print(this_opt_perm_err)
            
                
        #    print(np.dot(self.opt_mol_fit[i], self.mol.cxns))
        #    print(self.opt_perms)
        #print(self.opt_perms)
        #sys.exit()



        return totalRMSE

    def construct_curr_UC_GA(self, xuse):
        """
        Creates the current representation of the unit cell so that we can
        evaluate how well the components are embedded in 3 space

        NOTE: we may have to break this optimization into several pieces:
            (1) determine embedding by fitting one molecule
            (2) only optimize the remaining molecule orientations/translations with the fixed
                embedding variables (F, {dCs})
        """

        # Steps to reconstruct unit cell
        # 1: stasrt with opt_vec[0] (F) and opt_vec[1:n_rods] (set of dCs)
        # 2: recompute cxn points based on F and dCs

        # recompute UC matrix transformation based on current scale factor
        to_cartesian = self.frame.update_UC_matrix(xuse[0], self.twod_direct)
        to_fractional = np.linalg.inv(to_cartesian)

        # get the current oriented and translated ligands
        oriented = self.orient_cxn_and_translate(xuse)


        # get final connection pt coords on molecule
        for i in range(len(oriented)):
            oriented[i][0:3,:] = np.dot(to_fractional, oriented[i][0:3,:])
            oriented[i][0:3,:] = self.frame.modGroupUC(oriented[i][0:3,:])
            oriented[i][0:3,:] = np.dot(to_cartesian, oriented[i][0:3,:])


        final_xyz = []
        #new_abc = np.copy(self.cxn_ref_abc)
        #print(new_abc)
        #print(np.shape(new_abc))
        #print(self.cxn_ref_abc)
        #print(np.shape(self.cxn_ref_abc))

        # get final connection pt coords on rod
        for i in range(len(self.cxns)):
            this_new_abc = np.copy(self.cxn_ref_abc[i])

            # start of this cxn info in the optimization vec
            conn_start_ind = 1 + len(self.rods) + i*6 + i*len(self.cxns[i])

            for j in range(len(self.cxns[i])):

                # shift ref pt based on curr val of rod shift
                # print(xuse[1 + self.connect_to_rod[i][j]])
                this_new_abc[self.oned_direct,j] += xuse[1 + self.connect_to_rod[i][j]]

                # shift ref pt as well based on the periodic image indicated in the opt_Vec
                periodic_image = round(xuse[conn_start_ind + 6 + j], 0)
                this_new_abc[self.oned_direct,j] += periodic_image

            # get the new xyz
            this_new_xyz = np.dot(to_cartesian, this_new_abc)
            # apply shift (this is the fixed relative positions in rod constraint)
            # non-trivial if we have non perpendicular oned_direct, but we took care of this
            # in self.get_cxn_coords()
            #shifted_xyz = new_xyz + self.cxn_disp_xyz[i][:,j]
            this_shifted_xyz = this_new_xyz + self.cxn_disp_xyz[i]

            
            # use modified UC matrix to get back abc coords
            this_shifted_abc = np.dot(to_fractional, this_shifted_xyz)
            # mod cxns back into the UC if they left
            this_modded_abc = self.frame.modGroupUC(this_shifted_abc)


            # final xyz coords in UC
            this_final_xyz = np.dot(to_cartesian, this_modded_abc)
            final_xyz.append(this_final_xyz)

        #print(final_xyz)

        # finally evaluate RMSE from all possible permutations
        minRMSE = 0.0

        # i indexes the molecule we are fitting
        # need to iterate over this one first bc each molecule could have its own permutation
        # NOTE for now just optiize based on 1 molecule fitting
        for i in range(len(self.cxns)):
        # for i in [1]:

            thisMolRMSE = 1000000.0

            indOfMinPerm = -1
            currInd = 0
            for it in self.mol.permutations:
                thisPermRMSE = 0.0

                # j indexes each connection pt in the molecule
                for j in range(len(self.cxns[i])):
                    thisPermRMSE += self.rmse(final_xyz[i][:,j], oriented[i][0:3,it[j]])


                if(thisPermRMSE < thisMolRMSE):
                    thisMolRMSE = float(thisPermRMSE)
                    indOfMinPerm = int(currInd)
                currInd += 1

            minRMSE += thisMolRMSE

        #pass
        #print("%d %f" % (indOfMinPerm, minRMSE))
        return minRMSE,

    def rmse(self, xyz1, xyz2):
        return np.exp(np.linalg.norm(xyz1 - xyz2))






    def run_optimization_deterministic(self):
        xvec = deepcopy(self.opt_vec)
        bounds = deepcopy(self.opt_bounds)

        result = scipy.optimize.minimize(self.construct_curr_UC_SVD, 
                                         xvec, 
                                         method='SLSQP', 
                                         bounds=bounds,
                                         options ={'ftol': 0.5, 'maxiter':1000000})

        self.opt_vec = result.x
        print(result)

    def run_optimization_stochastic(self):
        xvec = deepcopy(self.opt_vec)
        bounds = deepcopy(self.opt_bounds)
    
        result = scipy.optimize.minimize(self.construct_curr_UC_SVD, 
                                         xvec, 
                                         method='Nelder-Mead', 
                                         options ={'xtol': 0.1, 'ftol':0.1,  
                                                   #'maxiter':2, 'maxfev':1})
                                                   'maxiter':100000, 'maxfev':100000})

        self.opt_vec = result.x
        print(result)
    
    def geometry_mutation(self, some_list):
        return([0.5 for i in range(len(some_list))])

    def checkStrategy(self, minstrategy, maxstrategy, minvalue, maxvalue):
        """
        Decorator that limits the min and max values of all individuals' attributes
        and strength of those attributes' mutations
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                children = func(*args, **kwargs)
                for child in children:
                    for i, s in enumerate(child.strategy):
                        if s < minstrategy:
                            child.strategy[i] = minstrategy
                        #if s > maxstrategy:
                        #    # is it possible to not use a max strategy ?
                        #    child.strategy[i] = maxstrategy
                        #if child[i] < minvalue:
                        #    child[i] = minvalue
                        #if child[i] > maxvalue:
                        #    # is it possible to not bound upper attribute limit ?
                        #    child[i] = maxvalue
                return children
            return wrapper
        return decorator

    def generateES(self, some_list, icls, scls, size, imin, imax, smin, smax):
        """
        Initialization function for an evolution strategy
        (http://deap.gel.ulaval.ca/doc/dev/examples/es_fctmin.html)
    
        Evolution strategy where mutation strength is learned throughout the evolution
        e.g. Control the standard deviation of the mutation for each attribute of an individual
        by an evolution similar to individual evolution in a classic GA
    
        Evolution strategies are critical for solution convergence if initital guesses are 
        far from true solution
    
        This is crucial for complicated potentials where an good initial guess is extremely
        non-trivial
        (If we are fitting LJ pot we can always use UFF/Dreiding as a reasonable starting
        guess, in which case we can usually converge fairly easily to a solution, but if we
        start out with guesses of 1 for all eps, sig, then the algo will not gain traction
        in any reasonable time frame)
    
        icls = class of individual to instantiate
        scls = class of strategy to use as strategy
        size = size of individ
        imin = minimum value for individual
        imax = maximum value for individual
        smin = minimum value for standard deviation of all individual's attributes' mutation
        smax = maximum value for standard deviation of all individual's attributes' mutation
        """
    
        # Use a random starting guess for parameters (pretty bad idea)
        # ind = icls(random.uniform(imin, imax) for _ in range(size))
        # Use a good starting guess for parameters
        ind = icls(some_list)
    
        # Use a random starting guess for each parameters mutation strength (pretty bad idea)
        # ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
        # Use a good starting guess for parameter mutation strength
        ind.strategy = scls(self.geometry_mutation(some_list))
    
        return ind

    def custom_migRing(self, populations, k, selection, replacement=None, migarray=None):
        nbr_demes = len(populations)
        if migarray is None:
            migarray = range(1, nbr_demes) + [0]
    
        immigrants = [[] for i in xrange(nbr_demes)]
        emigrants = [[] for i in xrange(nbr_demes)]
    
        for from_deme in xrange(nbr_demes):
            emigrants[from_deme].extend(selection(populations[from_deme], k))
            if replacement is None:
                # If no replacement strategy is selected, replace those who migrate
                immigrants[from_deme] = emigrants[from_deme]
            else:
                # Else select those who will be replaced
                immigrants[from_deme].extend(replacement(populations[from_deme], k))
    
        for from_deme, to_deme in enumerate(migarray):
            for i, immigrant in enumerate(immigrants[to_deme]):
                indx = populations[to_deme].index(immigrant)
                populations[to_deme][indx] = emigrants[from_deme][i]

    def run_optimization_GA(self):

        # Shape of optimization parameters
        OPT_SHAPE = (len(self.opt_vec))
        
        # flattening of optimization parameters (size of an individual genome)
        IND_SIZE = np.prod(OPT_SHAPE)
        
        # population size for parameter optimization
        # 3 * # attributes per individual
        POP_SIZE = IND_SIZE * 4
        
        # number of islands (subpopulations that evolve independently until a migration)
        NISLANDS = 3
        
        # set max number of generations to run for
        NGEN = 60
        
        # Migrations frequency
        MIG_FREQ = 20
        
        # Evolution strategy variables
        MIN_VALUE = 0.0            # individual attribute min 
        MAX_VALUE = 7.0     # individual attribute max
        MIN_STRATEGY = 0.0         # min value of strength of mutation
        MAX_STRATEGY = 1.5      # max value of strength of mutation
        
        # If we want to run optimization in parallel, all information must be accessed
        # through picklable data types in python
        #ffobj.optimization_shape=(ffobj.guest.ncomp, ffobj.grid.ncomp, ffobj.model.num_params)
        #pickled = convert_ffobj_to_dict(ffobj)
        
        opt_weights = (-1.0,)
        
        
        
        
        creator.create("FitnessMin", base.Fitness, weights = opt_weights)
        creator.create("Individual", list, fitness=creator.FitnessMin, strategy = None)
        creator.create("Strategy", list)
        
        toolbox = base.Toolbox()
        
        # function calls to chromosome intialization (random vs intelligent assignment)
        #toolbox.register("rand_float", np.random.uniform)
        #toolbox.register("assign_guess", self.assign_UFF_starting) 
        
        # create individual intialization method (random vs intelligent assignment)
        toolbox.register("individual", self.generateES, self.opt_vec, creator.Individual, creator.Strategy,
                                                                                IND_SIZE,
                                                                                MIN_VALUE,
                                                                                MAX_VALUE,
                                                                                MIN_STRATEGY,
                                                                                MAX_STRATEGY)
        #toolbox.register("individual", toolbox.assign_guess, creator.Individual)
        
        
        
        # objective function for this minimization 
        # toolbox.register("evaluate", self.deap_multi_evalFitness)
        toolbox.register("evaluate", self.construct_curr_UC_GA)
        
        # define evolution strategies
        toolbox.register("mate", tools.cxESBlend, alpha=0.5)
        toolbox.decorate("mate", self.checkStrategy(MIN_VALUE,
                                               MAX_VALUE,
                                               MAX_STRATEGY,
                                               MAX_STRATEGY)
                        )

        ###toolbox.register("mutate", tools.mutPolynomialBounded, eta = 0.0001, low = 0.0, up = 10000.0, indpb = 0.1)
        toolbox.register("mutate", tools.mutESLogNormal, c = 1.0, indpb = 0.9)
        toolbox.decorate("mutate", self.checkStrategy(MIN_VALUE,
                                                 MAX_VALUE,
                                                 MAX_STRATEGY,
                                                 MAX_STRATEGY)
                        )
        ###toolbox.register("mutate", tools.mutESLogNormal, c = 1, indpb = 0.1)
        
        toolbox.register("select", tools.selTournament, tournsize = int(POP_SIZE/2))
        ###toolbox.register("select", tools.selTournament, k = 10, tournsize = 64)
        
        
        # parallelize or no
        #pool = multiprocessing.Pool(processes = 7)
        #toolbox.register("map", pool.map)
        
        
        
        # create a population of individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n = POP_SIZE)
        population = toolbox.population()

        # create islands to contain distinct populations
        islands = [toolbox.population() for i in range(NISLANDS)]
        
        # create a hall of fame for each island
        hofsize = max(1, int(POP_SIZE/10))
        famous = [tools.HallOfFame(maxsize = hofsize) for i in range(NISLANDS)]
        
        # create a stats log for each island
        stats = [tools.Statistics(lambda ind: ind.fitness.values) for i in range(NISLANDS)]
        
        for i in range(NISLANDS):
            stats[i].register("avg", np.mean)
            stats[i].register("std", np.std)
            stats[i].register("min", np.min)
            stats[i].register("max", np.max)
        
        
        # MU, LAMDA parameters
        MU, LAMBDA = POP_SIZE, POP_SIZE*2
        
        # run optimization with periodic migration between islands
        for i in range(int(NGEN/MIG_FREQ)):
            print("----------------")
            print("Evolution period: " + str(i))
            print("----------------")
            for k in range(len(islands)):
                print("------------------------")
                print("Island " + str(k) + " evolution:")
                print("------------------------")
                #islands[k], log = algorithms.eaGenerateUpdate(toolbox, ngen = MIG_FREQ, halloffame = famous[k], stats = stats[k])
                islands[k], log = algorithms.eaMuCommaLambda(islands[k], toolbox, mu=MU, lambda_ = LAMBDA, cxpb = 0.4, mutpb = 0.6, ngen = MIG_FREQ, halloffame = famous[k], stats = stats[k])
            print("---------------")
            print("MIGRATION!")
            print("---------------")
            self.custom_migRing(islands, 10, tools.selBest, replacement = tools.selWorst)
        
        # Create final population for the last run
        final_famous = tools.HallOfFame(maxsize = 1)
        final_stats = tools.Statistics(lambda ind: ind.fitness.values)
        final_stats.register("avg", np.mean)
        final_stats.register("std", np.std)
        final_stats.register("min", np.min)
        final_stats.register("max", np.max)
        toolbox.register("final_population", tools.initRepeat, list, toolbox.individual, n = hofsize * NISLANDS)
        final_population = toolbox.final_population()
        
        # copy over each island's famous individuals into last 
        for i in range(NISLANDS):
            for j in range(hofsize):
                final_population[i*j + j] = famous[i][j]
        
        # make sure our ultimate hall of fame starts out as the best we've ever seen
        final_famous.update(final_population)
        
        # reset MU, LAMBDA and rerun final evolution
        MU, LAMBDA = hofsize*NISLANDS, hofsize*NISLANDS*2
        final_pop, log = algorithms.eaMuCommaLambda(final_population, toolbox, mu=MU, lambda_ = LAMBDA, cxpb = 0.4, mutpb = 0.6, ngen = MIG_FREQ, halloffame = final_famous, stats = final_stats)


        self.opt_vec = np.array(final_famous[0])



    def write_results(self):
        """
        Print optimization state for all vars
        """

        print("(1) Scale factor, F: %f" % (self.opt_vec[0]))
        print("(2) Vector of rod shifts, dCs: %s " % (self.opt_vec[1:1+len(self.rods)]))
        print("(3) Molecular transformations:")
        for i in range(len(self.cxns)):
            conn_start_ind = 1 + len(self.rods) + i*6 + i*len(self.cxns[i])
            print("     Transform %d: %s" % (i, self.opt_vec[conn_start_ind:conn_start_ind+6]))
            print("     Periodic images of cxns: %s" % (str(self.opt_vec[conn_start_ind + 6: conn_start_ind + 6 + len(self.cxns[i])])))



    def construct_final_UC_SVD(self):
        print("Optimal permuations are:")
        print(self.opt_perms)
        self.opt_vec[0]# *= 2
        to_cartesian = self.frame.update_UC_matrix(self.opt_vec[0], self.twod_direct)
        to_fractional = np.linalg.inv(to_cartesian)

        oriented = [np.dot(self.opt_mol_fit[i],self.mol.molecule) for i in range(len(self.cxns))]
        # produce optimized orientation
        for i in range(len(oriented)):
            oriented[i] = np.dot(to_fractional, oriented[i][0:3,:])
            oriented[i] = self.frame.modGroupUC(oriented[i][0:3,:])
        
        # produced optimized rod shift
        # for i in range(len(rods)):
        #     for j in range(len(rods[i])):
        #         self.rod_coords_abc[i][self.oned_direct,j] += self.opt_vec[1+i]
        #     self.rod_coords_abc[i] = self.frame.modGroupUC(self.rod_coords_abc[i])


        final_rods_abc = []
        for i in range(len(self.rods)):
            this_new_abc = np.copy(self.rod_ref_abc[i])

            # start of this cxn info in the optimization vec
            conn_start_ind = 1 + i

            for j in range(len(self.rods[i])):

                # shift ref pt based on curr val of rod shift
                # print(xuse[1 + self.connect_to_rod[i][j]])
                this_new_abc[self.oned_direct,j] += self.opt_vec[1 + i]


            # get the new xyz
            this_new_xyz = np.dot(to_cartesian, this_new_abc)
            # apply shift (this is the fixed relative positions in rod constraint)
            # non-trivial if we have non perpendicular oned_direct, but we took care of this
            # in self.get_rod_coords()
            this_shifted_xyz = this_new_xyz + self.rod_disp_xyz[i]

            
            # use modified UC matrix to get back abc coords
            this_shifted_abc = np.dot(to_fractional, this_shifted_xyz)
            # mod rods back into the UC if they left
            this_modded_abc = self.frame.modGroupUC(this_shifted_abc)
            #final_rods_abc.append(this_modded_abc)
            self.rod_coords_abc[i] = this_modded_abc

        # modify UC parameters
        final_a = self.frame.a
        final_b = self.frame.b
        final_c = self.frame.c
        for direct in self.twod_direct:
             if(direct == 0):
                 final_a = self.frame.a * self.opt_vec[0]
             elif(direct == 1):
                 final_b = self.frame.b * self.opt_vec[0]
             elif(direct == 2):
                 final_c = self.frame.c * self.opt_vec[0]

        # create lists for final atomic coords/labels
        final_ra = [] 
        final_rb = [] 
        final_rc = []
        final_atmtype = []
        final_label = []
        overall_ind = 0

        # add oriented molecules
        for i in range(len(oriented)):
            for j in range(np.shape(oriented[i])[1]):
                final_atmtype.append(self.mol.labels[j])
                final_label.append(self.mol.labels[j]+str(overall_ind))
                overall_ind += 1

                final_ra.append(float("{0:.6f}".format(oriented[i][0,j])))
                final_rb.append(float("{0:.6f}".format(oriented[i][1,j])))
                final_rc.append(float("{0:.6f}".format(oriented[i][2,j])))
                #final_ra.append(oriented[i][0,j])
                #final_rb.append(oriented[i][1,j])
                #final_rc.append(oriented[i][2,j])

        # add shifted rods
        for i in range(len(rods)):
            for j in range(len(rods[i])):
                final_atmtype.append(self.rod_atmtype[i][j])
                final_label.append(self.rod_atmtype[i][j] + str(overall_ind))
                overall_ind += 1

                final_ra.append(float("{0:.6f}".format(self.rod_coords_abc[i][0,j])))
                final_rb.append(float("{0:.6f}".format(self.rod_coords_abc[i][1,j])))
                final_rc.append(float("{0:.6f}".format(self.rod_coords_abc[i][2,j])))
                #final_ra.append(self.rod_coords_abc[i][0,j])
                #final_rb.append(self.rod_coords_abc[i][1,j])
                #final_rc.append(self.rod_coords_abc[i][2,j])

        self.frame.reconstruct_cif(final_a, final_b, final_c, final_ra, final_rb, final_rc, 
                                   final_label, final_atmtype, self.mol.molname)


    def construct_final_UC(self):
        self.opt_vec[0]# *= 2
        to_cartesian = self.frame.update_UC_matrix(self.opt_vec[0], self.twod_direct)
        to_fractional = np.linalg.inv(to_cartesian)
        

        # get the current oriented and translated ligands

        oriented = self.orient_molecule_and_translate(self.opt_vec)
        #print(oriented)


        # produce optimized orientation
        for i in range(len(oriented)):
            oriented[i] = np.dot(to_fractional, oriented[i][0:3,:])
            oriented[i] = self.frame.modGroupUC(oriented[i][0:3,:])
        
        # produced optimized rod shift
        # for i in range(len(rods)):
        #     for j in range(len(rods[i])):
        #         self.rod_coords_abc[i][self.oned_direct,j] += self.opt_vec[1+i]
        #     self.rod_coords_abc[i] = self.frame.modGroupUC(self.rod_coords_abc[i])


        final_rods_abc = []
        for i in range(len(self.rods)):
            this_new_abc = np.copy(self.rod_ref_abc[i])

            # start of this cxn info in the optimization vec
            conn_start_ind = 1 + len(self.rods) + i*6 + i*len(self.cxns[i])

            for j in range(len(self.rods[i])):

                # shift ref pt based on curr val of rod shift
                # print(xuse[1 + self.connect_to_rod[i][j]])
                this_new_abc[self.oned_direct,j] += self.opt_vec[1 + i]

                # shift ref pt as well based on the periodic image indicated in the opt_Vec
                periodic_image = round(self.opt_vec[conn_start_ind + 6 + j], 0)
                this_new_abc[self.oned_direct,j] += periodic_image

            # get the new xyz
            this_new_xyz = np.dot(to_cartesian, this_new_abc)
            # apply shift (this is the fixed relative positions in rod constraint)
            # non-trivial if we have non perpendicular oned_direct, but we took care of this
            # in self.get_rod_coords()
            this_shifted_xyz = this_new_xyz + self.rod_disp_xyz[i]

            
            # use modified UC matrix to get back abc coords
            this_shifted_abc = np.dot(to_fractional, this_shifted_xyz)
            # mod rods back into the UC if they left
            this_modded_abc = self.frame.modGroupUC(this_shifted_abc)
            #final_rods_abc.append(this_modded_abc)
            self.rod_coords_abc[i] = this_modded_abc




        # modify UC parameters
        final_a = self.frame.a
        final_b = self.frame.b
        final_c = self.frame.c
        for direct in self.twod_direct:
             if(direct == 0):
                 final_a = self.frame.a * self.opt_vec[0]
             elif(direct == 1):
                 final_b = self.frame.b * self.opt_vec[0]
             elif(direct == 2):
                 final_c = self.frame.c * self.opt_vec[0]

        # create lists for final atomic coords/labels
        final_ra = [] 
        final_rb = [] 
        final_rc = []
        final_atmtype = []
        final_label = []
        overall_ind = 0

        # add oriented molecules
        for i in range(len(oriented)):
            for j in range(np.shape(oriented[i])[1]):
                final_atmtype.append(self.mol.labels[j])
                final_label.append(self.mol.labels[j]+str(overall_ind))
                overall_ind += 1

                final_ra.append(float("{0:.6f}".format(oriented[i][0,j])))
                final_rb.append(float("{0:.6f}".format(oriented[i][1,j])))
                final_rc.append(float("{0:.6f}".format(oriented[i][2,j])))
                #final_ra.append(self.rod_coords_abc[i][0,j])
                #final_rb.append(self.rod_coords_abc[i][1,j])
                #final_rc.append(self.rod_coords_abc[i][2,j])

        # add shifted rods
        for i in range(len(rods)):
            for j in range(len(rods[i])):
                final_atmtype.append(self.rod_atmtype[i][j])
                final_label.append(self.rod_atmtype[i][j] + str(overall_ind))
                overall_ind += 1
                final_ra.append(float("{0:.6f}".format(self.rod_coords_abc[i][0,j])))
                final_rb.append(float("{0:.6f}".format(self.rod_coords_abc[i][1,j])))
                final_rc.append(float("{0:.6f}".format(self.rod_coords_abc[i][2,j])))
                #final_ra.append(self.rod_coords_abc[i][0,j])
                #final_rb.append(self.rod_coords_abc[i][1,j])
                #final_rc.append(self.rod_coords_abc[i][2,j])

        self.frame.reconstruct_cif(final_a, final_b, final_c, final_ra, final_rb, final_rc, 
                                   final_label, final_atmtype, self.mol.molname)

        


if(__name__ == "__main__"):
    # NOTE CWD must be the directory that has an input file onedMOF.input and then all the data files
    # described in oneDMOF.input
    framework_name, molecule_name, dimensionality, rods, rod_centers, cxns, connect_to_rod = parse_input()

    assemble = Assembly(framework_name, molecule_name, dimensionality, rods, rod_centers, cxns, connect_to_rod)
