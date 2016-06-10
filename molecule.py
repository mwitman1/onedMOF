#! /usr/bin/env python
import numpy as np
import itertools

class Molecule(object):

    def __init__(self, filename):
        self.name = filename
        self.molname = filename[:-4]
        # NOTE fourth dimension is for affine transformation
        self.molecule = [[],[],[],[]]
        self.cxns = [[],[],[],[]]
        self.labels = []

        self.read_xyz(filename)
        self.compute_extent()
    

    def read_xyz(self, filename):
        xyzfile = open(filename, "r")
        lines = xyzfile.readlines()

        numatms = int(lines[0].strip().split()[0])

        
        for i in range(len(lines[2:])):
            parsed = lines[2+i].strip().split()

            if(parsed[0] != "Q"):
                self.labels.append(parsed[0])
                self.molecule[0].append(float(parsed[1]))
                self.molecule[1].append(float(parsed[2]))
                self.molecule[2].append(float(parsed[3]))
                self.molecule[3].append(1.0)
            else:
                self.cxns[0].append(float(parsed[1]))
                self.cxns[1].append(float(parsed[2]))
                self.cxns[2].append(float(parsed[3]))
                self.cxns[3].append(1.0)

        # properties of the molecule we need to access for the optimization
        self.molecule = np.array(self.molecule)            
        self.cxns = np.array(self.cxns)
        self.permutations = list(itertools.permutations([i for i in range(np.shape(self.cxns)[1])]))
        # good to have the center so we have a good starting point
        self.center = np.zeros((3))
        self.center[0] = np.average(self.cxns[0,:])
        self.center[1] = np.average(self.cxns[1,:])
        self.center[2] = np.average(self.cxns[2,:])

        self.opt_perm = -1

    def compute_extent(self):
        # get the largest distance between any two connection points in the molecule
        self.max_extent = 0.0
        for i in range(np.shape(self.cxns)[1]):
            for j in range(np.shape(self.cxns)[1]):
                if(i != j):
                    extent = np.linalg.norm(self.cxns[0:3,i]-self.cxns[0:3,j])
                    if(extent > self.max_extent):
                        self.max_extent = float(extent)

