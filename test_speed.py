#! /usr/bin/env python

import transformations as trans
import numpy as np

import timeit

np.set_printoptions(precision=3)

def test_speed():

    #a = np.array([[4,3,2,1],[4,3,2,1],[4,3,2,1]])
    #b = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])

    print(np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,1]]))

    a = [[0,0,1,1,0.5],
         [0,1,1,0, -1],
         [0,0,0,0,  0]]
    a_mod = [[0,0,1,1,0.5],
             [0,1,1,0, -1],
             [0,0,0,0,  0],
             [1,1,1,1,  1]]
    b_bad = [[0,1,0,1,0.5],
             [0,0,0,0,  0],
             [0,1,1,0, -1]]
    b_bad_mod = [[0,1,0,1,0.5],
                 [0,0,0,0,  0],
                 [0,1,1,0, -1], 
                 [1,1,1,1,  1]]
    b_good = [[0,0,1,1,0.5],
              [0,0,0,0,0  ],
              [0,1,1,0,-1 ]]
    b_good_mod = [[0,0,0.5,1,1],
                  [0,0,  0,0,0],
                  [0,1, -1,1,0], 
                  [1,1,  1,1,1]]
    #for i in range(20000):
    M_bad = trans.superimposition_matrix(a, b_bad)
    M_good = trans.superimposition_matrix(a, b_good)

    print(np.dot(M_bad, a_mod))
    print(np.dot(M_good, a_mod))

if __name__ == "__main__":
    test_speed()
    #timeit.timeit(test_speed, number = 1)
