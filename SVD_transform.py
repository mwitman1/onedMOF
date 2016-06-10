#! usr/bin/python


import numpy as np
from numpy import linalg as LA
import math
import itertools as it

def compute_ROT_TRANS(src, dest):
    # for now let's just assume the program accepts two numpy Nx3 arrays of source and destination points
    print str(src)
    print str(dest)

    if (len(src) != len(dest)):
        print "WARNING! Source and destination points are not the same length! Exiting..."
        exit(-1)
    else:
        point_size = len(src)

        # initialize vectors and matrices needed for decomposition
        
        # vectors from the origin to the barycentric center of each point cloud
        cen_src = np.zeros((3))
        cen_dest = np.zeros((3))

        # Source and destination matrices that have been offset s.t. barycentric center = (0,0,0)
        S = np.zeros((point_size,3))
        D = np.zeros((point_size,3))

    for i in range(0,point_size):
        cen_src[0] += src[i,0]
        cen_src[1] += src[i,1]
        cen_src[2] += src[i,2]
        cen_dest[0] += dest[i,0]
        cen_dest[1] += dest[i,1]
        cen_dest[2] += dest[i,2]

    cen_src /= point_size
    cen_dest /= point_size

    for i in range(0,point_size):
        for j in range(0,3):
            S[i,j] = src[i,j] - cen_src[j]
            D[i,j] = dest[i,j] - cen_dest[j]

    Dt = np.transpose(D)
    H = np.dot(Dt,S)

    U, s, V = LA.svd(H, full_matrices = True, compute_uv = True)

    Vt = np.transpose(V)
    R = np.dot(U, Vt)
    t = cen_dest - np.dot(R, cen_src)

    return R,t


def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.dot(np.transpose(AA),BB)

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    #if np.linalg.det(R) < 0:
    #   print "Reflection detected"
    #   Vt[2,:] *= -1
    #   R = np.dot(Vt.T, U.T)

    t = -np.dot(R,centroid_A.T) + centroid_B.T

    #print t

    return R, t

    
def transform_data(A, B):
    """
    Function that accounts for the fact that we have to SVD all
    permutations between cxn pts to know which one is the best

    src = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    dest = np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])

    ROT, TRANS = compute_ROT_TRANS(src, dest)

    print str(ROT)
    print str(TRANS)
    """

    # Random rotation and translation
    #R = np.mat(np.random.rand(3,3))
    #t = np.mat(np.random.rand(3,1))

    # make R a proper rotation matrix, force orthonormal
    #U, S, Vt = np.linalg.svd(R)
    #print str(new)
    #R = U*Vt

    # remove reflection
    #if np.linalg.det(R) < 0:
    #   Vt[2,:] *= -1
    #   R = U*Vt

    # number of points
    #n = 10

    #A = np.mat(np.random.rand(n,3));
    #B = R*A.T + np.tile(t, (1, n))
    #A = np.mat([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    #B = np.mat([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
    """DEBUG
    print "Points A"
    print A
    print ""

    print "Points B"
    print B
    print ""
    """
    template = np.array(A)
    permutations = list(it.permutations([i for i in range(0,len(A))]))
    print(permutations)
    print(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))

    lowest_rmse = 10000000
    index_lowest_rmse = -1
    
    counter = 0
    for p in permutations:
        for i in range(0, len(p)):
            A[counter] = template[p[i]]
            counter += 1
        counter = 0
            
        #B = B.T;
        #print A
        # recover the transformation
        curr_R, curr_t = rigid_transform_3D(A, B)



        #Return transformed points

        print(curr_R, curr_t)
        # NOTE depends if A was passed in as np array or list 
        # list input:
        A2 = np.dot(curr_R,A.T) + np.tile(curr_t, (1, len(A)))
        # np array input
        # A2 = np.dot(curr_R,A.T) + np.tile(curr_t, (1, np.shape(A)[0]))
        A2 = A2.T
        
        # Find the error
        err = A2 - B

        err = np.multiply(err, err)
        err = sum(err)
        rmse = np.sqrt(err/len(A));
        #print rmse 
        if LA.norm(rmse) < lowest_rmse:
            ret_R = np.array(curr_R)
            ret_t = np.array(curr_t)

            lowest_rmse = LA.norm(rmse)

    if (lowest_rmse > 1.0):
        print "Warning, RMSE of rigid 3D transform via SVD failed"
        print "RMSE=" + str(rmse)
        print "Structure likely not valid..."
        #exit(-1)
        
            

    return ret_R, ret_t, lowest_rmse
    """ DEBUG





    print "Points A new"
    print A2
    print ""

    print "Points B"
    print B
    print ""

    print "Rotation"
    print R
    print ""

    print "Translation"
    print t
    print ""

    print "RMSE:", rmse
    print "If RMSE is near zero, the function is correct!"
    """
    


if __name__== "__main__":
    print "Debugging..."
    test_data()
    
