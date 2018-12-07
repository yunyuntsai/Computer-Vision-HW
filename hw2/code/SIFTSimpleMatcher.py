# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 21:36:46 2017

@author: HGY
"""

import numpy as np
from scipy.io import loadmat


#%% SIFTSimpleMatcher function
def SIFTSimpleMatcher(descriptor1, descriptor2, THRESH=0.7):
    '''
    SIFTSimpleMatcher 
    Match one set of SIFT descriptors (descriptor1) to another set of
    descriptors (decriptor2). Each descriptor from descriptor1 can at
    most be matched to one member of descriptor2, but descriptors from
    descriptor2 can be matched more than once.
    
    Matches are determined as follows:
    For each descriptor vector in descriptor1, find the Euclidean distance
    between it and each descriptor vector in descriptor2. If the smallest
    distance is less than thresh*(the next smallest distance), we say that
    the two vectors are a match, and we add the row [d1 index, d2 index] to
    the "match" array.
    
    INPUT:
    - descriptor1: N1 * 128 matrix, each row is a SIFT descriptor.
    - descriptor2: N2 * 128 matrix, each row is a SIFT descriptor.
    - thresh: a given threshold of ratio. Typically 0.7
    
    OUTPUT:
    - Match: N * 2 matrix, each row is a match. For example, Match[k, :] = [i, j] means i-th descriptor in
        descriptor1 is matched to j-th descriptor in descriptor2.
    '''

    #############################################################################
    #                                                                           #
    #                              YOUR CODE HERE                               #
    #                                                                           #
    #############################################################################
    N1 = descriptor1.shape[0] #1128
    N2 = descriptor2.shape[0] #613
    #print (N1)
    #print (N2)
    match = []
    for idx,des_vec1 in enumerate(descriptor1):
        #print (len(des_vec1))
        repeat_tile = np.tile(des_vec1, (N2,1))
        distance =  np.linalg.norm(repeat_tile-descriptor2, ord=2, axis=1)
        
        dict = {}
        for i in range(0,N2):
            dict.update({i: distance[i]})
        
        #print( dict )
        sorted_by_value = sorted(dict.items(),key=lambda kv: kv[1])
        fst_value = sorted_by_value[0][1]
        fst_index = sorted_by_value[0][0]
        snd_value = sorted_by_value[1][1]
        snd_index = sorted_by_value[1][0]
        if(fst_value < THRESH * snd_value):
            match.append([idx, fst_index])
    match = np.array(match)
    #print(match)
    #############################################################################
    #                                                                           #
    #                             END OF YOUR CODE                              #
    #                                                                           #
    #############################################################################   
    
    return match
