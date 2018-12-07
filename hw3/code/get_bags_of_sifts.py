from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
import pdb

def get_bags_of_sifts(image_paths):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
                                                                    
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    
    # You will want to construct SIFT features here in the same way you        #
    # did in build_vocabulary.m (except for possibly changing the sampling     #
    # rate) and then assign each local feature to its nearest cluster center   #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''
    vocab = pickle.load(open('vocab.pkl','rb'))
    image_feats = []

    start_time = time()

    for path_idx in range(len(image_paths)):
        img = np.asarray(Image.open(image_paths[path_idx]),dtype='float32')
        frames,descriptors = dsift(img,step=[5,5],fast=True)
        dist = distance.cdist(descriptors,vocab,'euclidean')
        target_idx = np.argmin(dist,axis=1)
        image_feats.append(np.histogram(target_idx,np.arange(vocab.shape[0] + 1),density=True)[0].reshape([1,-1]))

    image_feats = np.vstack([image_feats])

    end_time = time()
    print("It takes ",(start_time - end_time)," to compute feature.")
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats
