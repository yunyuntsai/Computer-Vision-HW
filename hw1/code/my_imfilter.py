import numpy as np

def my_imfilter(image, imfilter):
    """function which imitates the default behavior of the build in scipy.misc.imfilter function.

    Input:
        image: A 3d array represent the input image.
        imfilter: The gaussian filter.
    Output:
        output: The filtered image.
    """
    ###################################################################################
    # TODO:                                                                           #
    # This function is intended to behave like the scipy.ndimage.filters.correlate    #
    # (2-D correlation is related to 2-D convolution by a 180 degree rotation         #
    # of the filter matrix.)                                                          #
    # Your function should work for color images. Simply filter each color            #
    # channel independently.                                                          #
    # Your function should work for filters of any width and height                   #
    # combination, as long as the width and height are odd (e.g. 1, 7, 9). This       #
    # restriction makes it unambigious which pixel in the filter is the center        #
    # pixel.                                                                          #
    # Boundary handling can be tricky. The filter can't be centered on pixels         #
    # at the image boundary without parts of the filter being out of bounds. You      #
    # should simply recreate the default behavior of scipy.signal.convolve2d --       #
    # pad the input image with zeros, and return a filtered image which matches the   #
    # input resolution. A better approach is to mirror the image content over the     #
    # boundaries for padding.                                                         #
    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can #
    # see the desired behavior.                                                       #
    # When you write your actual solution, you can't use the convolution functions    #
    # from numpy scipy ... etc. (e.g. numpy.convolve, scipy.signal)                   #
    # Simply loop over all the pixels and do the actual computation.                  #
    # It might be slow.                                                               #
    ###################################################################################
    ###################################################################################
    # NOTE:                                                                           #
    # Some useful functions                                                           #
    #     numpy.pad or numpy.lib.pad                                                  #
    # #################################################################################

    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can
    # see the desired behavior.
    w1 = image.shape[0]
    h1 = image.shape[1]
    ch1 = image.shape[2]
    w2 = imfilter.shape[0]
    h2 = imfilter.shape[1]
    pad_w1 = int((w2-1)/2)
    pad_h1 = int((h2-1)/2)
    
    #print("pad w1: ",pad_w1)
    #print("pad w2: ",pad_h1)
    
    pading_R=np.pad(image[:,:,0],((pad_h1,pad_h1),(pad_w1,pad_w1)),'constant')
    pading_G=np.pad(image[:,:,1],((pad_h1,pad_h1),(pad_w1,pad_w1)),'constant')
    pading_B=np.pad(image[:,:,2],((pad_h1,pad_h1),(pad_w1,pad_w1)),'constant')
    #print("after padding: ", pading_R.shape)
    output = np.zeros_like(image)
    
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            output[i][j][0] = sum(sum(np.multiply(imfilter,pading_R[i:i+imfilter.shape[0],j:j+imfilter.shape[1]])))
            output[i][j][1] = sum(sum(np.multiply(imfilter,pading_G[i:i+imfilter.shape[0],j:j+imfilter.shape[1]])))
            output[i][j][2] = sum(sum(np.multiply(imfilter,pading_B[i:i+imfilter.shape[0],j:j+imfilter.shape[1]])))
            
    #output = np.zeros_like(image)
    #import scipy.ndimage as ndimage
    #output = np.zeros_like(image)
    #for ch in range(image.shape[2]):
     #   output[:,:,ch] = ndimage.filters.correlate(image[:,:,ch], imfilter, mode='constant')
    
    ###################################################################################
    #                                 END OF YOUR CODE                                #
    ###################################################################################
    return output