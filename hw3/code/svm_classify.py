from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import pdb
import numpy as np



def svm_classify(train_image_feats, train_labels, test_image_feats):
    #################################################################################
    # TODO :                                                                        #
    # This function will train a set of linear SVMs for multi-class classification  #
    # and then use the learned linear classifiers to predict the category of        #
    # every test image.                                                             # 
    #################################################################################
    ##################################################################################
    # NOTE: Some useful functions                                                    #
    # LinearSVC :                                                                    #
    #   The multi-class svm classifier                                               #
    #        e.g. LinearSVC(C= ? , class_weight=None, dual=True, fit_intercept=True, #
    #                intercept_scaling=1, loss='squared_hinge', max_iter= ?,         #
    #                multi_class= ?, penalty='l2', random_state=0, tol= ?,           #
    #                verbose=0)                                                      #
    #                                                                                #
    #             C is the penalty term of svm classifier, your performance is highly#
    #          sensitive to the value of C.                                          #
    #   Train the classifier                                                         #
    #        e.g. classifier.fit(? , ?)                                              #
    #   Predict the results                                                          #
    #        e.g. classifier.predict( ? )                                            #
    ##################################################################################
    '''
    Input : 
        train_image_feats : training images features
        train_labels : training images labels
        test_image_feats : testing images features
    Output :
        Predict labels : a list of predict labels of testing images (Dtype = String).
    '''
    Train_image_feats = np.zeros((1500,400))
    Test_image_feats = np.zeros((1500,400))
    for i in range (1, 1500):
        imgs1 = train_image_feats[i][0]
        imgs2 = test_image_feats[i][0]
        Train_image_feats[i] = imgs1
        Test_image_feats[i] = imgs2

    clf = LinearSVC(C=10,class_weight=None,dual=True,fit_intercept=True,
                    intercept_scaling=1,loss='squared_hinge',max_iter=1000,
                    multi_class='ovr',penalty='l2',random_state=0,tol=1e-4,
                    verbose=0)

    clf.fit(Train_image_feats,train_labels)
    pred_label = clf.predict(Test_image_feats)
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return pred_label
