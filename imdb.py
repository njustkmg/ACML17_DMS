from __future__ import print_function
from six.moves import xrange
import six.moves.cPickle as pickle

import gzip
import os
from pyflann import *
import numpy
import theano
import scipy.io as sio
import sys, os
# pwd = sys.path[0]

def masklist(modal_len, feanum):
    '''
    obtain mask matrix, whose size is num_of_modal * feanum,
    the jth element in ith row is 1 represents the ith modal has this feature
    model_len: represent the number of feature in each modal
    feanum: the dimension of feature
    '''
    # mask = disturb * random.random(size = (len(modal_len), feanum))
    mask = numpy.zeros([len(modal_len), feanum], dtype = 'int32')
    index = 0
    for i in range(len(modal_len)):
        mask[i, index:index + modal_len[i]] = 1
        index += modal_len[i]
    return mask

def get_dataset_file(dataname):
    '''
    obtain data from .mat file
    '''
    datapath = './data/' + dataname
    dataset = sio.loadmat(datapath)
    data = dataset.get('dat')
    modal_len = data['flens'][0][0][0]
    trainx = data['train'][0][0].T
    trainy = data['trainlabel'][0][0][0]
    testx = data['test'][0][0].T
    testy = data['testlabel'][0][0][0]
    n_samples = len(trainy)
    sidx = numpy.random.permutation(n_samples)
    trainx = [trainx[s] for s in sidx]
    trainy = [trainy[s] for s in sidx]
    trainx = numpy.array(trainx)
    trainy = numpy.array(trainy)
    # kdtree = spatial.cKDTree(trainx, leafsize=10)
    print('trainx', trainx.shape)
    print('trainy', trainy.shape)
    print('testx', testx.shape)
    print('testy', testy.shape)
    return (trainx,trainy,testx,testy,modal_len)

def load_data(valid_portion=0.3):

    numIter = 100
    (trainx,trainlabel,testx,testlabel,model_len
        ) = get_dataset_file('sat_clear.mat')
    print(model_len)
    trainlabel = numpy.array(trainlabel)
    testlabel = numpy.array(testlabel)
    fea_dim = trainx.shape[1]
    mask = masklist(model_len, fea_dim)
    train_set_y = list(trainlabel)
    test_set_y = list(testlabel)
    train_fea_x = trainx
    flann = FLANN()
    params = flann.build_index(train_fea_x, target_precision=0.0, log_level = 'info')
    for k in range(numIter):
        # result1 = kdtree.query(train_fea_x, 4)
        result2 = flann.nn_index(train_fea_x, 4, checks=params['checks'])
    train_n3 = result2[0][:,1:]
    # sio.savemat('../../data/sat_neigh_clear_2part.mat', {'train_n3':train_n3})
    print('train_n3', train_n3.shape)
    # exit()
    # dd  = sio.loadmat('../../data/sat_neigh_clear_2part.mat')
    # train_n3 = dd['train_n3']
    # print('train_n3', train_n3.shape)
    # exit()
    train_set = (train_fea_x, numpy.array(train_set_y) - 1)
    test_set = (testx, numpy.array(test_set_y) - 1)
    valid_set = test_set
    return train_set, valid_set, test_set, mask, model_len, train_n3
