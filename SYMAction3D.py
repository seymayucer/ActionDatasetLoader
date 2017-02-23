# -*- coding: utf-8 -*-
from __future__ import print_function
import sklearn,bunch
import numpy as np
from sklearn import datasets
import common
#loads files in recursively,loads them in bunch object
data_dir = '/home/sym-gtu/Data/SYMAct3D'
def read():
    print('Loading SYMAct 3D Data, data directory %s' % data_dir)
    data, labels, lens = [], [], []
    dataset= sklearn.datasets.load_files(data_dir)
    index=0

    for action in dataset.data:
        action=action.replace('\r\n ', ' ')
        action=action.split()
        action = np.asarray(action)
        action = action.astype(np.float)

        frame_size=len(action)/75 # 25 iskeleton num x,y,z 3D points
        lens.append(frame_size)
        action=action.reshape(frame_size,75)
        dataset.data[index] = action
        #print('Id %d shape %s'%(index,dataset.data[index].shape)),
        index+=1

    data = np.asarray(dataset.data)
    labels = np.asarray(dataset.target)
    lens = np.asarray(lens)


    print('data shape: %s, label shape: %s,lens shape %s'%(data.shape,labels.shape,lens.shape))

    return common.test_train_splitter_SYM_NTU(data, labels, lens)




read()