# -*- coding: utf-8 -*-
from __future__ import print_function
import sklearn

import numpy as np
from sklearn import datasets

#loads files in recursively,loads them in bunch object
data_dir = '/home/sym-gtu/Data/SYMAct3D'
def read():
    print('Loading SYMAct 3D Data, data directory %s' % data_dir)
    dataset= sklearn.datasets.load_files(data_dir)
    index=0
    lens=[]
    for action in dataset.data:
        action=action.replace('\r\n ', ' ')
        action=action.split()
        action = np.asarray(action)
        action = action.astype(np.float)
        # data.data[index]=map(float,action)
        frame_size=len(action)/75
        lens.append(frame_size)
        action=action.reshape(frame_size,75)
        np.savetxt('actionsample.txt',action)
        dataset.data[index] = action
        print('Action Id %d, Action shape %s'%(index,dataset.data[index].shape))
        index+=1

    print('data shape: %s, label shape: %s', np.asarray(dataset.data).shape,np.asarray(dataset.target).shape)
    return (dataset.data,dataset.target)

read()