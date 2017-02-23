from __future__ import print_function
from os.path import join
from os import listdir
import numpy as np

data_dir='/home/sym-gtu/Data/MSR/MSRAction3DSkeletonReal3D'

def read():
    print('Loading NTU 3D Data, data directory %s' % data_dir)
    data,labels,lens=[],[],[]
    filenames = []
    documents = [join(data_dir, d)
                 for d in sorted(listdir(data_dir))]
    filenames.extend(documents)
    filenames = np.array(filenames)


    for file in filenames:
        action=np.loadtxt(file)[:,:3]
        print(action.shape)
        data.append(action)
        labels.append(full_fname2_str(file))
        frame_size = len(action) / 60 # 20 iskeleton num x,y,z 3D points
        lens.append(frame_size)

    data = np.asarray(data)
    labels = np.asarray(labels)
    lens = np.asarray(lens)

    print('data shape: %s, label shape: %s,lens shape %s' % (data.shape, labels.shape, lens.shape))
    return (data, labels, lens)

def full_fname2_str(fname):
    fnametostr = ''.join(fname).replace(data_dir, '')
    ind = int(fnametostr.index('a'))
    label = int(fnametostr[ind + 1:ind + 3])
    return label - 1



read()