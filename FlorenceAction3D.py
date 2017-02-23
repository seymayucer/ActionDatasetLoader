import numpy as np
import collections

data_dir='/home/sym-gtu/Data/Florence_3D_Actions/Florence_dataset_WorldCoordinates.txt'
def read():
    print('Loading Florence Data, data directory %s'%data_dir)
    florence=np.loadtxt(data_dir)

    labels=florence[:,1:2].flatten()
    frame_len=florence[:,0:1].flatten()
    counts=collections.Counter(frame_len)

    data,label=[],[]
    first,second=0,0
    for frame_num in counts:
        #print(frame_num,counts[frame_num])
        second+=counts[frame_num]
        data.append(florence[first:second])
        label.append(int(labels[first]))
        first=second

    print labels,label
    data=np.asarray(data)
    labels=np.asarray(label)
    print('data shape: %s, label shape: %s',data.shape,labels.shape)
    return (data,labels)