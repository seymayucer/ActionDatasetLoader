import numpy as np
import collections

data_dir='/home/sym-gtu/Data/Florence_3D_Actions/Florence_dataset_WorldCoordinates.txt'
def read():
    print('Loading Florence Data, data directory %s'%data_dir)
    data, labels, lens = [], [], []
    florence=np.loadtxt(data_dir)

    label_array=florence[:,1:2].flatten()
    frame_len=florence[:,0:1].flatten()
    counts=collections.Counter(frame_len)


    first,second=0,0
    for frame_num in counts:
        #print(frame_num,counts[frame_num])
        second+=counts[frame_num]
        data.append(florence[first:second])
        lens.append(counts[frame_num])
        labels.append(int(label_array[first]))
        first=second

    print labels,label_array
    data=np.asarray(data)
    labels=np.asarray(labels)
    lens = np.asarray(lens)

    print('data shape: %s, label shape: %s, action lens shape %s:'%(data.shape,labels.shape,lens.shape))
    return (data,labels,lens)


read()