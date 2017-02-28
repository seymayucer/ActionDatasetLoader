import numpy as np
import collections
import common
data_dir='/home/sym-gtu/Data/Florence_3D_Actions/Florence_dataset_WorldCoordinates.txt'


def read():
    print('Loading Florence Data, data directory %s'%data_dir)
    data, labels, lens,subjects = [], [], [],[]
    florence=np.loadtxt(data_dir)

    frame_len=florence[:,0:1].flatten()
    label_array = florence[:, 1:2].flatten()
    subjects_array=florence[:,2:3].flatten()
    counts=collections.Counter(frame_len)


    first,second=0,0
    for frame_num in counts:
        second+=counts[frame_num]
        data.append(florence[first:second][:,3:])
        lens.append(counts[frame_num])
        labels.append(int(label_array[first]))
        subjects.append(int(subjects_array[first]))
        first=second

    data=np.asarray(data)
    labels=np.asarray(labels)
    lens = np.asarray(lens)
    subjects=np.asarray(subjects)

    print('data shape: %s, label shape: %s, action lens shape %s: %s'%(data.shape,labels.shape,lens.shape,subjects.shape))
    return common.test_train_splitter_MSR_FLOR(1, data, labels, lens, subjects)
