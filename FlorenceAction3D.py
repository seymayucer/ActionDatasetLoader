import collections
import numpy as np

import train_test_splitter,helpers

def read(data_dir,subject_id,split=False):

    print('Loading Florence Data, data directory %s'%data_dir)
    data, labels, lens,subjects = [], [], [],[]
    florence=np.loadtxt(data_dir)

    frame_len=florence[:,0:1].flatten()
    subject_array = florence[:, 1:2].flatten()
    label_array=florence[:,2:3].flatten()
    counts=collections.Counter(frame_len)

    first,second=0,0
    for frame_num in counts:
        second+=counts[frame_num]

        #frame normalizer
        action =florence[first:second][:,3:]
        _action=[]
        for frame in action:

            frame=helpers.frame_normalizer(frame=frame)

            _action.append(frame)

        data.append(_action)

        lens.append(counts[frame_num])
        labels.append(int(label_array[first]))
        subjects.append(int(subject_array[first]))
        first=second

    data=np.asarray(data)
    labels=np.asarray(labels)
    lens = np.asarray(lens)
    subjects=np.asarray(subjects)
    print('data shape: %s, label shape: %s, action lens shape %s: %s' % (data.shape, labels.shape, lens.shape, subjects.shape))
    data=helpers.normalizer(data)
    labels=labels-1

    if split:
        return train_test_splitter.test_train_splitter_MSR_FLOR(subject_id, data, labels, lens, subjects)

    else:
        return data, labels, lens
