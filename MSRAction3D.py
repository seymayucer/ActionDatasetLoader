from __future__ import print_function

from os import listdir
from os.path import join

import numpy as np

import train_test_splitter,helpers


def read(data_dir,split,subject_id):
    print('Loading MSR 3D Data, data directory %s' % data_dir)
    data, labels, lens, subjects = [], [], [], []
    filenames = []
    documents = [join(data_dir, d)
                 for d in sorted(listdir(data_dir))]
    filenames.extend(documents)
    filenames = np.array(filenames)

    for file in filenames:
        action = np.loadtxt(file)[:, :3].flatten()
        labels.append(helpers.full_fname2_str(data_dir, file, 'a'))
        frame_size = len(action) / 60  # 20 iskeleton num x,y,z 3D points
        lens.append(frame_size)
        action = np.asarray(action).reshape(frame_size, 60)
        data.append(action)
        subjects.append(helpers.full_fname2_str(data_dir, file, 's'))
        # print(action.shape,frame_size)
    data = np.asarray(data)
    labels = np.asarray(labels)
    lens = np.asarray(lens)
    subjects = np.asarray(subjects)
    # data, labels, lens, subjects = get_half(data, labels, lens, subjects)
    print('data shape: %s, label shape: %s,lens shape %s' % (data.shape, labels.shape, lens.shape))
    if split:
        return train_test_splitter.test_train_splitter_MSR_FLOR(subject_id,data, labels, lens, subjects)
    else:

        return (data,labels,lens)

