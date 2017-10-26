from __future__ import print_function

from os import listdir
from os.path import join

import numpy as np

import train_test_splitter,helpers


def read(data_dir,split=False):
    print('Loading NTU 3D Data, data directory %s' % data_dir)
    data,labels,lens=[],[],[]
    filenames = []
    documents = [join(data_dir, d)
                 for d in sorted(listdir(data_dir))]
    filenames.extend(documents)
    filenames = np.array(filenames)
    action_id = 0

    for file in filenames:
        action, one_act_label = [], []
        with open(file, 'r') as f:
            frame_count = f.readline().split(' ')
            frame_count = int(frame_count[0])
            for frame in range(0, frame_count):

                body_count = f.readline().split(' ')
                body_count = int(body_count[0])
                for b in range(0, body_count):

                    body_jointCount = f.readline().split(' ')
                    body_jointCount = int(body_jointCount[0])

                    # no of  joints(25)
                    a_frame = []
                    for j in range(0, body_jointCount):
                        jointinfo = f.readline().split(' ')
                        # 3D location of the joint j
                        a_frame.append(float(jointinfo[0]))  # x
                        a_frame.append(float(jointinfo[1]))  # y
                        a_frame.append(float(jointinfo[2]))  # z
                    if (b == 0):  # take one subject move
                        a_frame= helpers.frame_normalizer(np.asarray(a_frame))  #if you need
                        #print(a_frame)
                        action.append(a_frame)
                        one_act_label.append(helpers.full_fname2_str(data_dir, file, 'A'))


            lens.append(len(action))
            labels.append(helpers.full_fname2_str(data_dir, file, 'A'))
            action = np.asarray(action).reshape(len(action),75)
            data.append(action)
            action_id += 1


    data = np.asarray(data)
    labels = np.asarray(labels)
    lens = np.asarray(lens)

    #if you need
    data= helpers.dataset_normalizer(data)

    print('data shape: %s, label shape: %s,lens shape %s' % (data.shape, labels.shape, lens.shape))
    if split:
        return train_test_splitter.test_train_splitter_NTU(data, labels, lens)

    else:
        return data,labels,lens

