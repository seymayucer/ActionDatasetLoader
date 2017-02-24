from __future__ import print_function
from os.path import join
from os import listdir
import numpy as np
import common

data_dir='/home/sym-gtu/Data/NTU/NTUDemo/'

def read():
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
                        action.append(a_frame)
                        one_act_label.append(common.full_fname2_str(data_dir,file,'A'))


            lens.append(len(action))
            labels.append(common.full_fname2_str(data_dir,file,'A'))
            action = np.asarray(action).reshape(len(action),75 )
            data.append(action)
            action_id += 1


    data = np.asarray(data)

    labels = np.asarray(labels)
    lens = np.asarray(lens)

    print('data shape: %s, label shape: %s,lens shape %s' % (data.shape, labels.shape, lens.shape))
    return common.test_train_splitter_SYM_NTU(data, labels, lens)


