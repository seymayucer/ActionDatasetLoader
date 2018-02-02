# -*- coding: utf-8 -*-
from __future__ import print_function
import sklearn
import numpy as np
from sklearn import datasets
import helpers
from helpers import logger
#    GTUAction 3D Dataset parameters
#    14 action type - 25 joint
#    body part id's:
#                1-torso 2-left arm 3-right arm 4-left leg 5-right leg

max_frame_num = 195
frame_size = 75
num_classes = 14
body_part_ids = [[3, 2, 20, 1, 0],
                   [4, 5, 6, 7, 21, 22],
                   [8, 9, 10, 11, 23, 24],
                   [12, 13, 14, 15],
                   [16, 17, 18, 19]]

logger = helpers.logging.getLogger(__name__)
def read(data_dir, split, subject_id):
    print('Loading SYMAct 3D Data, data directory %s' % data_dir)
    data, labels, lens = [], [], []
    dataset = sklearn.datasets.load_files(data_dir, shuffle=False)
    index = 0
    new_set = []
    for action in dataset.data:
        action = action.replace('\r\n ', ' ')
        action = action.split()
        action = np.asarray(action)
        action = action.astype(np.float)

        frame_size = len(action) / 75  # 25 iskeleton num x,y,z 3D points
        lens.append(frame_size)
        action = action.reshape(frame_size, 75)
        new_act = []
        for frame in action:
            new_frame = helpers.frame_normalizer(frame=frame, frame_size=75)
            new_act.append(new_frame)
        new_set.append(new_act)
        index += 1

    data = np.asarray(new_set)
    labels = np.asarray(dataset.target)
    lens = np.asarray(lens)
    data = helpers.dataset_normalizer(data)

    print('initial shapes [data label len]: %s %s %s' % (data.shape, labels.shape, lens.shape))
    if split:
        return helpers.test_train_splitter_GTU(data, labels, lens, subject_id)
    else:
        return data, labels, lens
