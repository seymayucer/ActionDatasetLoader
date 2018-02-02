import collections
import numpy as np
import helpers
from helpers import logger

#    Florence Dataset parameters
#    9 action type - 15 joint
#    body part id's:
#                1-torso 2-left arm 3-right arm 4-left leg 5-right leg

max_frame_num = 35
frame_size = 45
num_classes = 9
body_part_ids = [[0, 1, 2], [3, 4, 5],
                   [6, 7, 8], [9, 10, 11],
                   [12, 13, 14]]

logger = helpers.logging.getLogger(__name__)
def read(data_dir, subject_id, split=False):
    logger.info('Loading Florence Data')
    logger.info('Data directory: %s' % data_dir)
    data, labels, lens, subjects = [], [], [], []
    florence = np.loadtxt(data_dir)

    frame_len = florence[:, 0:1].flatten()
    subject_array = florence[:, 1:2].flatten()
    label_array = florence[:, 2:3].flatten()
    counts = collections.Counter(frame_len)

    first, second = 0, 0
    for frame_num in counts:
        second += counts[frame_num]

        # frame normalizer
        action = florence[first:second][:, 3:]
        new_action = []
        for frame in action:
            frame = helpers.frame_normalizer(frame=frame, frame_size=frame_size) #frame normalizer

            new_action.append(frame)

        data.append(new_action)

        lens.append(counts[frame_num])
        labels.append(int(label_array[first]))
        subjects.append(int(subject_array[first]))
        first = second

    data = np.asarray(data)
    labels = np.asarray(labels) - 1
    lens = np.asarray(lens)
    subjects = np.asarray(subjects)
    logger.info('initial shapes data-label-len: %s %s %s' % (data.shape, labels.shape, lens.shape))
    data = helpers.dataset_normalizer(data)

    if split:
        return helpers.test_train_splitter(subject_id, data, labels, lens, subjects)

    else:
        return data, labels, lens
