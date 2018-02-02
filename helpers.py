import numpy as np


# some helpers
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def full_fname2_str(data_dir, fname, sep_char):
    fnametostr = ''.join(fname).replace(data_dir, '')
    ind = int(fnametostr.index(sep_char))
    label = int(fnametostr[ind + 1:ind + 3])
    return label


def frame_normalizer(frame, frame_size):
    # print('...frame normalizer')
    assert frame.shape[0] == frame_size
    frame = frame.reshape(frame_size / 3, 3)
    spine_mid = frame[1]
    new_frame = []
    j = 0
    for joint in frame:
        new_frame.append(joint - spine_mid)
        j += 1
    new_frame = np.asarray(new_frame)
    return (list(new_frame.flatten()))


def dataset_normalizer(data):
    logger.info('Dataset normalizer')
    min_list = []
    max_list = []
    for act in data:
        max_list.append(np.amax(act))
        min_list.append(np.amin(act))

    # print('min num:', min(min_list), 'max num', max(max_list))
    min_num = min(min_list)
    max_num = max(max_list)
    new_data = []
    i = 0
    for act in data:
        act = (act + abs(min_num)) / (max_num + abs(min_num))
        new_data.append(act)
        i += 0

    return np.asarray(new_data)



# ntu actions
def get_action_name(label):
    act_names = 'drinkWater', 'eat meal-snack', 'brushing teeth', 'brushing hair', 'drop', 'pickup', 'throw', 'sitting down', 'standing up (from sitting position)', 'clapping', 'reading', 'writing', 'tear up paper', 'wear jacket', 'take off jacket', 'wear a shoe', 'take off a shoe', 'wear on glasses', 'take off glasses', 'put on a hat-cap', 'take off a hat-cap', 'cheer up', 'hand waving', 'kicking something', 'put something inside pocket - take out something from pocket', 'hopping (one foot jumping)', 'jump up', 'make a phone call-answer phone', ' ' \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'playing with phone-tablet', 'typing on a keyboard', 'pointing to something with finger', 'taking a selfie', 'check time (from watch)', 'rub two hands together', 'nod head-bow', 'shake head', 'wipe face', 'salute', 'put the palms together', 'cross hands in front (say stop)', 'sneeze-cough', 'staggering', 'falling', 'touch head (headache)', 'touch chest (stomachache-heart pain)', 'touch back (backache)', 'touch neck (neckache)', 'nausea or vomiting condition', 'use a fan (with hand or paper)-feeling warm', 'punching-slapping other person', 'kicking other person', 'pushing other person', 'pat on back of other person', 'point finger at the other person', 'hugging other person', 'giving something to other person', 'touch other persons pocket', 'handshaking', 'walking towards each other', 'walking apart from each other'
    print(label, act_names[label])
    return str(act_names[label])


# kinect 2 joints
def get_joint_names(joint_id):
    joints = 'SpineBase', 'SpineMid', 'Neck', 'Head', 'ShoulderLeft', 'ElbowLeft', 'WristLeft', 'HandLeft', 'ShoulderRight', 'ElbowRight', 'WristRight', 'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft', 'HipRight', 'KneeRight', 'AnkleRight', 'FootRight', 'SpineShoulder', 'HandTipLeft', 'ThumbLeft', 'HandTipRight', 'ThumbRight'
    return str(joints[joint_id])

import numpy as np
import bunch


def test_train_splitter(subject_id, data, labels, lens, subject):
    logger.info('Test-Train Cross Subject Splitting')
    indices=np.argwhere(subject==subject_id)
    other_indices = np.ones(len(subject), dtype=bool)
    other_indices[indices]=False
    indices=indices.flatten()

    train_data = data[other_indices]
    train_labels = labels[other_indices].flatten()
    train_lens = lens[other_indices].flatten()

    test_data = data[indices]
    test_labels = labels[indices].flatten()
    test_lens = lens[indices].flatten()


    dataset = bunch
    dataset.train_data, dataset.train_labels, dataset.train_lens = train_data, train_labels, train_lens
    dataset.test_data, dataset.test_labels, dataset.test_lens = test_data, test_labels, test_lens
    logger.info ('Train Size', dataset.train_data.shape, dataset.train_labels.shape, dataset.train_lens.shape)
    logger.info ('Test Size', dataset.test_data.shape, dataset.test_labels.shape, dataset.test_lens.shape)
    return dataset


def test_train_splitter_GTU(data, labels, lens, k):
    logger.info('Test-Train k-Cross Splitting',k)

    indices = np.arange(len(lens))
    test_indices = np.arange(k, len(lens), 10)
    mask = np.ones(len(indices), dtype=bool)
    mask[test_indices] = False
    train_indices = indices[mask]

    train_data = data[train_indices]
    train_labels = labels[train_indices]
    train_lens=lens[train_indices]

    test_data = data[test_indices]
    test_labels = labels[test_indices]
    test_lens = lens[test_indices]

    dataset = bunch
    dataset.train_data, dataset.train_labels, dataset.train_lens = train_data, train_labels, train_lens
    dataset.test_data, dataset.test_labels, dataset.test_lens = test_data, test_labels, test_lens
    logger.info ('Train Size',dataset.train_data.shape, dataset.train_labels.shape, dataset.train_lens.shape )
    logger.info ('Test Size', dataset.test_data.shape, dataset.test_labels.shape, dataset.test_lens.shape)

    return dataset



