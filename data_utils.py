import numpy as np
import sys
from os import path
sys.path.append(path.abspath('../'))
import bunch
import workspace_generals
import FlorenceAction3D,GTUAction3D,MSRAction3D,NTUAction3D
from helpers import logger
import helpers
logger = helpers.logging.getLogger(__name__)
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def zero_padder(data, max_num, pad_frame_len):
    zero_frame = np.zeros(pad_frame_len)
    padded_data = []

    for act in data:
        new_act = list(act)
        for i in range(0, max_num - len(act)):
            new_act.append(zero_frame)
        padded_data.append(new_act)

    padded_data = np.asarray(padded_data)
    padded_data = padded_data.reshape(len(padded_data), max_num, pad_frame_len)
    logger.info("zero padded data shape %s", padded_data.shape)
    return padded_data


def get_data(one_hot, subject_id, split):
    if workspace_generals.DATA_FLAG == 'FLO':
        dataset = FlorenceAction3D.read(workspace_generals.FLO_data_dir,split=split,subject_id=subject_id)
        max_frame_num = FlorenceAction3D.max_frame_num
        frame_size = FlorenceAction3D.frame_size
        num_classes = FlorenceAction3D.num_classes
    elif workspace_generals.DATA_FLAG == 'GTU':

        dataset = GTUAction3D.read(workspace_generals.GTU_data_dir, split=split, subject_id=subject_id)
        max_frame_num = GTUAction3D.max_frame_num
        frame_size = GTUAction3D.frame_size
        num_classes = GTUAction3D.num_classes

    elif workspace_generals.DATA_FLAG == 'MSR':

        dataset = MSRAction3D.read(workspace_generals.MSR_data_dir, split=split, subject_id=subject_id)
        max_frame_num = MSRAction3D.max_frame_num
        frame_size = MSRAction3D.frame_size
        num_classes = MSRAction3D.num_classes

    elif workspace_generals.DATA_FLAG == 'NTU':
        dataset = NTUAction3D.read(workspace_generals.NTU_data_dir, split=split, subject_id=subject_id)
        max_frame_num = NTUAction3D.max_frame_num
        frame_size = NTUAction3D.frame_size
        num_classes = NTUAction3D.num_classes

    else:
        print "There is no data!"


    if split==True:
        # zero pading part
        train_data = zero_padder(dataset.train_data, max_frame_num, frame_size)
        test_data = zero_padder(dataset.test_data, max_frame_num, frame_size)

        train_labels = dataset.train_labels
        train_lens = dataset.train_lens
        test_labels = dataset.test_labels
        test_lens = dataset.test_lens


        # convert list to array
        train_data, train_labels, train_lens = np.asarray(train_data), np.asarray(train_labels), np.asarray(train_lens)
        test_data, test_labels, test_lens = np.asarray(test_data), np.asarray(test_labels), np.asarray(test_lens)

        if one_hot:
            logger.info('one hot')
            train_labels = dense_to_one_hot(dataset.train_labels, num_classes=num_classes)
            test_labels = dense_to_one_hot(dataset.test_labels, num_classes=num_classes)

        tr = (train_data, train_labels, train_lens)
        te = (test_data, test_labels, test_lens)
        return tr,te
    else:
        data, labels, lens = dataset[0],dataset[1],dataset[2]
        data = zero_padder(data, max_frame_num, frame_size)
        data, labels, lens = np.asarray(data), np.asarray(labels), np.asarray(lens)
        if one_hot:
            logger.info('one hot')
            labels = dense_to_one_hot(labels, num_classes=num_classes)
        dt = ( data, labels, lens )
        return dt

get_data(True,1,split=False)