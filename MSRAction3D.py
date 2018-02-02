from os import listdir
from os.path import join
import numpy as np
import helpers
from helpers import logger
max_frame_num = 196
frame_size = 60
num_classes = 20
body_part_ids = [[19, 2, 3, 6, 0],
                   [0, 7, 4, 9],
                   [1, 8, 5, 10],
                   [11, 13, 15, 17],
                   [12, 14, 16, 18]]
logger = helpers.logging.getLogger(__name__)
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
        new_act = []
        for frame in action:
            new_frame = helpers.frame_normalizer(frame=frame, frame_size=60)
            new_act.append(new_frame)

        data.append(new_act)
        subjects.append(helpers.full_fname2_str(data_dir, file, 's'))
        # print(action.shape,frame_size)
    data = np.asarray(data)
    labels = np.asarray(labels) -1
    lens = np.asarray(lens)
    data=helpers.dataset_normalizer(data)
    subjects = np.asarray(subjects)
    # data, labels, lens, subjects = get_half(data, labels, lens, subjects)
    print('initial shapes [data label len]: %s %s %s' % (data.shape, labels.shape, lens.shape))
    if split:
        return helpers.test_train_splitter(subject_id, data, labels, lens, subjects)
    else:

        return data,labels,lens

