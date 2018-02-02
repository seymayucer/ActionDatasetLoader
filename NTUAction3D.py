from os import listdir
from os.path import join
import numpy as np
import helpers
from helpers import logger
max_frame_num = 300
frame_size = 75
num_classes = 60

body_part_ids = [[3, 2, 20, 1, 0],
                   [4, 5, 6, 7, 21, 22],
                   [8, 9, 10, 11, 23, 24],
                   [12, 13, 14, 15],
                   [16, 17, 18, 19]]

logger = helpers.logging.getLogger(__name__)
def read(data_dir,subject_id,split=False):
    print('Loading NTU 3D Data, data directory %s' % data_dir)
    data,labels,lens,subjects=[],[],[],[]

    filenames = []
    documents = [join(data_dir, d)
                 for d in sorted(listdir(data_dir))]
    filenames.extend(documents)
    filenames = np.array(filenames)
    action_id = 0

    for filename in sorted(listdir(data_dir)):
        file = join(data_dir, filename)
        subjects.append(int(filename[1:4]))

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
                        a_frame= helpers.frame_normalizer(np.asarray(a_frame),len(a_frame))  #if you need
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
        return helpers.test_train_splitter(data, labels, lens,subjects)

    else:
        return data,labels,lens

