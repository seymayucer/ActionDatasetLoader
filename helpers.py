import numpy as np
#some helpers

def full_fname2_str(data_dir,fname,sep_char):
    fnametostr = ''.join(fname).replace(data_dir, '')
    ind = int(fnametostr.index(sep_char))
    label = int(fnametostr[ind + 1:ind + 3])
    return label

def frame_normalizer(frame):
    assert frame.shape[0]==45
    frame=frame.reshape(15,3)
    spine_mid=frame[1]
    new_frame=[]
    j=0
    for joint in frame:
        new_frame.append(joint-spine_mid)
        j+=1
    new_frame=np.asarray(new_frame)
    return(list(new_frame.flatten()))


def dataset_normalizer(data):
    print('NORMALIZER')
    min_list=[]
    max_list=[]
    for act in data:
        max_list.append(np.amax(act))
        min_list.append(np.amin(act))

    print('min num:',min(min_list),'max num',max(max_list))
    min_num=min(min_list)
    max_num=max(max_list)
    new_data=[]
    i =0
    for act in data:
        act=(act+abs(min_num))/(max_num+abs(min_num))
        new_data.append(act)
        i += 0

    return np.asarray(new_data)

def normalizer(data):
    new_data = []
    for act in data:
        new_act = []
        for frame in act:

            new_frame = frame_normalizer(np.asarray(frame))
            new_act.append(new_frame)

        new_data.append(new_act)
    data = dataset_normalizer(new_data)
    return data