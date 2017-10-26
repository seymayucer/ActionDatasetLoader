import numpy as np
import bunch


def test_train_splitter_MSR_FLOR(subject_id, data, labels, lens, subject):
    print('Test-Train Cross Subject Splitting')
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
    print ('Train Size', dataset.train_data.shape, dataset.train_labels.shape, dataset.train_lens.shape)
    print ('Test Size', dataset.test_data.shape, dataset.test_labels.shape, dataset.test_lens.shape)
    return dataset


def test_train_splitter_SYM(data,labels,lens,k):
    print('Test-Train k-Cross Splitting',k)

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
    print ('Train Size',dataset.train_data.shape, dataset.train_labels.shape, dataset.train_lens.shape )
    print ('Test Size', dataset.test_data.shape, dataset.test_labels.shape, dataset.test_lens.shape)

    return dataset


def test_train_splitter_NTU(data,labels,lens):
    print('Test-Train Half Subject Splitting')
    perm = np.arange(len(lens))
    np.random.shuffle(perm)
    data = data[perm]
    labels = labels[perm]
    lens = lens[perm]

    TEST_SIZE = len(lens) / 2
    train_data = data[:TEST_SIZE]
    train_labels = labels[:TEST_SIZE]
    train_lens=lens[:TEST_SIZE]
    test_data = data[TEST_SIZE:]
    test_labels = labels[TEST_SIZE:]
    test_lens = lens[TEST_SIZE:]

    dataset = bunch
    dataset.train_data, dataset.train_labels, dataset.train_lens = train_data, train_labels, train_lens
    dataset.test_data, dataset.test_labels, dataset.test_lens = test_data, test_labels, test_lens
    print ('Train Size',dataset.train_data.shape, dataset.train_labels.shape, dataset.train_lens.shape )
    print ('Test Size', dataset.test_data.shape, dataset.test_labels.shape, dataset.test_lens.shape)

    return dataset
