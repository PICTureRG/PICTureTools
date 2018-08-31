import scipy.io
import numpy as np

cal16_split_dict = scipy.io.loadmat('caltech101_silhouettes_16_split1.mat')
cal28_split_dict = scipy.io.loadmat('caltech101_silhouettes_28_split1.mat')

cal16_train = cal16_split_dict['train_data']
cal28_train = cal28_split_dict['train_data']

cal16_valid = cal16_split_dict['val_data']
cal28_valid = cal28_split_dict['val_data']

cal16_test = cal16_split_dict['test_data']
cal28_test = cal28_split_dict['test_data']

np.savetxt('cal16_train.txt', cal16_train, fmt='%d')
np.savetxt('cal16_valid.txt', cal16_valid, fmt='%d')
np.savetxt('cal16_test.txt', cal16_test, fmt='%d')
np.savetxt('cal28_train.txt', cal28_train, fmt='%d')
np.savetxt('cal28_valid.txt', cal28_valid, fmt='%d')
np.savetxt('cal28_test.txt', cal28_test, fmt='%d')
