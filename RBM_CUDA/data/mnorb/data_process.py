import scipy.io
import numpy as np

norb_dict = scipy.io.loadmat('trainingData.mat')
norb = norb_dict['inputs']
np.savetxt('micro-norb.txt',norb, fmt='%d')

