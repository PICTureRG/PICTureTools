import scipy.io
import numpy as np

news_dict = scipy.io.loadmat('documents.mat')
news = np.transpose(news_dict['fullDocuments'])
np.savetxt('news.txt',news, fmt='%d')

