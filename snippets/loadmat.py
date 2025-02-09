from scipy import io
import matplotlib.pyplot as plt
import numpy as np

feature = io.loadmat('emotiv_7sub_5class.mat')
data = feature['emotiv_7sub_5class']
print(data.shape)

train_data = data[:10, 0:13]
train_label = data[:10, 14:16]
print(train_data.shape)
print(train_label.shape)

concatenated = np.concatenate((train_data, train_label), axis=1)

print(concatenated)
