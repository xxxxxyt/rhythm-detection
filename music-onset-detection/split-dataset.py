import pickle
import matplotlib.pyplot as plt
import numpy as np

group = 3
n = 6821

dir_name = 'audio_{}'.format(group)

i = 100
while i < n:
    f = open('{}/dataset_{}_{}.pkl'.format(dir_name, group, i), 'rb')
    dataset = pickle.load(f)
    for x in dataset:
        (g, idx) = x[0]
        print('({}, {})'.format(g, idx))
        f_label = open('{}_split/label_{}_{}.pkl'.format(dir_name, g, idx), 'wb')
        label = np.concatenate(x[1:4], axis=1)
        # print(label.shape)
        # assert False
        pickle.dump(label, f_label)
        f_label.close()

        # f_feature = open('{}_split/feature_{}_{}.pkl'.format(dir_name, g, idx), 'wb')
        # feature = np.concatenate(x[1:-1], axis=1)
        # # print(feature.shape)
        # pickle.dump(feature, f_feature)
        # f_feature.close()
        
    i += 100