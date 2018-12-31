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
        (idx, g) = x[0]
        print('({}, {})'.format(idx, g))
        # print(x[1][30:100, 0]) # onset strength
        # print(x[2][30:100, 0]) # onset
        # assert False
        
        onset = x[1]    # (T, 1)
        strength = x[2] # (T, 2)

        # plt.figure()
        # plt.plot(range(len(feature[:,0])), feature[:,0])
        # plt.show()

        cnt = 0
        for feature in x[1 : -1]:
            # print(feature.shape)
            cnt += feature.shape[1]
            print(cnt)
            # plt.figure()
            # plt.plot(range(len(feature[:,0])), feature[:,0])
            # plt.show()
        assert False
    i += 100