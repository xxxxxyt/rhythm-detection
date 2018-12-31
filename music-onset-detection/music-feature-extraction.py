import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

group = 3
file_path = 'audio_{}/'.format(group)
n = 6821

sr_0 = 20
sr_1 = 512 * sr_0

def one_hot(T_, indicators):
    ret = np.zeros(T_)
    for i in indicators:
        ret[i] = 1.
    return ret.reshape(-1, 1)

dataset = [] # dataset = [data points] list

for idx in range(3500, n):

    print('idx = {}'.format(idx))
    x = [(group, idx)]
    # x (data point) = [group, idx, features] list
    # feature.shape = (T * 16,) ndarray

    try:
        file_name = '{}{}_{}.mp3'.format(file_path, group, idx)
        y, sr = librosa.load(file_name)
        y = librosa.resample(y, sr, sr_1)
        sr = sr_1

        o_env = librosa.onset.onset_strength(y, sr=sr)
        times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
        onset_bt = librosa.onset.onset_backtrack(onset_frames, o_env)
        rmse = librosa.feature.rmse(S=np.abs(librosa.stft(y=y)))
        onset_bt_rmse = librosa.onset.onset_backtrack(onset_frames, rmse[0])
        onset_subbands = librosa.onset.onset_strength_multi(y=y, sr=sr, channels=[0, 32, 64, 96, 128])

        print(times)
        
        T_ = times.shape[0] # T * 16
        x.append(o_env.reshape(-1, 1)) # onset strength
        x.append(one_hot(T_, onset_frames)) # onset_detect
        x.append(one_hot(T_, onset_bt)) # onset_backtrack
        x.append(one_hot(T_, onset_bt_rmse)) # onset_backtrack
        x.append(rmse) # rmse
        x.append(onset_subbands) # onset_strength_multi

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        S = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        melspectrogram = librosa.feature.melspectrogram(S=(S ** 2))
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(melspectrogram))
        mag, phase = librosa.magphase(librosa.stft(y=y))
        centroid = librosa.feature.spectral_centroid(S=mag)
        bandwidth = librosa.feature.spectral_bandwidth(S=mag)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr, fmin=100)
        flatness = librosa.feature.spectral_flatness(S=mag)
        rolloff = librosa.feature.spectral_rolloff(S=mag, sr=sr)
        p0 = librosa.feature.poly_features(S=S, order=0)
        p1 = librosa.feature.poly_features(S=S, order=1)
        p2 = librosa.feature.poly_features(S=S, order=2)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

        x.append(chroma_stft)
        x.append(chroma)
        x.append(chroma_cq)
        x.append(chroma_cens)
        x.append(melspectrogram)
        x.append(mfcc)
        x.append(centroid)
        x.append(bandwidth)
        x.append(contrast)
        x.append(flatness)
        x.append(rolloff)
        x += [p0, p1, p2]
        x.append(tonnetz)
        x.append(zero_crossing_rate)

        # T = int(math.ceil(1. * T_ / sr_0 * 2))
        for i in range(1, len(x)):
            if x[i].shape[0] != T_:
                x[i] = x[i].transpose()
            # split = np.array_split(x[i], T)
            # for j in range(len(split)):
            #     split[j] = np.mean(split[j], axis=0)
            # x[i] = np.stack(split)

            # print(x[i].shape)

            # plt.figure()
            # plt.plot(range(T_), x[i][:,0])
            # plt.show()

        # assert False

    except Exception as e:
        print(e.message)

    else:
        dataset.append(x)

    if (idx + 1) % 100 == 0:
        f = open('dataset_{}_{}.pkl'.format(group, idx + 1), 'wb')
        pickle.dump(dataset, f)
        f.close()
        dataset = []
