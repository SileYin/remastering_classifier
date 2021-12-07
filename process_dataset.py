from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
import librosa
import os
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import pickle
import math


def block_audio(x, blockSize, hopSize):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize) + 1
    xb = np.zeros([numBlocks, blockSize])
    x = np.concatenate((np.zeros(hopSize), x, np.zeros(blockSize)),axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return xb


def sub_power_ratio(y, n_fft=2048, hop_size=512):
    spec = librosa.stft(y, n_fft=n_fft, hop_length=hop_size)
    return (np.sum(spec[:6, :]*spec.conj()[:6, :], axis=0)/(np.sum(spec[:, :]*spec.conj()[:, :], axis=0)+1e-8)).real


def stereo_width(y, blockSize=2048, hopSize=512):
    mid = 0.5*(y[0, :]+y[1, :])
    side = 0.5*(y[0, :] - y[1, :])
    mb = block_audio(mid, blockSize, hopSize)
    sb = block_audio(side, blockSize, hopSize)
    return np.mean(sb**2, axis=1)/(np.mean(mb**2, axis=1)+1e-8)


def high_power_ratio(y, n_fft=2048, hop_size=512):
    spec = librosa.stft(y, n_fft=n_fft, hop_length=hop_size)
    return (np.sum(spec[192:, :] * spec.conj()[192:, :], axis=0) / (
                np.sum(spec[:, :] * spec.conj()[:, :], axis=0) + 1e-8)).real


def get_feature(path, avg_over=100):
    filenames = os.listdir(path)
    # mfcc_mean = np.empty((13, len(filenames)))
    features = Parallel(n_jobs=6, backend='multiprocessing')(delayed(get_mfcc_mean)(os.path.join(path, filename)) for filename in tqdm(filenames))
    # for i, filename in enumerate(filenames):
    #    y, fs = librosa.load(os.path.join(path, filename), sr=None)
    #    mfcc_mean[:, i] = get_mfcc_mean(y, fs)
    #    print(f'{i+1}/{len(filenames)} file(s) loaded in {path}.')
    f_train, f_test, _ph0, _ph1 = train_test_split(features, np.zeros(len(features)), test_size=0.33,
                                                   random_state=33)
    f_train = np.concatenate(f_train, axis=1)
    f_train = f_train[:, :int(f_train.shape[1]/avg_over)*avg_over]
    f_train = f_train.reshape(-1, avg_over).mean(1).reshape(f_train.shape[0], -1).T
    f_test = np.concatenate(f_test, axis=1)
    f_test = f_test[:, :int(f_test.shape[1]/avg_over)*avg_over]
    f_test = f_test.reshape(-1, avg_over).mean(1).reshape(f_test.shape[0], -1).T
    return f_train, f_test


def get_mfcc_mean(file_path):
    y, fs = librosa.load(file_path, sr=None, mono=False)
    if y.shape[0]!=2:
        y = np.vstack((y, y))
    mfcc = librosa.feature.mfcc(0.5*(y[0, :]+y[1, :]), sr=fs, n_mfcc=13)
    spr = sub_power_ratio(0.5*(y[0, :]+y[1, :]))
    sw = stereo_width(y)
    mel = librosa.feature.melspectrogram(0.5*(y[0, :]+y[1, :]), sr=fs)
    # zcr = librosa.feature.zero_crossing_rate(y)
    # bw = librosa.feature.spectral_bandwidth(y, sr=fs)
    return np.vstack((mfcc, spr, sw, mel))


if __name__ == '__main__':
    average_over = 256
    V_train, V_test = get_feature('vinyl', average_over)
    R_train, R_test = get_feature('remaster', average_over)
    X_train = np.concatenate((V_train, R_train), axis=0)
    y_train = np.concatenate((np.zeros(V_train.shape[0]), np.ones(R_train.shape[0])), axis=0)
    shuffler = np.random.permutation(X_train.shape[0])
    X_train = X_train[shuffler, :]
    y_train = y_train[shuffler]
    X_test = np.concatenate((V_test, R_test), axis=0)
    y_test = np.concatenate((np.zeros(V_train.shape[0]), np.ones(R_train.shape[0])), axis=0)
    shuffler = np.random.permutation(X_test.shape[0])
    X_test = X_test[shuffler, :]
    y_test = y_test[shuffler]

    with open('X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open('y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    '''
    clf = SVC(kernel='rbf', tol=1e-3, C=10, gamma=1e-5)
    clf.fit(X_train, y_train)
    print(f"Training score of classifier: {clf.score(X_train, y_train)}")
    print(f"Test score of classifier: {clf.score(X_test, y_test)}")
    '''
