import numpy as np
import torch
import os
import glob
from torch.utils.data import Dataset
import librosa
from tqdm import tqdm


class cnn_dataset(Dataset):

    def __init__(self, chunk_size, hop_length, fs=44100, path=""):
        super(cnn_dataset, self).__init__()
        self.fs = fs
        self.path = path
        self.chunk_size = chunk_size
        self.hop_length = hop_length
        self.audio_chunks = []
        self.labels = []
        self.frames = int(chunk_size/512) + 1
        for audiofile in tqdm(glob.glob(os.path.join(path, 'vinyl/*'))):
            audio, fs = librosa.load(audiofile, sr=self.fs, mono=False)
            if len(audio.shape) != 2:
                audio = np.vstack((audio, audio))
            available_chunks = int((audio.shape[-1]-chunk_size)/hop_length) + 1
            useless_points = audio.shape[-1] - (available_chunks - 1) * hop_length - chunk_size
            for i in range(available_chunks):
                start_point = useless_points + i * hop_length
                tmp = audio[:, start_point:start_point+chunk_size]
                self.audio_chunks.append(torch.tensor(tmp))
                self.labels.append(torch.tensor(0))
        for audiofile in tqdm(glob.glob(os.path.join(path, 'remaster/*'))):
            audio, fs = librosa.load(audiofile, sr=self.fs, mono=False)
            if len(audio.shape) != 2:
                audio = np.vstack((audio, audio))
            available_chunks = int((audio.shape[0]-chunk_size)/hop_length) + 1
            useless_points = audio.shape[-1] - (available_chunks - 1) * hop_length - chunk_size
            for i in range(available_chunks):
                start_point = useless_points + i * hop_length
                tmp = audio[:, start_point:start_point + chunk_size]
                self.audio_chunks.append(torch.tensor(tmp))
                self.labels.append(torch.tensor(1))
        print(len(self.audio_chunks))


    def __len__(self):
        return len(self.audio_chunks)


    def __getitem__(self, idx):
        return self.audio_chunks[idx], self.labels[idx]


def worker_init(worker_id):
    """
    used with PyTorch DataLoader so that we can grab random bits of files or
    synth random input data on the fly
    Without this you get the same thing every epoch
    """
    # NOTE that this current implementation prevents strict reproducability
    np.random.seed()