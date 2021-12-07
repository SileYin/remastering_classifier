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
        self.audio_address = []
        self.labels = np.empty((0), dtype=np.float32)
        self.frames = int(chunk_size/512) + 1
        j = 0
        for audiofile in tqdm(glob.glob(os.path.join(path, 'remaster/*'))):
            audio, fs = librosa.load(audiofile, sr=self.fs, mono=False)
            if len(audio.shape) != 2:
                audio = np.vstack((audio, audio))
            available_chunks = int((audio.shape[-1]-chunk_size)/hop_length) + 1
            useless_points = audio.shape[-1] - (available_chunks - 1) * hop_length - chunk_size
            for i in range(available_chunks):
                start_point = useless_points + i * hop_length
                self.audio_address.append((j, start_point))
            self.audio_chunks.append(audio)
            self.labels = np.append(self.labels, np.zeros(available_chunks, dtype=np.float32))
            j += 1
        print(len(self.audio_address))
        print(self.labels.shape[0])
        for audiofile in tqdm(glob.glob(os.path.join(path, 'vinyl/*'))):
            audio, fs = librosa.load(audiofile, sr=self.fs, mono=False)
            if len(audio.shape) != 2:
                audio = np.vstack((audio, audio))
            available_chunks = int((audio.shape[-1]-chunk_size)/hop_length) + 1
            useless_points = audio.shape[-1] - (available_chunks - 1) * hop_length - chunk_size
            for i in range(available_chunks):
                start_point = useless_points + i * hop_length
                self.audio_address.append((j, start_point))
            self.audio_chunks.append(audio)
            self.labels = np.append(self.labels, np.ones(available_chunks, dtype=np.float32))
            j += 1
        print(len(self.audio_address))
        print(self.labels.shape[0])
        


    def __len__(self):
        return len(self.audio_address)


    def __getitem__(self, idx):
        audio_num, start_point = self.audio_address[idx]
        return torch.tensor(self.audio_chunks[audio_num][:,start_point:start_point+self.chunk_size]).float(), torch.tensor(self.labels[idx]).float()


def worker_init(worker_id):
    """
    used with PyTorch DataLoader so that we can grab random bits of files or
    synth random input data on the fly
    Without this you get the same thing every epoch
    """
    # NOTE that this current implementation prevents strict reproducability
    np.random.seed()