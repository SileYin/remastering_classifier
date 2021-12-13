import torch
import librosa
import cnn_model
import argparse
import numpy as np
import math


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        torch.cuda.manual_seed(218)
        print('Using GPU')
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device("cpu")
        print('Using CPU')
        # torch.set_default_tensor_type('torch.FloatTensor')
    parser = argparse.ArgumentParser(description='Use trained model to predict class of given file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', help='Path to saved model file.')
    parser.add_argument('audio', help='Path to the audiofile you want to classify.')
    args = parser.parse_args()
    model = cnn_model.CNN_classifier(device=device)
    state = torch.load(args.model)
    model.load_state_dict(state['state_dict'])
    model.to(device)
    fs = state['fs']
    chunk_size = state['chunk_size']
    audio, fs = librosa.load(args.audio, sr=fs, mono=False)
    if len(audio.shape) != 2:
        audio = np.vstack((audio, audio))
    available_chunks = int(audio.shape[-1]/chunk_size)
    useless_points = audio.shape[-1] - available_chunks * chunk_size
    audio_chunks = np.zeros((available_chunks, 2, chunk_size))
    for i in range(available_chunks):
        audio_chunks[i, :, :] = audio[:, useless_points+i*chunk_size:useless_points+(i+1)*chunk_size]
    if audio_chunks.shape[0]>10:
        batch_size = 10
    model.eval()
    chunk_pred = np.empty(0, dtype=np.int32)
    for b in range(math.ceil(audio_chunks.shape[0]/batch_size)):
        chunks_to_process = torch.tensor(audio_chunks[b*batch_size:(b+1)*batch_size, :, :]).float().to(device)
        result = model.forward(chunks_to_process)
        chunk_pred = np.append(chunk_pred, torch.round(result).squeeze().cpu().detach().numpy().astype(np.int32))
    if np.count_nonzero(chunk_pred)/chunk_pred.shape[0] > 0.5:
        print(f'Possible medium for file is vinyl. Confidence: {np.count_nonzero(chunk_pred)/chunk_pred.shape[0]: 5f}')
    else:
        print(f'Possible medium for file is CD or other type of digital remaster. \
            Confidence: {1 - np.count_nonzero(chunk_pred)/chunk_pred.shape[0]: 5f}')