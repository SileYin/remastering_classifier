import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import time
import os
import cnn_dataset
import cnn_model
from tqdm import tqdm


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

if __name__ == '__main__':
    batch_size = 20
    chunk_size = 131072
    hop_length = 65536
    fs = 44100
    epochs = 50
    status_every = 10
    lr = 1e-5
    datapath = "mydata"


    traindata = cnn_dataset.cnn_dataset(chunk_size, hop_length, fs=fs, path=datapath + "/Train/")
    valdata = cnn_dataset.cnn_dataset(chunk_size, hop_length, fs=fs, path=datapath + "/Val/")
    dataloader = DataLoader(traindata, batch_size=batch_size, num_workers=2, shuffle=True)  # need worker_init for more variance
    dataloader_val = DataLoader(valdata, batch_size=batch_size, num_workers=2, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(218)
        print('Using GPU')
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device("cpu")
        print('Using CPU')
        # torch.set_default_tensor_type('torch.FloatTensor')

    model_to_train = cnn_model.CNN_classifier()
    if os.path.isfile('savedmodel.tar'):
        model_to_train.load_state_dict(torch.load('savedmodel.tar'))
    optimizer = torch.optim.Adam(list(model_to_train.parameters()), lr=lr, weight_decay=0)
    model_to_train.to(device)

    for epoch in range(epochs):
        train_loss = 0
        val_acc = 0
        loss_func = nn.BCELoss()
        model_to_train.train()
        for spec, label in tqdm(dataloader):
            spec, label = spec.to(device), label.to(device)
            pred = model_to_train(spec)
            loss = loss_func(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model_to_train.eval()
        for spec, label in dataloader_val:
            spec, label = spec.to(device), label.to(device)
            pred = model_to_train(spec)
            val_acc += binary_acc(pred, label)
        print(
            f'Epoch {epoch + 1:03}: | Loss: {train_loss / len(dataloader):.5f} | Acc: {val_acc / len(dataloader_val):.3f}')

    torch.save(model_to_train.state_dict(), 'savedmodel.tar')