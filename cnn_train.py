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
    try:
        acc = correct_results_sum / y_test.shape[0]
    except:
        acc = correct_results_sum / 1
    acc = acc * 100

    return acc

if __name__ == '__main__':
    batch_size = 10
    chunk_size = 131072
    hop_length = 65536
    fs = 44100
    epochs = 10
    status_every = 10
    lr = 1e-4
    datapath = "mydata"


    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        torch.cuda.manual_seed(218)
        print('Using GPU')
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device("cpu")
        print('Using CPU')
        # torch.set_default_tensor_type('torch.FloatTensor')

    model_to_train = cnn_model.CNN_classifier(device=device, mono=False, mask=False)
    if os.path.isfile('savedmodel.tar'):
        state = torch.load('savedmodel.tar')
        model_to_train.load_state_dict(state['state_dict'])
        fs = state['fs']
        chunk_size = state['chunk_size']
    
    traindata = cnn_dataset.cnn_dataset(chunk_size, hop_length, fs=fs, path=datapath + "/Train/")
    valdata = cnn_dataset.cnn_dataset(chunk_size, hop_length, fs=fs, path=datapath + "/Val/")
    dataloader = DataLoader(traindata, batch_size=batch_size, num_workers=4, shuffle=True, worker_init_fn=cnn_dataset.worker_init)  # need worker_init for more variance
    dataloader_val = DataLoader(valdata, batch_size=batch_size, num_workers=4, shuffle=True, worker_init_fn=cnn_dataset.worker_init)


    optimizer = torch.optim.Adam(list(model_to_train.parameters()), lr=lr, weight_decay=0)
    model_to_train.to(device)

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        loss_func = nn.BCELoss()
        model_to_train.train()
        for spec, label in tqdm(dataloader):
            spec, label = spec.to(device), label.to(device)
            pred = model_to_train(spec)
            loss = loss_func(pred.squeeze(), label.squeeze())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += binary_acc(pred.squeeze(), label.squeeze())

        model_to_train.eval()
        for spec, label in dataloader_val:
            spec_val, label_val = spec.to(device), label.to(device)
            pred_val = model_to_train(spec_val)
            pred_tag = torch.round(pred_val.squeeze())
            conf_vector = pred_tag / label_val.squeeze()
            tp += torch.sum(conf_vector==1).item()
            fp += torch.sum(conf_vector==float('inf')).item()
            tn += torch.sum(torch.isnan(conf_vector)).item()
            fn += torch.sum(conf_vector==0).item()
            with torch.no_grad():
                val_loss += loss_func(pred_val.squeeze(), label_val.squeeze())
            val_acc += binary_acc(pred_val.squeeze(), label_val.squeeze())
        print(
            f'Epoch {epoch + 1:03}: | Train Loss: {train_loss / len(dataloader):.5f} \
                | Train Acc = {train_acc / len(dataloader):.3f} \
                | Val Loss = {val_loss / len(dataloader_val):5f}    | Val Acc: {val_acc / len(dataloader_val):.3f}')
        with open("train_loss.dat", "a") as myfile: 
            myfile.write(f"{epoch+1} {train_loss / len(dataloader):.3e}\n")
        with open("train_acc.dat", "a") as myfile: 
            myfile.write(f"{epoch+1} {train_acc / len(dataloader):.2f}\n")
        with open("val_loss.dat", "a") as myfile: 
            myfile.write(f"{epoch+1} {val_loss / len(dataloader_val):.3e}\n")
        with open("val_acc.dat", "a") as myfile: 
            myfile.write(f"{epoch+1} {val_acc / len(dataloader_val):.2f}\n")
        with open("val_conf_mat.dat", "a") as myfile: 
            myfile.write(f"{epoch+1: 3d} {tp} {fp}\n\
                    {fn} {tn}\n")
    state = {'state_dict': model_to_train.state_dict(), 'fs': fs, 'chunk_size': chunk_size}
    torch.save(state, 'savedmodel.tar')