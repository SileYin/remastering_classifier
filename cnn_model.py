import numpy as np
import torch
import torch.nn as nn
import torchaudio


class CNN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CNN_layer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv(x)


class CNN_classifier(nn.Module):
    def __init__(self, debug_mode=False, device='cpu'):
        super(CNN_classifier, self).__init__()
        self.debug_mode = debug_mode
        self.device = device
        self.cnn_layers = nn.Sequential(CNN_layer(2, 4, 3), CNN_layer(4, 16, 3), CNN_layer(16, 64, 3),
                                        CNN_layer(64, 32, 3), CNN_layer(32, 16, 3))
        self.cnn_to_fc = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=(16, 8), stride=1,
                      padding='valid'),
            nn.Sigmoid()
        )
        self.fc_layers = nn.Sequential(nn.Linear(128, 32), nn.Linear(32, 8), nn.Linear(8, 1))

    def forward(self, x):
        l = torch.abs(torch.stft(x[:, 0, :], n_fft=1024, hop_length=512, window=torch.hann_window(1024),
                                             win_length=1024, normalized=False, onesided=True, return_complex=True))
        r = torch.abs(torch.stft(x[:, 1, :], n_fft=1024, hop_length=512, window=torch.hann_window(1024),
                                             win_length=1024, normalized=False, onesided=True, return_complex=True))
        z = torch.cat((l.unsqueeze(1), r.unsqueeze(1)), 1)
        if self.debug_mode:
            print(z.size())
        for i in range(5):
            z = self.cnn_layers[i](z)
            if self.debug_mode:
                print(z.size())
        z = self.cnn_to_fc(z).squeeze()
        if self.debug_mode:
            print(z.size())
        for i in range(3):
            z = self.fc_layers[i](z)
            z = nn.Sigmoid()(z)
            if self.debug_mode:
                print(z.size())
        return z




if __name__ == '__main__':
    model = CNN_classifier(debug_mode=True)
    input = torch.randn(10, 2, 131072)
    out = model.forward(input)
    print(out.size())