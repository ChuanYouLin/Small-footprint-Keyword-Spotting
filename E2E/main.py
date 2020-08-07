from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
# import matplotlib.pyplot as plt
# import scipy.io.wavfile as wav
import sys
import os
import re
import hashlib
import math
import random
import torch
import torch.nn as nn
import numpy as np
# torch.autograd.set_detect_anomaly(True)
from tdnn import TDNN
# from plot_confusion_matrix import plot_confusion_matrix

device = torch.device('cuda')
keyword = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

def load_data(input_path):
    training_X = np.load(os.path.join(input_path, "train_X.npy"))
    training_Y_spk = np.load(os.path.join(input_path, "train_Y_spk.npy"))
    training_Y_text = np.load(os.path.join(input_path, "train_Y_text.npy"))
    validation_X = np.load(os.path.join(input_path, "validation_X.npy"))
    validation_Y_spk = np.load(os.path.join(input_path, "validation_Y_spk.npy"))
    validation_Y_text = np.load(os.path.join(input_path, "validation_Y_text.npy"))
    return training_X, training_Y_spk, training_Y_text, validation_X, validation_Y_spk, validation_Y_text

class AudioDataset(Dataset):
    def __init__(self, x_train, y_train_spk, y_train_text, transform=None):
        self.x_train = x_train
        self.y_train_spk = y_train_spk
        self.y_train_text = y_train_text
    def __len__(self):
        return self.x_train.shape[0]
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train_spk[idx], self.y_train_text[idx]

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.tdnn = TDNN(input_dim=32, output_dim=64, context_size=3, dilation=1)
        self.tdnn2 = TDNN(input_dim=64, output_dim=128, context_size=3, dilation=1)
        self.classifier = nn.Sequential(
            # nn.Linear(31*32, 31*32),
            # nn.BatchNorm1d(31*32),
            # nn.ReLU(inplace=True),
            nn.Linear(38*128, 1776),
            nn.BatchNorm1d(1776),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Softmax(dim=1)
    def forward(self, x):
        out = self.tdnn(x)
        out = self.tdnn2(out)
        out = self.classifier(out.reshape(out.size()[0],out.size()[1]*out.size()[2]))
        # out = self.output(out)
        return out

class classifier2(nn.Module):
    def __init__(self):
        super(classifier2, self).__init__()
        self.tdnn_0 = TDNN(input_dim=40, output_dim=32, context_size=3, dilation=1, stride=3)
        # self.tdnn_1 = TDNN(input_dim=32, output_dim=32, context_size=3, dilation=1)
        # self.tdnn_2 = TDNN(input_dim=32, output_dim=32, context_size=3, dilation=1)
        # self.test = nn.Sequential(
        #     nn.AvgPool2d((29, 1),stride=1),
        #     nn.Linear(32, 11),
        # )
        # self.output = nn.Softmax(dim=1)
    def forward(self, x):
        out = self.tdnn_0(x)
        # out = self.tdnn_1(out)
        # code = self.tdnn_2(out)
        # out = self.test(code)
        # out = out.squeeze(1)
        # out = self.output(out)
        return out

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.tdnn_1 = TDNN(input_dim=32, output_dim=32, context_size=3, dilation=1)
        self.tdnn_2 = TDNN(input_dim=32, output_dim=32, context_size=3, dilation=1)
        self.test = nn.Sequential(
            nn.AvgPool2d((38, 1),stride=1),
            nn.Linear(32, 11),
        )
        self.output = nn.Softmax(dim=1)
    def forward(self, x):
        code = self.tdnn_1(x)
        code = self.tdnn_2(code)
        out = self.test(code)
        out = out.squeeze(1)
        # out = self.output(out)
        return out

def main():
    setup_seed(18)
    # print("seed = {}".format(sys.argv[1]))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # f = open("{}.txt".format(sys.argv[1]), "w")
    
    training_X, training_Y_spk, training_Y_text, validation_X, validation_Y_spk, validation_Y_text = load_data(sys.argv[1])
    train_dataset = AudioDataset(training_X, training_Y_spk, training_Y_text)
    train_loader = DataLoader(dataset = train_dataset, batch_size = 32, shuffle = True)
    validation_dataset = AudioDataset(validation_X, validation_Y_spk, validation_Y_text)
    validation_loader = DataLoader(dataset = validation_dataset, batch_size = 32, shuffle = False)

    C = classifier()
    C.apply(init_weights)
    C.to(device)
    learning_rate_C = 0.001
    optimizer_C = Adam(C.parameters(), lr=learning_rate_C)
    a = [1.] * 1775
    a.append(0.5)
    CrossEntropyLossWeight = torch.tensor(a).to(device)
    loss_fn_C = nn.CrossEntropyLoss(CrossEntropyLossWeight)
    
    C2 = classifier2()
    C2.apply(init_weights)
    C2.to(device)
    learning_rate_C2 = 0.001
    optimizer_C2 = Adam(C2.parameters(), lr=learning_rate_C2)
    a = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    CrossEntropyLossWeight = torch.tensor(a).to(device)
    loss_fn_C2 = nn.CrossEntropyLoss(weight = CrossEntropyLossWeight)
    scheduler_C2 = torch.optim.lr_scheduler.StepLR(optimizer_C2, step_size=100, gamma=0.1)

    D = decoder()
    D.apply(init_weights)
    D.to(device)
    learning_rate_D = 0.001
    optimizer_D = Adam(D.parameters(), lr=learning_rate_D)
    a = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    CrossEntropyLossWeight = torch.tensor(a).to(device)
    loss_fn_D = nn.CrossEntropyLoss(weight = CrossEntropyLossWeight)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=100, gamma=0.1)

    print('# encoder parameters:', sum(param.numel() for param in C2.parameters()))
    print('# decoder parameters:', sum(param.numel() for param in D.parameters()))
    # exit()

    train_his_text = []
    train_his_spk = []
    val_his = []
    adversarial_his = []
    best_val = 1.
    for epoch in range(300):
        train_loss = []
        train_acc = []
        train_loss2 = []
        train_acc2 = []
        train_acc3 = []

        D.train()
        C2.train()
        for _, (mfccs, target_spk, target_text) in enumerate(train_loader):
            mfccs_cuda = mfccs.to(device).float()
            target_cuda_spk = target_spk.to(device)
            target_cuda_text = target_text.to(device)

            optimizer_D.zero_grad()
            optimizer_C2.zero_grad()
            output = C2(mfccs_cuda)
            output2 = D(output)
            
            loss_text = loss_fn_D(output2, target_cuda_text)
            loss_text.backward()
            optimizer_D.step()
            optimizer_C2.step()

            predict = torch.max(output2, 1)[1]
            acc = np.mean((target_cuda_text == predict).cpu().numpy())

            train_acc.append(acc)
            train_loss.append(loss_text.item())

            output = C2(mfccs_cuda)
            output3 = C(output)

            predict2 = torch.max(output3, 1)[1]
            acc2 = np.mean((target_cuda_spk == predict2).cpu().numpy())
            
            train_acc3.append(acc2)

        
        C.train()
        C2.train()
        for _, (mfccs, target_spk, target_text) in enumerate(train_loader):
            mfccs_cuda = mfccs.to(device).float()
            target_cuda_spk = target_spk.to(device)
            target_cuda_text = target_text.to(device)

            optimizer_C.zero_grad()
            optimizer_C2.zero_grad()
            output = C2(mfccs_cuda)
            output2 = C(output)
            
            # loss_text = loss_fn_C2(output, target_cuda_text)
            # loss_text.backward(retain_graph = True)
            # optimizer_C2.step()
            
            loss_spk = loss_fn_C(output2, target_cuda_spk)
            loss_spk.backward(retain_graph = True)

            optimizer_C2.zero_grad()
            loss_spk2 = loss_fn_C(output2, target_cuda_spk) * -1. * 0.03
            loss_spk2.backward()
            optimizer_C2.step()
            optimizer_C.step()

            # predict = torch.max(output, 1)[1]
            # acc = np.mean((target_cuda_text == predict).cpu().numpy())

            # train_acc.append(acc)
            # train_loss.append(loss_text.item())

            predict2 = torch.max(output2, 1)[1]
            acc2 = np.mean((target_cuda_spk == predict2).cpu().numpy())

            train_acc2.append(acc2)
            train_loss2.append(loss_spk.item())

        val_loss = []
        val_acc = []
        val_acc2 = []
        C2.eval()
        D.eval()
        for _, (mfccs, target_spk, target_text) in enumerate(validation_loader):
            mfccs_cuda = mfccs.to(device).float()
            target_cuda_text = target_text.to(device)
            target_cuda_spk = target_spk.to(device)

            output = C2(mfccs_cuda)
            output2 = C(output)
            output3 = D(output)
            loss = loss_fn_D(output3, target_cuda_text)

            predict = torch.max(output3, 1)[1]
            acc = np.mean((target_cuda_text == predict).cpu().numpy())
            predict2 = torch.max(output2, 1)[1]
            acc2 = np.mean((target_cuda_spk == predict2).cpu().numpy())

            val_acc.append(acc)
            val_acc2.append(acc2)
            val_loss.append(loss.item())
        if 1 - np.mean(val_acc) < best_val:
            best_val = 1 - np.mean(val_acc)
            # torch.save(C.state_dict(), "feature_disentangle_ClassifierSpk8_seed18_repro.pkl")
            torch.save(C2.state_dict(), "encoder.pkl")
            torch.save(D.state_dict(), "decoder.pkl")
        train_his_text.append(1 - np.mean(train_acc))
        train_his_spk.append(1 - np.mean(train_acc2))
        val_his.append(1 - np.mean(val_acc))
        adversarial_his.append(1 - np.mean(train_acc3))
        scheduler_C2.step()
        scheduler_D.step()
        print("Epoch: {}, Train_loss: {:.4f}, Train_ER: {:.4f}, Train_ER_spk: {:.4f}, adversarial: {:.4f}, \
Validation_ER: {:.4f}, Validation_ER_spk: {:.4f}, Best_val: {:.4f}".format(
                    epoch + 1, np.mean(train_loss), train_his_text[-1], train_his_spk[-1], adversarial_his[-1], 
                    val_his[-1], 1 - np.mean(val_acc2), min(val_his)))

    print(min(val_his))
    print(val_his.index(min(val_his)))

if __name__ == "__main__": main()