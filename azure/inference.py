import os
import glob
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable

from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import sys

import wandb

wandb.init(project='cycleai', entity='gonvas_')

config = wandb.config

config.learning_rate = 1e-4


class NeuralNetwork(nn.Module):

    def __init__(self, input_in=66):
        super().__init__()

        self.fc1 = nn.Linear(input_in, input_in*2)
        self.b1 = nn.BatchNorm1d(input_in*2)
        self.fc2 = nn.Linear(input_in*2, input_in*4)
        self.b2 = nn.BatchNorm1d(input_in*4)
        self.fc3 = nn.Linear(input_in*4, input_in)
        self.b3 = nn.BatchNorm1d(input_in)
        self.fc4 = nn.Linear(input_in, 11)


    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = self.b1(x)
        x = F.relu(self.fc2(x))
        x = self.b2(x)
        x = F.relu(self.fc3(x))
        x = self.b3(x)
        x = F.relu(self.fc4(x))

        return F.log_softmax(x)


def train_validate(network, epochs, x_train, y_train, save_Model=True, batch_size=256):
    total_acc = 0
    kfold = KFold(n_splits=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    
    for fold, (train_index, test_index) in enumerate(kfold.split(x_train, y_train)):
        ### Dividing data into folds
        x_train_fold = x_train[train_index]
        x_test_fold = x_train[test_index]
        y_train_fold = y_train[train_index]
        y_test_fold = y_train[test_index]
        
        print(x_train_fold.shape)
        print(y_train_fold.shape)

        train = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_fold), y_train_fold)
        test = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_fold), y_test_fold)
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle = False)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle = False)

        for epoch in range(epochs):
            print('\nEpoch {} / {} \nFold number {} / {}'.format(epoch + 1, epochs, fold + 1 , kfold.get_n_splits()))
            correct = 0
            network.train()
            for batch_index, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                out = network(x_batch)
                loss = criterion(out, torch.max(y_batch, 1)[1])
                loss.backward()
                optimizer.step()
                wandb.log({"loss": loss})
                pred = torch.max(out.data, dim=1)[1]
                correct += (pred == torch.max(y_batch, 1)[1]).sum()
                wandb.log({"correct": correct})

                if epoch % 1000 == 0 and epoch != 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, os.getcwd() + f'/nn_weights_{epoch}.pth')

                    wandb.save(os.getcwd() + f'/nn_weights_{epoch}.pth')

                if (batch_index + 1) % 3 == 0:
                    print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                        (batch_index + 1)*len(x_batch), len(train_loader.dataset),
                        100.*batch_index / len(train_loader), loss.data, float(correct*100) / float(batch_size*(batch_index+1))))
                    wandb.log({'batch_acc': float(correct*100) / float(batch_size*(batch_index+1))})
        

        total_acc += float(correct*100) / float(batch_size*(batch_index+1))
        wandb.log({"total_acc": total_acc})



        test_correct = 0
        for batch_index, (x_batch, y_batch) in enumerate(test_loader):
            out = network(x_batch)
            pred = torch.max(out.data, dim=1)[1]
            test_correct += (pred == torch.max(y_batch, 1)[1]).sum()
            wandb.log({"test_correct": test_correct})
            wandb.log({'test_per_correct': float(test_correct*100) / float(batch_size*(batch_index+1))})

    
    total_acc = (total_acc / kfold.get_n_splits())
    wandb.log({"total_acc_val": total_acc})
    print('\n\nTotal accuracy cross validation: {:.3f}%'.format(total_acc))


def run_inference(pix_hist_folder, skill_txt):
    pix_hist_vals = {}

    for filename in glob.glob(os.path.join(pix_hist_folder, '*.txt')):
       with open(os.path.join(os.getcwd(), filename), 'r') as f: 
          pix_vals = f.readlines()
          pix_vals = [int(pix_val.split(',')[-1].split('\n')[0]) for pix_val in pix_vals]
          pix_hist_vals[filename.split('/')[-1].split('.')[0].split('_pixels')[0]] = pix_vals

    f = open(skill_txt, "r") 
    skill_output = f.readlines()

    final_vals = []

    for out in skill_output:
        name = out.split('trueskill')[0].strip().split('.')[0]
        skill_vals = float(out.split('score')[-1][2:-1])
        try:
            final_vals.append([pix_hist_vals[name], skill_vals])
        except:
            print('One image found in the pixels folder not in the skill vals')
            continue

    df = pd.DataFrame.from_dict(final_vals)
    df.columns = ['x', 'y']

    new_pd = df.x.apply(pd.Series).astype(int)

    min_max_scaler = preprocessing.MinMaxScaler()
    new_pd = min_max_scaler.fit_transform(new_pd)

    X_train, X_test, y_train, y_test = train_test_split(new_pd, df.y.astype(int), test_size = 0.20)


    svclassifier = svm.SVC(kernel='linear', verbose=1)
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))


    net = NeuralNetwork()

    criterion = nn.CrossEntropyLoss()
    epochs = 1_000


    tvy = torch.tensor(df.y.astype(int).values)
    hot_encoded = torch.zeros(len(tvy), tvy.max()+1).scatter_(1, tvy.unsqueeze(1), 1.)

    #print(torch.Tensor(np.array(X_train)).shape)
    #print(hot_encoded.shape)
    #train = data_utils.TensorDataset(torch.Tensor(np.array(X_train)), hot_encoded)
    #train_loader = data_utils.DataLoader(train, batch_size = 10, shuffle = True)
    wandb.watch(net)

    train_validate(net, epochs, new_pd, hot_encoded)




if __name__ == "__main__":
    #pix_hist_folder = os.getcwd() +  '/images_pixels/'
    #skill_txt =  os.getcwd() +  '/sorted_imgs.txt'

    if(len(sys.argv) != 3):
        print('Wrong number of args, call with folder of pictures and txt loc')
    
    else:
        run_inference(sys.argv[1], sys.argv[2])