import torch
import random
from torch import nn
import math
import numpy as np
from torch.nn.modules.conv import Conv1d
from torch.nn.modules.pooling import AvgPool1d, MaxPool1d
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import LeaveOneGroupOut


class DeepNeuralNetwork(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(DeepNeuralNetwork, self).__init__()
        closest_pow2 = pow(2,int(math.floor(math.log(input_size ,2))))
        self.bn0 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(in_features=input_size, out_features=closest_pow2)
        self.elu = nn.ELU()
        self.mish = nn.Mish()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        self.feature_extractor = torch.nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.ELU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ELU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 16, 3, padding=1),
            nn.ELU(),
            nn.AvgPool1d(2, 2)
        )
        self.bn1 = nn.BatchNorm1d(closest_pow2)
        self.fc2 = nn.Linear(in_features=closest_pow2, out_features=closest_pow2 // 2) 
        self.bn2 = nn.BatchNorm1d(closest_pow2 // 2)
        self.fc3 = nn.Linear(in_features=closest_pow2 // 2, out_features=output_size)
        self.fc4 = nn.Linear(in_features=closest_pow2 * 2, out_features=output_size)
        
    def forward(self, features):
        activation = self.bn0(features)
        activation = self.fc1(features)
        activation = self.mish(activation)
        activation = self.dropout1(activation)
        activation = self.bn1(activation)
        activation = self.fc2(activation)
        activation = self.mish(activation)
        activation = self.dropout2(activation)
        activation = self.bn2(activation)
        # activation = torch.reshape(activation, (activation.shape[0],1,activation.shape[1]))
        # activation = self.feature_extractor(activation)
        # d = activation.shape
        # activation = torch.reshape(activation, (d[0],d[1]*d[2]))
        activation = self.fc3(activation)
        activation = self.mish(activation)
        output = nn.functional.softmax(activation, dim=1)
        return output


def split_train_test_cv(X, y, test_size = 0.2):
    X_train = np.array([])
    X_test = np.array([])
    y_train = np.array([])
    y_test = np.array([])
    num_items = len(y)
    first_pointer = 0
    train_size = 1 - test_size
    for i in range(1, num_items):
        if y[i] != y[i-1] or i == num_items - 1:
            if i == num_items - 1: i += 1 
            _y = y[first_pointer:i]
            _X = X[first_pointer:i]
            train_index = int(train_size * len(_y))
            X_train = np.append(X_train, _X[:train_index])
            y_train = np.append(y_train, _y[:train_index])
            X_test = np.append(X_test, _X[train_index:])
            y_test = np.append(y_test, _y[train_index:])
            first_pointer = i
    X_train = X_train.reshape(len(y_train), -1)
    y_train = np.array(y_train)
    X_test = X_test.reshape(len(y_test), -1)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test
		

def create_train_loader(train_data, train_labels, device, batch_size=10):
    shuffle_index = np.arange(len(train_labels))
    random.shuffle(shuffle_index)
    train_loader = []
    _train_labels = OneHotEncoder().fit_transform(train_labels.reshape(-1, 1)).toarray()
    labels = []
    tensor_data = []
    tensor_labels = []
    
    for i, data in enumerate(train_data):
        tensor_data.append(np.array(train_data[shuffle_index[i]]).flatten())
        tensor_labels.append(_train_labels[shuffle_index[i]])
        if len(tensor_data) == batch_size:
            train_loader.append(tensor_data)
            labels.append(tensor_labels)
            tensor_labels = []
            tensor_data = []

    if len(tensor_data) != 0:
        train_loader.append(tensor_data)
        labels.append(tensor_labels)
        # print("Train data concatenated due to incompatible batch_size!")
    # labels = OneHotEncoder().fit_transform(np.array(labels).reshape(-1, 1)).toarray()
    return train_loader, labels 
    # return torch.FloatTensor(train_loader).to(device), torch.FloatTensor(labels).to(device)