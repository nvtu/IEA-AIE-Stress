# Add additional library
from genericpath import sameopenfile
import sys, os



data_lib = os.path.abspath('../data')
eda_sp_lib = os.path.abspath('../signal_processing')
model_lib = os.path.abspath('../models')
if data_lib not in sys.path:
    sys.path.append(data_lib)
if eda_sp_lib not in sys.path:
    sys.path.append(eda_sp_lib)
if model_lib not in sys.path:
    sys.path.append(model_lib)


import numpy as np
from data_utils import *
from autoencoder import *
from eda_signal_processing import *
from tqdm import tqdm
import os.path as osp
import pickle
import warnings
import torch
from sklearn.model_selection import LeaveOneGroupOut
from torch import nn
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')


EDA_SAMPLING_RATE = 4
# DATASET_NAME = 'AffectiveROAD'
DATASET_NAME = 'WESAD'
DEVICE = 'wrist'
# DEVICE = 'right'
SIGNAL_NAME = 'EDA_auto'
# NORMALIZER = '_nonorm'
NORMALIZER = ''

WINDOW_SIZE = 60
WINDOW_SHIFT = 0.25

dp_manager = get_data_path_manager()
swt_denoiser = SWT_Threshold_Denoiser()
eda_processor = EDA_Signal_Processor()


def prepare_automatic(gsr_signal, target_user_id, k=32, epochs=100, batch_size=10):
    gsrdata = np.array(gsr_signal)

    #################################################################################
    ############################ Train the Autoencoder ##############################

    # set the input shape to model
    input_shape = gsrdata.shape[1]

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AutoEncoder(input_shape=input_shape, latent_size=k).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    # create tensor data
    train_loader = create_train_loader(gsrdata, batch_size)

    # Training the network
    for epoch in range(epochs):
        loss = 0
        for batch_features in train_loader:
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            # compute reconstructions
            outputs,_ = model(batch_features)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
            
        # compute the epoch training loss
        loss = loss / len(train_loader)
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
        
    # Save the network
    model_path = osp.join(get_model_path(DATASET_NAME), '{}_{}_{}_checkpoint.t7'.format(DATASET_NAME, DEVICE, target_user_id))
    torch.save(model, model_path)
    

def process_automatic(gsr_signal, target_user_id):
	#################################################################################
	############################ Feature Extraction Part ############################
	
    # Load the network
    model_path = osp.join(get_model_path(DATASET_NAME), f'{DATASET_NAME}_{DEVICE}_{target_user_id}_checkpoint.t7')
    model = torch.load(model_path)

    # Extract the features
    gsr_signal = np.reshape(gsr_signal, (1, gsr_signal.shape[0]))
    train_outputs, latent_variable = model(torch.FloatTensor(gsr_signal))
    return latent_variable.detach().numpy()[0];


if __name__ == '__main__':
    eda_stats_features = []

    dataset_path = osp.join(dp_manager.dataset_path, f'{DATASET_NAME}_{DEVICE}_dataset.pkl')
    dataset = pickle.load(open(dataset_path, 'rb'))

    preprocessed_eda = []
    groups = []
    labels = []
    preprocessed_eda_path = osp.join(get_dataset_stats_features_path(dp_manager, DATASET_NAME), f'{DATASET_NAME}_{DEVICE}_preprocessed_eda_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
    groups_path = osp.join(get_dataset_stats_features_path(dp_manager, DATASET_NAME), f'{DATASET_NAME}_{DEVICE}_groups_automatic_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
    ground_truth_path = osp.join(get_dataset_stats_features_path(dp_manager, DATASET_NAME), f'{DATASET_NAME}_{DEVICE}_ground_truth_automatic_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
    if not osp.exists(preprocessed_eda_path):
        for user_id, data in dataset['eda'].items():
            print(f"Extracting EDA Features of user {user_id}")
            for task_id, eda_signal in data.items():
                len_eda_signal = len(eda_signal)
                step = int(WINDOW_SHIFT * EDA_SAMPLING_RATE) # The true step to slide along the time axis of the signal
                first_iter = int(WINDOW_SIZE * EDA_SAMPLING_RATE) # The true index of the signal at a time-point 
                for current_iter in tqdm(range(first_iter, len_eda_signal, step)): # current_iter is "second_iter"
                    previous_iter = current_iter - first_iter
                    signal = swt_denoiser.denoise(eda_signal[previous_iter:current_iter])
                    if NORMALIZER != '_nonorm':
                        signal = MinMaxScaler().fit_transform(signal.reshape(-1, 1)).ravel()
                    signal = eda_processor.eda_clean(signal, EDA_SAMPLING_RATE)
                    preprocessed_eda.append(signal)
                    groups.append(user_id)
                    labels.append(dataset['ground_truth'][user_id][task_id + '.csv'])
        np.save(preprocessed_eda_path, np.array(preprocessed_eda))
        np.save(groups_path, np.array(groups))
        np.save(ground_truth_path, np.array(labels))
    else:
        preprocessed_eda_path = np.load(preprocessed_eda_path)
        groups = np.load(groups_path)
        ground_truth = np.load(ground_truth_path)


    for train_index, test_index in LeaveOneGroupOut().split(groups = groups):
        target_user_id = groups[test_index[0]]
        train_preprocessed_eda = preprocessed_eda[train_index]
        prepare_automatic(train_preprocessed_eda, target_user_id) 
        test_preprocessed_eda = preprocessed_eda[test_index]
        stats_features = process_automatic(test_preprocessed_eda, target_user_id)
        print(stats_features.shape)
        break


    # eda_stats_features.append(stats_feature)


    # eda_stats_features = np.array(eda_stats_features) # Transform to numpy array format
    # output_file_path = osp.join(get_dataset_stats_features_path(dp_manager, DATASET_NAME), f'{DATASET_NAME}_{DEVICE}_{SIGNAL_NAME}{NORMALIZER}_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
    # np.save(output_file_path, eda_stats_features)
