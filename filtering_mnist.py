#%%
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys 
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.lines as mlines
# import matplotlib.pyplot as plt
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# import matplotlib as mpl
# from ing_theme_matplotlib import mpl_style
# mpl_style(False)
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
import torch.optim as optim


from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, KMeansSMOTE
from imblearn.combine import SMOTEENN
from imblearn.metrics import geometric_mean_score
# from sklearn.calibration import CalibratedClassifierCV

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

# from catboost import CatBoostClassifier
# from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, KernelDensity
from sklearn.cluster import MiniBatchKMeans

import matplotlib.pyplot as plt 
# import seaborn as sns

from model_for_mnist import * 
from utils_for_mnist import *
from train import * 

from torchvision import transforms, datasets

import pickle
# %%

# print(train_dataset.data)
# print(train_dataset.targets)

# np.unique(train_dataset.targets.numpy(), return_counts=True)
#%%
# MNIST 데이터셋 불러오기
# target이 8인 데이터의 인덱스 찾기

if __name__ == "__main__" :
    BATCH_SIZE = 64
    EPOCHS = 10

    test_dataset = datasets.MNIST(root = "./data/MNIST",
                                train = False,
                                transform = transforms.ToTensor())

    test_dataset.targets = torch.isin(test_dataset.targets, torch.tensor([3, 4])).to(torch.int32)

    test_dataset = TensorDataset((test_dataset.data.unsqueeze(1)/255).to(device), test_dataset.targets.to(device))

    test_loader =  torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = BATCH_SIZE,
                                            shuffle = False)


    def train(model, epoch, criterion, optimizer, dataloader):
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            probs = model(data)
            log_probs = torch.log(probs)
            loss = criterion(log_probs, target)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), len(dataloader.dataset),
                    100. * (batch_idx + 1) / len(dataloader), loss.item()))

    p = 1.0

    dbsmote_total_precision_list = []
    dbsmote_total_recall_list = []
    dbsmote_total_auprc_list = []
    dbsmote_total_auc_list = []

    dfbs_total_precision_list = []
    dfbs_total_recall_list = []
    dfbs_total_auprc_list = []
    dfbs_total_auc_list = []

    ddhs_total_precision_list = []
    ddhs_total_recall_list = []
    ddhs_total_auprc_list = []
    ddhs_total_auc_list = []

    for rho in [0.05, 0.1, 0.2]:
        dbsmote_recall_list = []
        dbsmote_precision_list = []
        dbsmote_auprc_list = []
        dbsmote_auc_list = []
        
        dfbs_recall_list = []
        dfbs_precision_list = []
        dfbs_auprc_list = []
        dfbs_auc_list = []
        
        ddhs_recall_list = []
        ddhs_precision_list = []
        ddhs_auprc_list = []
        ddhs_auc_list = []
        
        noise_ratio = rho
        if rho == 0.2:
            EPOCHS2 = 8
            BATCH_SIZE2 = 128
        else:
            EPOCHS2 = 10
            BATCH_SIZE2 = 64
        
        for seed in range(10):
            
            train_dataset = datasets.MNIST(root = "./data/MNIST",
                                train = True,
                                download = True,
                                transform = transforms.ToTensor())
            
            set_random_seed(seed)
            indices_of_8 = (train_dataset.targets == 8).nonzero(as_tuple=True)[0]
            num_samples_to_change_in_8 = int(noise_ratio * len(indices_of_8))  
            indices_to_change_in_8 = np.random.choice(indices_of_8, num_samples_to_change_in_8, replace=False)
            train_dataset.targets[indices_to_change_in_8] = 3

            indices_of_9 = (train_dataset.targets == 9).nonzero(as_tuple=True)[0]
            num_samples_to_change_in_9 = int(noise_ratio * len(indices_of_9))  
            indices_to_change_in_9 = np.random.choice(indices_of_9, num_samples_to_change_in_9, replace=False)
            train_dataset.targets[indices_to_change_in_9] = 4

            train_dataset.targets = torch.isin(train_dataset.targets, torch.tensor([3, 4])).to(torch.int32)

            noise_index = np.concatenate([indices_to_change_in_9, indices_to_change_in_8], axis=0)


            _, n_labels = np.unique(train_dataset.targets.detach().numpy(), return_counts=True)
            target_num_samples = n_labels[0] - n_labels[1]
            
            train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = BATCH_SIZE,
                                                    shuffle = True)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            x_train_knn = train_dataset.data.reshape(len(train_dataset), -1).numpy()
            y_train_knn = train_dataset.targets.numpy()

            neigh = KNeighborsClassifier(n_neighbors=5)
            neigh.fit(x_train_knn, y_train_knn)
            y_pred_knn = neigh.predict(x_train_knn)
            confusion_matrix(y_train_knn, y_pred_knn)

            y_tmp = y_train_knn.copy()
            y_tmp[(y_train_knn == 0) & (y_pred_knn == 1)] = 2
            y_tmp[(y_train_knn == 1) & (y_pred_knn == 0)] = 3
            
            clf2 = CNN().to(device)

            train_dataset_clf2 = TensorDataset((train_dataset.data.unsqueeze(1)/255).to(device), torch.LongTensor(y_tmp).to(device))

            train_loader_clf2 =  torch.utils.data.DataLoader(dataset = train_dataset_clf2,
                                                    batch_size = BATCH_SIZE2,
                                                    shuffle = False)
            clf2_criterion = nn.NLLLoss()
            clf2_optimizer = optim.Adam(clf2.parameters(), lr=0.001)



            for epoch in range(EPOCHS2): 
                train(clf2, epoch, clf2_criterion, clf2_optimizer, train_loader_clf2)

            clf2.eval()
            
            config = {
                "dataset": "mnist",
                "input_dim": 28 * 28,
                "latent_dim": 2,
                "batch_size": 256,
                "epochs": 300,
                "recon": "l1",
                "activation": "sigmoid",
                "beta": 1.0,
                "sigma1" : 0.2,
                "sigma2" : 1.0,
                "dist": 1.0,
            } 

            config["method"] = "dbsmote"     
            # config["classifier"] = "rf"

            train_dataset_dbsmote = TensorDataset((train_dataset.data.unsqueeze(1).reshape(-1, 28 * 28)/255).to(device),
                                                  torch.LongTensor(y_tmp).to(device))

            train_loader_dbsmote =  torch.utils.data.DataLoader(dataset = train_dataset_dbsmote,
                                                    batch_size = config["batch_size"],
                                                    shuffle = False)

            model = MixtureMLPVAE(config, class_num=4, classifier=clf2).to(device)
            config["clf"] = clf2

            """optimizer"""
            optimizer = torch.optim.Adam(
                    list(model.encoder.parameters()) + list(model.decoder.parameters()), 
                    lr=0.001
                )
            config["beta"] = 5.0

            pbar = tqdm(range(50))

            for epoch in pbar:
                logs = dbsmote_image_train(model, train_loader_dbsmote, optimizer, config, device)
                pbar.set_description('====> Seed: {} Epoch: {} ELBO: {:.4f} KL {:.4f} Recon {:.4f} CCE {:.4f}'.format(
                    0, epoch, logs['elbo'][0], logs['kl'][0], logs['recon'][0], logs['cce'][0] ))

            ###  DFBS
            train_dataset_benchmark = TensorDataset((train_dataset.data.unsqueeze(1).reshape(-1, 28 * 28)/255).to(device),
                                                    train_dataset.targets.unsqueeze(-1).float().to(device))

            train_loader_benchmark =  torch.utils.data.DataLoader(dataset = train_dataset_benchmark,
                                                    batch_size = config["batch_size"],
                                                    shuffle = False)

            dfbs = AutoEncoder(config, device)

            dfbs_optimizer = torch.optim.Adam(
                    dfbs.parameters(), 
                    lr=0.001
                )

            pbar = tqdm(range(300))

            for epoch in pbar:
                logs = dfbs_train(dfbs, train_loader_benchmark, dfbs_optimizer, config, device)

            ### DDHS
            ddhs = DDHS(config, device) 

            ddhs_optimizer = torch.optim.Adam(
                    ddhs.parameters(), 
                    lr=0.001
                )

            pbar = tqdm(range(300))

            for epoch in pbar:
                logs = ddhs_train(ddhs, train_loader_benchmark, ddhs_optimizer, config, device)

            z_train_dbsmote = []
            z_train_dfbs = []
            z_train_ddhs = []
            
            y_train_dbsmote = []
            y_train = []
            x_train = []

            model.eval()
            dfbs.eval()
            ddhs.eval()
            
            for batch_x, batch_y in train_loader_dbsmote:
                _, _, _, _, _, z_tilde_dbsmote, _ = model(batch_x)
                z_train_dbsmote.append(z_tilde_dbsmote)
                y_train_dbsmote.append(batch_y)
                
            for batch_x, batch_y in train_loader_benchmark:
                z_train_1 = dfbs.encode(batch_x)
                z_train_2 = ddhs.encode(batch_x)
                
                z_train_dfbs.append(z_train_1)
                z_train_ddhs.append(z_train_2)
                
                x_train.append(batch_x)
                y_train.append(batch_y)
            
            z_train_dbsmote = torch.cat(z_train_dbsmote, dim=0).detach().numpy()
            z_train_dfbs = torch.cat(z_train_dfbs, dim=0).detach().numpy()
            z_train_ddhs = torch.cat(z_train_ddhs, dim=0).detach().numpy()    
            
            x_train = torch.cat(x_train, dim=0).detach().numpy()
            
            y_train_dbsmote = torch.cat(y_train_dbsmote, dim=0).squeeze().detach().numpy()
            y_train = torch.cat(y_train, dim=0).squeeze().detach().numpy()
            minor_idx, = np.where(y_train == 1)
            
            ################################# SMOTE-CLS #################################
            z_train_dbmoste_easy_minor = z_train_dbsmote[y_train_dbsmote == 1, :]
            z_train_dbsmote_hard_minor = z_train_dbsmote[y_train_dbsmote == 3, :]

            minor_easy_kde = KernelDensity(kernel='gaussian', bandwidth="scott").fit(z_train_dbmoste_easy_minor)
            minor_hard_kde = KernelDensity(kernel='gaussian', bandwidth="scott").fit(z_train_dbsmote_hard_minor)

            minor_easy_score = minor_easy_kde.score_samples(z_train_dbmoste_easy_minor)
            minor_hard_score = minor_hard_kde.score_samples(z_train_dbsmote_hard_minor)

            minor_easy_threshold = np.quantile(minor_easy_score, 0.1)
            minor_hard_threshold = np.quantile(minor_hard_score, 0.95)

            idx_easy_minor, = np.where(y_train_dbsmote == 1)
            idx_hard_minor, = np.where(y_train_dbsmote == 3)

            noise_scores_in_easy = minor_easy_kde.score_samples(z_train_dbsmote[list(set(idx_easy_minor).intersection(noise_index))])
            noise_scores_in_hard = minor_hard_kde.score_samples(z_train_dbsmote[list(set(idx_hard_minor).intersection(noise_index))])

            noise_deteced_in_easy = noise_scores_in_easy < minor_easy_threshold
            noise_deteced_in_hard = noise_scores_in_hard < minor_hard_threshold

            filtered_out_in_easy = minor_easy_score < minor_easy_threshold
            filtered_out_in_hard = minor_hard_score < minor_hard_threshold
            
            recall_dbsmote = (noise_deteced_in_easy.sum() + noise_deteced_in_hard.sum()) / noise_index.shape[0]
            precision_dbsmote = (noise_deteced_in_easy.sum() + noise_deteced_in_hard.sum()) / (filtered_out_in_easy.sum() + filtered_out_in_hard.sum())
            
            X_res1, y_res1 = dbsmote_sampling(model, round(n_labels[0] * p), config, train_loader_dbsmote, train_loader_benchmark, seed)
            
            donwstream_clf = CNN(output_dim=2).to(device)

            train_dataset_donwstream_clf = TensorDataset(torch.Tensor(X_res1).reshape(-1, 1, 28, 28).to(device), torch.LongTensor(y_res1).to(device))

            train_loader_donwstream_clf =  torch.utils.data.DataLoader(dataset = train_dataset_donwstream_clf,
                                                    batch_size = BATCH_SIZE,
                                                    shuffle = True)
            donwstream_clf_criterion = nn.NLLLoss()
            donwstream_clf_optimizer = optim.Adam(donwstream_clf.parameters(), lr=0.001)

            for epoch in range(EPOCHS): 
                train(donwstream_clf, epoch, donwstream_clf_criterion, donwstream_clf_optimizer, train_loader_donwstream_clf)

            donwstream_clf.eval()
            
            y_preds = []
            y_true = []
            
            for x, y  in test_loader:
                y_pred = donwstream_clf(x)
                y_preds.append(y_pred)
                y_true.append(y)
            
            y_preds = torch.cat(y_preds, dim=0).detach().numpy()
            y_true = torch.cat(y_true, dim=0).detach().numpy()
                
            auc_dbsmote = metrics.roc_auc_score(y_true, y_preds[:, 1])
            auprc_dbsmote = metrics.average_precision_score(y_true, y_preds[:, 1], average='macro')
            
            ################################# DFBS #################################
            major_center = z_train_dfbs[y_train == 0, :].mean(axis=0)
            minor_center = z_train_dfbs[y_train == 1, :].mean(axis=0)
            all_center = z_train_dfbs.mean(axis=0)

            noise_z_dfbs = z_train_dfbs[noise_index]
            noise_dist_M = np.power(noise_z_dfbs - major_center, 2).sum(axis=1) 
            noise_dist_A = np.power(noise_z_dfbs - all_center, 2).sum(axis=1)
            noise_dist_m = np.power(noise_z_dfbs - minor_center, 2).sum(axis=1) 

            noise_z_filtered_out_dfbs = noise_z_dfbs[((noise_dist_M <= noise_dist_A) | (noise_dist_A <= noise_dist_m))]

            minor_not_noise_z_dfbs = z_train_dfbs[list(set(minor_idx).difference(noise_index))]
            minor_not_noise_dist_M = np.power(minor_not_noise_z_dfbs - major_center, 2).sum(axis=1) 
            minor_not_noise_dist_A = np.power(minor_not_noise_z_dfbs - all_center, 2).sum(axis=1)
            minor_not_noise_dist_m = np.power(minor_not_noise_z_dfbs - minor_center, 2).sum(axis=1) 

            minor_not_noise_z_filtered_out_dbfs = minor_not_noise_z_dfbs[((minor_not_noise_dist_M <= minor_not_noise_dist_A) | (minor_not_noise_dist_A <= minor_not_noise_dist_m))]

            recall_dfbs = noise_z_filtered_out_dfbs.shape[0] / noise_index.shape[0]
            precision_dfbs = noise_z_filtered_out_dfbs.shape[0] / (noise_z_filtered_out_dfbs.shape[0] + minor_not_noise_z_filtered_out_dbfs.shape[0])

            x_oversampled_dfbs = dfbs_sampling(dfbs, target_num_samples, config, train_loader_benchmark, seed)
            
            donwstream_clf_dfbs = CNN(output_dim=2).to(device)

            train_dataset_donwstream_clf_dfbs = TensorDataset(torch.Tensor(np.concatenate([x_train, x_oversampled_dfbs], axis=0)).reshape(-1, 1, 28, 28).to(device),
                                                              torch.LongTensor(np.concatenate([y_train, np.ones((target_num_samples))], axis=0)).to(device))
            
            train_loader_donwstream_clf_dfbs =  torch.utils.data.DataLoader(dataset = train_dataset_donwstream_clf_dfbs,
                                                    batch_size = BATCH_SIZE,
                                                    shuffle = True)
            donwstream_clf_dfbs_criterion = nn.NLLLoss()
            donwstream_clf_dfbs_optimizer = optim.Adam(donwstream_clf_dfbs.parameters(), lr=0.001)

            for epoch in range(EPOCHS): 
                train(donwstream_clf_dfbs, epoch, donwstream_clf_dfbs_criterion, donwstream_clf_dfbs_optimizer, train_loader_donwstream_clf_dfbs)

            donwstream_clf_dfbs.eval()
            
            y_preds = []
            y_true = []
            
            for x, y  in test_loader:
                y_pred = donwstream_clf_dfbs(x)
                y_preds.append(y_pred)
                y_true.append(y)
            
            y_preds_dfbs = torch.cat(y_preds, dim=0).detach().numpy()
            y_true = torch.cat(y_true, dim=0).detach().numpy()
                
            auc_dfbs = metrics.roc_auc_score(y_true, y_preds_dfbs[:, 1])
            auprc_dfbs = metrics.average_precision_score(y_true, y_preds_dfbs[:, 1], average='macro')
            
            ################################# DDHS #################################
            minor_kde = KernelDensity(kernel='gaussian').fit(z_train_ddhs[y_train == 1, :])
            major_kde = KernelDensity(kernel='gaussian').fit(z_train_ddhs[y_train == 0, :])

            minor_score = minor_kde.score_samples(z_train_ddhs[y_train == 1, :])
            major_score = major_kde.score_samples(z_train_ddhs[y_train == 0, :])

            minor_score_q3 = np.quantile(minor_score, 0.75)
            major_score_q2 = np.quantile(major_score, 0.5)

            noise_z_ddhs = z_train_ddhs[noise_index]
            minor_not_noise_z_ddhs = z_train_ddhs[list(set(minor_idx).difference(noise_index))]

            minor_not_noise_score_ddhs = minor_kde.score_samples(minor_not_noise_z_ddhs)
            noise_score_ddhs = minor_kde.score_samples(noise_z_ddhs)

            filtered_out_minor_not_noise_z_ddhs = minor_not_noise_z_ddhs[minor_not_noise_score_ddhs < minor_score_q3]
            filtered_out_noise_z_ddhs = noise_z_ddhs[noise_score_ddhs < minor_score_q3]
            
            recall_ddhs = filtered_out_noise_z_ddhs.shape[0] / noise_index.shape[0]
            precision_ddhs = filtered_out_noise_z_ddhs.shape[0] / (filtered_out_minor_not_noise_z_ddhs.shape[0] + filtered_out_noise_z_ddhs.shape[0])
            
            x_oversampled_ddhs = ddhs_sampling(ddhs, target_num_samples, config, train_loader_benchmark, seed)
            
            donwstream_clf_ddhs = CNN(output_dim=2).to(device)

            train_dataset_donwstream_clf_ddhs = TensorDataset(torch.Tensor(np.concatenate([x_train, x_oversampled_ddhs], axis=0)).reshape(-1, 1, 28, 28).to(device),
                                                              torch.LongTensor(np.concatenate([y_train, np.ones((target_num_samples))], axis=0)).to(device))

            train_loader_donwstream_clf_ddhs =  torch.utils.data.DataLoader(dataset = train_dataset_donwstream_clf_ddhs,
                                                    batch_size = BATCH_SIZE,
                                                    shuffle = True)
            donwstream_clf_ddhs_criterion = nn.NLLLoss()
            donwstream_clf_ddhs_optimizer = optim.Adam(donwstream_clf_ddhs.parameters(), lr=0.001)

            for epoch in range(EPOCHS): 
                train(donwstream_clf_ddhs, epoch, donwstream_clf_ddhs_criterion, donwstream_clf_ddhs_optimizer, train_loader_donwstream_clf_ddhs)

            donwstream_clf_ddhs.eval()
            
            y_preds = []
            y_true = []
            
            for x, y  in test_loader:
                y_pred = donwstream_clf_ddhs(x)
                y_preds.append(y_pred)
                y_true.append(y)
            
            y_preds_ddhs = torch.cat(y_preds, dim=0).detach().numpy()
            y_true = torch.cat(y_true, dim=0).detach().numpy()
                
            auc_ddhs = metrics.roc_auc_score(y_true, y_preds_ddhs[:, 1])
            auprc_ddhs = metrics.average_precision_score(y_true, y_preds_ddhs[:, 1], average='macro')
            
            ################################# Results #################################
            dbsmote_recall_list.append(recall_dbsmote)
            dbsmote_precision_list.append(precision_dbsmote)
            dbsmote_auprc_list.append(auprc_dbsmote)
            dbsmote_auc_list.append(auc_dbsmote)
            
            dfbs_recall_list.append(recall_dfbs)
            dfbs_precision_list.append(precision_dfbs)
            dfbs_auprc_list.append(auprc_dfbs)
            dfbs_auc_list.append(auc_dfbs)

            ddhs_recall_list.append(recall_ddhs)
            ddhs_precision_list.append(precision_ddhs)
            ddhs_auprc_list.append(auprc_ddhs)
            ddhs_auc_list.append(auc_ddhs)
            
        dbsmote_total_precision_list.append(dbsmote_precision_list)
        dbsmote_total_recall_list.append(dbsmote_recall_list)
        dbsmote_total_auprc_list.append(dbsmote_auprc_list)
        dbsmote_total_auc_list.append(dbsmote_auc_list)
        
        dfbs_total_precision_list.append(dfbs_precision_list)
        dfbs_total_recall_list.append(dfbs_recall_list)
        dfbs_total_auprc_list.append(dfbs_auprc_list)
        dfbs_total_auc_list.append(dfbs_auc_list)

        ddhs_total_precision_list.append(ddhs_precision_list) 
        ddhs_total_recall_list.append(ddhs_recall_list)
        ddhs_total_auprc_list.append(ddhs_auprc_list)
        ddhs_total_auc_list.append(ddhs_auc_list)
        
        with open('/home/optim1/Desktop/hong/smotecls/results/dbsmote_precision.pkl', 'wb') as f:
            pickle.dump(dbsmote_total_precision_list, f)
            
        with open('/home/optim1/Desktop/hong/smotecls/results/dbsmote_recall.pkl', 'wb') as f:
            pickle.dump(dbsmote_total_recall_list, f)

        with open('/home/optim1/Desktop/hong/smotecls/results/dbsmote_auprc.pkl', 'wb') as f:
            pickle.dump(dbsmote_total_auprc_list, f)
            
        with open('/home/optim1/Desktop/hong/smotecls/results/dbsmote_auc.pkl', 'wb') as f:
            pickle.dump(dbsmote_total_auc_list, f)

        with open('/home/optim1/Desktop/hong/smotecls/results/dfbs_precision.pkl', 'wb') as f:
            pickle.dump(dfbs_total_precision_list, f)
            
        with open('/home/optim1/Desktop/hong/smotecls/results/dfbs_recall.pkl', 'wb') as f:
            pickle.dump(dfbs_total_recall_list, f)

        with open('/home/optim1/Desktop/hong/smotecls/results/dfbs_auprc.pkl', 'wb') as f:
            pickle.dump(dfbs_total_auprc_list, f)
            
        with open('/home/optim1/Desktop/hong/smotecls/results/dfbs_auc.pkl', 'wb') as f:
            pickle.dump(dfbs_total_auc_list, f)
            
        with open('/home/optim1/Desktop/hong/smotecls/results/ddhs_precision.pkl', 'wb') as f:
            pickle.dump(ddhs_total_precision_list, f)
            
        with open('/home/optim1/Desktop/hong/smotecls/results/ddhs_recall.pkl', 'wb') as f:
            pickle.dump(ddhs_total_recall_list, f)

        with open('/home/optim1/Desktop/hong/smotecls/results/ddhs_auprc.pkl', 'wb') as f:
            pickle.dump(ddhs_total_auprc_list, f)
            
        with open('/home/optim1/Desktop/hong/smotecls/results/ddhs_auc.pkl', 'wb') as f:
            pickle.dump(ddhs_total_auc_list, f)
#%% SMOTE-CLS
# samples = []
# true_y = []
# z_tildes = []
# predicted_y = []
# xhats = []

# for batch_x, batch_y in train_loader_dbsmote:
#     mean, logvar, probs, y_, z, z_tilde, xhat = model(batch_x, sampling=False)
#     samples.append(mean)
#     true_y.append(batch_y)
#     z_tildes.append(z_tilde)
#     predicted_y.append(y_)
#     xhats.append(xhat)

# samples = torch.cat(samples, dim=0)
# samples = samples.detach()
# true_y = torch.cat(true_y, dim=0)
# true_y = true_y.detach().squeeze()
# z_tildes = torch.cat(z_tildes, dim=0)
# z_tildes = z_tildes.detach().squeeze()
# predicted_y = torch.cat(predicted_y, dim=0)
# predicted_y = predicted_y.detach().squeeze()
# xhats = torch.cat(xhats, dim=0)

# mpl.rcParams["figure.figsize"] = (6,6)
# plt.rcParams['figure.dpi'] = 100
# sns.scatterplot(x=z_tildes[train_dataset.targets==0, 0], y=z_tildes[train_dataset.targets==0, 1], s=40, alpha=0.8, c="#66b2ff",  label="Major")
# sns.scatterplot(x=z_tildes[train_dataset.targets==1, 0], y=z_tildes[train_dataset.targets==1, 1], s=40, alpha=0.8, c="#ff8000", label="Minor")
# sns.scatterplot(x=z_tildes[noise_index, 0], y=z_tildes[noise_index, 1], s=40, alpha=0.8, c="k", label="Noise")
# plt.legend("", frameon=False)
# plt.show()

# %%
# samples = []
# # true_y = []
# z_tildes = []

# for batch_x, batch_y in train_loader_dbsmote:
#     xhat = dfbs(batch_x)
#     z_tilde = dfbs.encode(batch_x)
#     samples.append(xhat)
#     # true_y.append(batch_y)
#     z_tildes.append(z_tilde)

# samples = torch.cat(samples, dim=0)
# samples = samples.detach()
# # true_y = torch.cat(true_y, dim=0)
# # true_y = true_y.detach().squeeze()
# z_tildes = torch.cat(z_tildes, dim=0)
# z_tildes = z_tildes.detach().squeeze()

# mpl.rcParams["figure.figsize"] = (6,6)
# plt.rcParams['figure.dpi'] = 100

# sns.scatterplot(x=z_tildes[train_dataset.targets==0, 0], y=z_tildes[train_dataset.targets==0, 1], s=40, alpha=0.8, c="#66b2ff",  label="Major")
# sns.scatterplot(x=z_tildes[train_dataset.targets==1, 0], y=z_tildes[train_dataset.targets==1, 1], s=40, alpha=0.8, c="#ff8000", label="Minor")
# sns.scatterplot(x=z_tildes[noise_index, 0], y=z_tildes[noise_index, 1], s=40, alpha=0.8, c="k", label="Noise")
# # sns.scatterplot(x=z_tildes[true_y==3, 0], y=z_tildes[true_y==3, 1], s=40, alpha=0.8, c="#ff3333", label="Hard Minor")
# mark1 = mlines.Line2D([], [], c="#66b2ff", lw=0, marker='o', label="Major")
# mark2 = mlines.Line2D([], [], c="#ff8000", lw=0, marker='o', label="Minor")
# mark3 = mlines.Line2D([], [], c="k", lw=0, marker='o', label="Noise")
# plt.legend(handles=[mark1, mark2, mark3], fontsize="15", markerscale=2., loc='upper center', bbox_to_anchor=(0.5, 1.05),
#           ncol=3, fancybox=True, shadow=True)
# plt.show()
# %%
# samples = []
# z_tildes = []
# predicted_y = []

# for batch_x, batch_y in train_loader_benchmark:
#     xhat, yhat = ddhs(batch_x)
#     z_tilde = ddhs.encode(batch_x)
#     samples.append(xhat)
#     z_tildes.append(z_tilde)
#     predicted_y.append(yhat)

# samples = torch.cat(samples, dim=0)
# samples = samples.detach()
# z_tildes = torch.cat(z_tildes, dim=0)
# z_tildes = z_tildes.detach().squeeze()
# predicted_y = torch.cat(predicted_y, dim=0)
# predicted_y = predicted_y.detach().squeeze()

# sns.scatterplot(x=z_tildes[train_dataset.targets==0, 0], y=z_tildes[train_dataset.targets==0, 1], s=40, alpha=0.8, c="#66b2ff",  label="Major")
# sns.scatterplot(x=z_tildes[train_dataset.targets==1, 0], y=z_tildes[train_dataset.targets==1, 1], s=40, alpha=0.8, c="#ff8000", label="Minor")
# sns.scatterplot(x=z_tildes[noise_index, 0], y=z_tildes[noise_index, 1], s=40, alpha=0.8, c="k", label="Noise")
# plt.xlim(-1.2, 1.2)
# plt.ylim(-0.7, 1.25)
# plt.legend('',frameon=False)


# %%
# sns.scatterplot(x=z_tildes[train_dataset.targets==0, 0], y=z_tildes[train_dataset.targets==0, 1], s=40, alpha=0.8, c="#66b2ff",  label="Major")

# sns.scatterplot(x=z_train_easy_minor[minor_easy_score > minor_easy_threshold, 0], y=z_train_easy_minor[minor_easy_score > minor_easy_threshold, 1], s=40, alpha=0.8, c="#ff8000", label="Minor")
# sns.scatterplot(x=z_train_hard_minor[minor_hard_score > minor_hard_threshold, 0], y=z_train_hard_minor[minor_hard_score > minor_hard_threshold, 1], s=40, alpha=0.8, c="#ff8000", label="Minor")

# sns.scatterplot(x=z_tildes[list(set(idx_easy_minor).intersection(noise_index))][idx1, 0],
#                 y=z_tildes[list(set(idx_easy_minor).intersection(noise_index))][idx1, 1],
#                 s=40, alpha=0.8, c="k", label="Noise")
# sns.scatterplot(x=z_tildes[list(set(idx_hard_minor).intersection(noise_index))][idx2, 0],
#                 y=z_tildes[list(set(idx_hard_minor).intersection(noise_index))][idx2, 1],
#                 s=40, alpha=0.8, c="k", label="Noise")

# plt.legend('',frameon=False)
#%%

#%%
# sns.scatterplot(x=samples[train_dataset.targets==0, 0],
#                 y=samples[train_dataset.targets==0, 1], s=40, alpha=0.8, c="#66b2ff",  label="Major")

# sns.scatterplot(x=minor_not_noise_z_filtered[:, 0].detach().numpy(), 
#                 y=minor_not_noise_z_filtered[:, 1].detach().numpy(),
#                 s=40, alpha=0.8, c="#ff8000", label="Minor")

# sns.scatterplot(x=noise_z_filtered[:, 0].detach().numpy(),
#                 y=noise_z_filtered[:, 1].detach().numpy(),
#                 s=40, alpha=0.8, c="k", label="Noise")

# mark1 = mlines.Line2D([], [], c="#66b2ff", lw=0, marker='o', label="Major")
# mark2 = mlines.Line2D([], [], c="#ff8000", lw=0, marker='o', label="Minor")
# mark3 = mlines.Line2D([], [], c="k", lw=0, marker='o', label="Noise")
# plt.legend(handles=[mark1, mark2, mark3], fontsize="15", markerscale=2., loc='upper center', bbox_to_anchor=(0.5, 1.05),
#           ncol=3, fancybox=True, shadow=True)
# plt.show()
#%%
# sns.scatterplot(x=selected_major_z[:, 0],
#                 y=selected_major_z[:, 1], s=40, alpha=0.8, c="#66b2ff",  label="Major")

# sns.scatterplot(x=selected_minor_not_noise_z[:, 0].detach().numpy(), 
#                 y=selected_minor_not_noise_z[:, 1].detach().numpy(),
#                 s=40, alpha=0.8, c="#ff8000", label="Minor")

# sns.scatterplot(x=selected_noise_z[:, 0].detach().numpy(),
#                 y=selected_noise_z[:, 1].detach().numpy(),
#                 s=40, alpha=0.8, c="k", label="Noise")
# plt.xlim(-1.2, 1.2)
# plt.ylim(-0.7, 1.25)
# plt.legend('',frameon=False)
# plt.show()
#%%

# with open("/Users/shong/Desktop/lab/working_papers/SMOTE_CLS/src/mnist/results/dbsmote_precision.pkl","rb") as fr:
#     dbsmote_precision = pickle.load(fr)

# with open("/Users/shong/Desktop/lab/working_papers/SMOTE_CLS/src/mnist/results/dbsmote_recall.pkl","rb") as fr:
#     dbsmote_recall = pickle.load(fr)

# with open("/Users/shong/Desktop/lab/working_papers/SMOTE_CLS/src/mnist/results/dbsmote_auc.pkl","rb") as fr:
#     dbsmote_auc = pickle.load(fr)

# with open("/Users/shong/Desktop/lab/working_papers/SMOTE_CLS/src/mnist/results/dbsmote_auprc.pkl","rb") as fr:
#     dbsmote_auprc = pickle.load(fr)

# np.array(dbsmote_precision[0]).mean().round(3)
# np.array(dbsmote_recall[1]).mean().round(3)
# np.array(dbsmote_auprc[0]).mean().round(3)


# (np.array(dbsmote_recall[1]).std()/np.sqrt(10)).round(3)

# np.array(dbsmote_auprc[0]).std().round(3)
# np.array(dbsmote_auprc[1]).std().round(3)
# np.array(dbsmote_precision[1]).mean().round(3)
# np.array(dbsmote_recall[1]).mean().round(3)
# np.array(dbsmote_auprc[1]).mean().round(3)

# ((np.array(dbsmote_precision[1]) * np.array(dbsmote_precision[1]))/ (np.array(dbsmote_precision[1]) + np.array(dbsmote_precision[1]))).mean().round(3)*2

# ((((np.array(dbsmote_precision[0]) * np.array(dbsmote_precision[0]))/ (np.array(dbsmote_precision[0]) + np.array(dbsmote_precision[0]))) * 2).std()/np.sqrt(10)).round(3)