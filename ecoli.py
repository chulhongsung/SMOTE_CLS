from tqdm import tqdm
import os
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torch import nn
import torch.nn.functional as F

from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, KMeansSMOTE
from imblearn.combine import SMOTEENN
from imblearn.metrics import geometric_mean_score

from torch.utils.data import TensorDataset, DataLoader

from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.cluster import MiniBatchKMeans

from model import * 
from utils import *
from train import * 

def smote_cls_sampling(model, target_num_samples, train_loader1, train_loader2, seed):
    set_random_seed(seed)

    z_train = []
    y_train = []
    x_train = []

    model.eval()
    
    for batch_x, batch_y in train_loader1:

        _, _, _, _, _, z_tilde, _ = model(batch_x)
        z_train.append(z_tilde)
        y_train.append(batch_y)
        
    for batch_x, batch_y in train_loader2:
        x_train.append(batch_x)
        
    z_train = torch.cat(z_train, dim=0).detach().numpy()
    x_train = torch.cat(x_train, dim=0).detach().numpy()
    y_train = torch.cat(y_train, dim=0).squeeze().detach().numpy()
    
    x_train_minor = x_train[(y_train == 3) | (y_train == 1), :]
    x_train_major = x_train[(y_train == 0) | (y_train == 2), :]
    
    z_train_easy_minor = z_train[y_train == 1, :]
    z_train_hard_minor = z_train[y_train == 3, :]
    
    x_train_easy_minor = x_train[y_train == 1, :]
    x_train_hard_minor = x_train[y_train == 3, :]
    
    minor_easy_kde = KernelDensity(kernel='gaussian', bandwidth="scott").fit(z_train_easy_minor)
    minor_hard_kde = KernelDensity(kernel='gaussian', bandwidth="scott").fit(z_train_hard_minor)

    minor_easy_score = minor_easy_kde.score_samples(z_train_easy_minor)
    minor_hard_score = minor_hard_kde.score_samples(z_train_hard_minor)

    minor_easy_threshold = np.quantile(minor_easy_score, 0.05)
    minor_hard_threshold = np.quantile(minor_hard_score, 0.2)

    x_train_filtered_easy_minor = x_train_easy_minor[minor_easy_score >= minor_easy_threshold, :]
    x_train_filtered_hard_minor = x_train_hard_minor[minor_hard_score >= minor_hard_threshold, :]

    x_train_filtered_minor = np.concatenate([x_train_filtered_easy_minor, x_train_filtered_hard_minor], axis=0)
    
    x_train_filtered = np.concatenate([x_train_major, x_train_filtered_minor], axis=0)
    y_train_filtered = np.concatenate([np.zeros((x_train_major.shape[0])), np.ones((x_train_filtered_minor.shape[0]))], axis=0)

    sm = SMOTE(random_state=seed, sampling_strategy={1: target_num_samples})
    X_res, y_res = sm.fit_resample(x_train_filtered, y_train_filtered)
    
    bsm = BorderlineSMOTE(random_state=seed, sampling_strategy={1: target_num_samples})
    X_res2, y_res2 = bsm.fit_resample(x_train_filtered, y_train_filtered)

    return X_res, y_res, X_res2, y_res2

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = fetch_datasets()["ecoli"]
    X, y = df.data, df.target
    y = np.where(y == -1, 0, y)
    np.unique(y, return_counts=True)

    smote_cls_auc_list = []
    smote_cls_auprc_list = []
    smote_cls_gm_list = []
    smote_cls_f1_list = []

    smote_auc_list = []
    smote_auprc_list = []
    smote_gm_list = []
    smote_f1_list = []

    bsmote_auc_list = []
    bsmote_auprc_list = []
    bsmote_gm_list = []
    bsmote_f1_list = []

    sme_auc_list = []
    sme_auprc_list = []
    sme_gm_list = []
    sme_f1_list = []

    ksmote_auc_list = []
    ksmote_auprc_list = []
    ksmote_gm_list = []
    ksmote_f1_list = []

    cvae_auc_list = []
    cvae_auprc_list = []
    cvae_gm_list = []
    cvae_f1_list = []

    base_auc_list = []
    base_auprc_list = []
    base_gm_list = []
    base_f1_list = []

    ddhs_auc_list = []
    ddhs_auprc_list = []
    ddhs_gm_list = []
    ddhs_f1_list = []

    dfbs_auc_list = []
    dfbs_auprc_list = []
    dfbs_gm_list = []
    dfbs_f1_list = []

    dsmote_auc_list = []
    dsmote_auprc_list = []
    dsmote_gm_list = []
    dsmote_f1_list = []

    seed = 1
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    config = {
        "dataset": "ecoli",
        "latent_dim": 2,
        "batch_size": 512,
        "epochs": 300,
        "input_dim": x_train.shape[1],
        "recon": "l1",
        "activation": "identity",
        "beta": 1.0,
        "sigma1" : 0.1,
        "sigma2" : 1.0,
        "dist": 1.0,
    } 

    config["method"] = "dbsmote"     
    config["classifier"] = "rf"

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x_train, y_train)
    y_pred = neigh.predict(x_train)

    y_tmp = y_train.copy()
    y_tmp[(y_train == 0) & (y_pred == 1)] = 2
    y_tmp[(y_train == 1) & (y_pred == 0)] = 3

    clf2 = XGBClassifier(
            n_estimators=15, 
            max_depth=6, 
            gamma = 0.3, 
            importance_type='gain', 
            reg_lambda = 0.1, 
            random_state=0
        )

    clf2.fit(x_train, y_tmp)
    y_pred = clf2.predict(x_train)
    confusion_matrix(y_tmp, y_pred)

    dataset = TensorDataset(torch.from_numpy(x_train.astype('float32')), torch.Tensor(y_tmp[:, np.newaxis]))
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, drop_last=False)


    model = MixtureMLPVAE(config, class_num=4, classifier=clf2)
    config["clf"] = clf2

    """optimizer"""
    optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder.parameters()), 
            lr=0.001
        )
    config["beta"] = 0.15

    pbar = tqdm(range(400))

    for epoch in pbar:
        logs = dbsmote_train(model, dataloader, optimizer, config, device)
        pbar.set_description('====> Seed: {} Epoch: {} ELBO: {:.4f} KL {:.4f} Recon {:.4f} CCE {:.4f}'.format(
            0, epoch, logs['elbo'][0], logs['kl'][0], logs['recon'][0], logs['cce'][0] ))

    samples = []
    true_y = []
    z_tildes = []
    predicted_y = []
    xhats = []

    for batch_x, batch_y in dataloader:
        mean, logvar, probs, y_, z, z_tilde, xhat = model(batch_x, sampling=False)
        samples.append(mean)
        true_y.append(batch_y)
        z_tildes.append(z_tilde)
        predicted_y.append(y_)
        xhats.append(xhat)
 
    samples = torch.cat(samples, dim=0)
    samples = samples.detach()
    true_y = torch.cat(true_y, dim=0)
    true_y = true_y.detach().squeeze()
    z_tildes = torch.cat(z_tildes, dim=0)
    z_tildes = z_tildes.detach().squeeze()
    predicted_y = torch.cat(predicted_y, dim=0)
    predicted_y = predicted_y.detach().squeeze()
    xhats = torch.cat(xhats, dim=0)

    dataset_for_benchmark = TensorDataset(torch.from_numpy(x_train.astype('float32')), torch.Tensor(y_tmp[:, np.newaxis]))
    dataloader_for_benchmark = DataLoader(dataset_for_benchmark, batch_size=512, shuffle=False, drop_last=False)

    config["input_dim"] = x_train.shape[1]

    cvae = CVAE(config).to(device)
    
    optimizer = torch.optim.Adam(cvae.parameters(), lr=0.001)
    
    pbar = tqdm(range(500))

    for epoch in pbar:
        logs = cvae_train(cvae, dataloader_for_benchmark, optimizer, device)

    ddhs = DDHS(config, device) 

    optimizer = torch.optim.Adam(
            list(ddhs.encoder.parameters()) + list(ddhs.decoder.parameters()), 
            lr=0.001
        )

    pbar = tqdm(range(500))

    for epoch in pbar:
        logs = ddhs_train(ddhs, dataloader_for_benchmark, optimizer, config, device)


    dfbs = AutoEncoder(config, device)

    optimizer = torch.optim.Adam(
            list(dfbs.encoder.parameters()) + list(dfbs.decoder.parameters()), 
            lr=0.001
        )

    pbar = tqdm(range(500))

    for epoch in pbar:
        logs = dfbs_train(dfbs, dataloader_for_benchmark, optimizer, config, device)

    dsmote = AutoEncoder(config, device)

    optimizer = torch.optim.Adam(
            list(dsmote.encoder.parameters()) + list(dsmote.decoder.parameters()), 
            lr=0.001
        )

    pbar = tqdm(range(500))

    for epoch in pbar:
        logs = dsmote_train(dsmote, dataloader_for_benchmark, optimizer, config, device)


    _, n_labels = np.unique(y_train,return_counts=True)
    target_num_samples = n_labels[0] - n_labels[1]

    p = 1.0
    
    for seed in np.arange(10):
        
        X_res1, y_res1, X_res2, y_res2 = smote_cls_sampling(model, round(n_labels[0] * p), config, dataloader, dataloader_for_benchmark, seed)

        clf_smote_cls =  RandomForestClassifier(
                n_estimators=10, 
                max_depth=6, 
                criterion='gini', 
                random_state=seed
            )

        clf_smote_cls.fit(X_res1, y_res1)

        y_pred_clf_oversample = clf_smote_cls.predict_proba(x_test)[:,1]
    
        clf_smote_cls_auc = metrics.roc_auc_score(y_test, y_pred_clf_oversample)
        clf_smote_cls_auprc = metrics.average_precision_score(y_test, y_pred_clf_oversample, average='macro')
        
        clf_smote_cls_f1 = metrics.f1_score(y_test, clf_smote_cls.predict(x_test), average='macro')
        clf_smote_cls_gm = geometric_mean_score(y_test, clf_smote_cls.predict(x_test), average='macro')
                
        smote_cls_auc_list.append(clf_smote_cls_auc)
        smote_cls_auprc_list.append(clf_smote_cls_auprc)
        smote_cls_f1_list.append(clf_smote_cls_f1)
        smote_cls_gm_list.append(clf_smote_cls_gm)
        
        sm = SMOTE(random_state=seed, sampling_strategy={1: round(n_labels[0] * p)})
        X_res, y_res = sm.fit_resample(x_train, y_train)

        clf_smote =  RandomForestClassifier(
                n_estimators=10, 
                max_depth=6, 
                criterion='gini', 
                random_state=seed
            )
        clf_smote.fit(X_res, y_res)

        y_pred_clf_smote = clf_smote.predict_proba(x_test)[:,1]

        clf_smote_auc = metrics.roc_auc_score(y_test, y_pred_clf_smote)
        clf_smote_auprc = metrics.average_precision_score(y_test, y_pred_clf_smote, average='macro')
        
        clf_smote_f1 = metrics.f1_score(y_test, clf_smote.predict(x_test), average='macro')
        clf_smote_gm = geometric_mean_score(y_test, clf_smote.predict(x_test), average='macro')
        
        smote_auc_list.append(clf_smote_auc)
        smote_auprc_list.append(clf_smote_auprc)
        smote_f1_list.append(clf_smote_f1)
        smote_gm_list.append(clf_smote_gm)
   
        bsm = BorderlineSMOTE(random_state=seed, sampling_strategy={1: round(n_labels[0] * p)})
        X_res, y_res = bsm.fit_resample(x_train, y_train)

        clf_bsmote =  RandomForestClassifier(
                n_estimators=10, 
                max_depth=6, 
                criterion='gini', 
                random_state=seed
            )

        clf_bsmote.fit(X_res, y_res)

        y_pred_clf_bsmote = clf_bsmote.predict_proba(x_test)[:,1]

        clf_bsmote_auc = metrics.roc_auc_score(y_test, y_pred_clf_bsmote)
        clf_bsmote_auprc = metrics.average_precision_score(y_test, y_pred_clf_bsmote, average='macro')

        clf_bsmote_f1 = metrics.f1_score(y_test, clf_bsmote.predict(x_test), average='macro')
        clf_bsmote_gm = geometric_mean_score(y_test, clf_bsmote.predict(x_test), average='macro')
        
        bsmote_auc_list.append(clf_bsmote_auc)
        bsmote_auprc_list.append(clf_bsmote_auprc)
        bsmote_f1_list.append(clf_bsmote_f1)
        bsmote_gm_list.append(clf_bsmote_gm)

        sme = SMOTEENN(random_state=seed, sampling_strategy={1: round(n_labels[0] * p)})
        X_res, y_res = sme.fit_resample(x_train, y_train)

        clf_sme =  RandomForestClassifier(
                n_estimators=10, 
                max_depth=6, 
                criterion='gini', 
                random_state=seed
            )

        clf_sme.fit(X_res, y_res)

        y_pred_clf_sme = clf_sme.predict_proba(x_test)[:,1]

        clf_sme_auc = metrics.roc_auc_score(y_test, y_pred_clf_sme)
        clf_sme_auprc = metrics.average_precision_score(y_test, y_pred_clf_sme, average='macro')
        
        clf_sme_f1 = metrics.f1_score(y_test, clf_sme.predict(x_test), average='macro')
        clf_sme_gm = geometric_mean_score(y_test, clf_sme.predict(x_test), average='macro')
        
        sme_auc_list.append(clf_sme_auc)
        sme_auprc_list.append(clf_sme_auprc)
        sme_f1_list.append(clf_sme_f1)
        sme_gm_list.append(clf_sme_gm)
        
        ksmote = ksm = KMeansSMOTE(kmeans_estimator=MiniBatchKMeans(n_init=1, random_state=1), 
                  random_state=seed,
                  cluster_balance_threshold=0.05, 
                  sampling_strategy={1: round(n_labels[0] * p)})
        X_res, y_res = ksm.fit_resample(x_train, y_train)

        clf_ksmote =  RandomForestClassifier(
                n_estimators=10, 
                max_depth=6, 
                criterion='gini', 
                random_state=seed
            )

        clf_ksmote.fit(X_res, y_res)

        y_pred_clf_ksmote = clf_ksmote.predict_proba(x_test)[:,1]

        clf_ksmote_auc = metrics.roc_auc_score(y_test, y_pred_clf_ksmote)
        clf_ksmote_auprc = metrics.average_precision_score(y_test, y_pred_clf_ksmote, average='macro')
        
        clf_ksmote_f1 = metrics.f1_score(y_test, clf_ksmote.predict(x_test), average='macro')
        clf_ksmote_gm = geometric_mean_score(y_test, clf_ksmote.predict(x_test), average='macro')
        
        ksmote_auc_list.append(clf_ksmote_auc)
        ksmote_auprc_list.append(clf_ksmote_auprc)
        ksmote_f1_list.append(clf_ksmote_f1)
        ksmote_gm_list.append(clf_ksmote_gm)

        clf_baseline =  RandomForestClassifier(
                n_estimators=10, 
                max_depth=6, 
                criterion='gini', 
                random_state=seed
            )

        clf_baseline.fit(x_train, y_train)
        y_pred_clf_baseline = clf_baseline.predict_proba(x_test)[:,1]

        clf_base_auc = metrics.roc_auc_score(y_test, y_pred_clf_baseline)
        clf_base_auprc = metrics.average_precision_score(y_test, y_pred_clf_baseline, average='macro')
        
        clf_base_f1 = metrics.f1_score(y_test, clf_baseline.predict(x_test), average='macro')
        clf_base_gm = geometric_mean_score(y_test, clf_baseline.predict(x_test), average='macro')
            
        base_auc_list.append(clf_base_auc)
        base_auprc_list.append(clf_base_auprc)
        base_f1_list.append(clf_base_f1)
        base_gm_list.append(clf_base_gm)

    target_num_samples = round(p * n_labels[0] - n_labels[1])

    for seed in np.arange(10):
        
        x_oversampled = cvae_sampling(cvae, target_num_samples, config, dataloader_for_benchmark, seed)

        clf_cvae =  RandomForestClassifier(
                n_estimators=10, 
                max_depth=6, 
                criterion='gini', 
                random_state=seed
            )

        clf_cvae.fit(np.concatenate([x_train, x_oversampled], axis=0),
                    np.concatenate([y_train, np.ones((target_num_samples))], axis=0))

        y_pred_clf_oversample = clf_cvae.predict_proba(x_test)[:,1]

        clf_cvae_auc = metrics.roc_auc_score(y_test, y_pred_clf_oversample)
        clf_cvae_auprc = metrics.average_precision_score(y_test, y_pred_clf_oversample, average='macro')
        
        clf_cvae_f1 = metrics.f1_score(y_test, clf_cvae.predict(x_test), average='macro')
        clf_cvae_gm = geometric_mean_score(y_test, clf_cvae.predict(x_test), average='macro')
        
        cvae_auc_list.append(clf_cvae_auc)
        cvae_auprc_list.append(clf_cvae_auprc)
        cvae_f1_list.append(clf_cvae_f1)
        cvae_gm_list.append(clf_cvae_gm)
        
        x_oversampled = ddhs_sampling(ddhs, target_num_samples, config, dataloader_for_benchmark, seed)

        clf_ddhs =  RandomForestClassifier(
                n_estimators=10, 
                max_depth=6, 
                criterion='gini', 
                random_state=seed
            )

        clf_ddhs.fit(np.concatenate([x_train, x_oversampled], axis=0),
                    np.concatenate([y_train, np.ones((target_num_samples))], axis=0))

        y_pred_clf_oversample = clf_ddhs.predict_proba(x_test)[:,1]

        clf_ddhs_auc = metrics.roc_auc_score(y_test, y_pred_clf_oversample)
        clf_ddhs_auprc = metrics.average_precision_score(y_test, y_pred_clf_oversample, average='macro')
        
        clf_ddhs_f1 = metrics.f1_score(y_test, clf_ddhs.predict(x_test), average='macro')
        clf_ddhs_gm = geometric_mean_score(y_test, clf_ddhs.predict(x_test), average='macro')
        
        ddhs_auc_list.append(clf_ddhs_auc)
        ddhs_auprc_list.append(clf_ddhs_auprc)
        ddhs_f1_list.append(clf_ddhs_f1)
        ddhs_gm_list.append(clf_ddhs_gm)
        
        x_oversampled = dfbs_sampling(dfbs, target_num_samples, config, dataloader_for_benchmark, seed)

        clf_dfbs =  RandomForestClassifier(
                n_estimators=10, 
                max_depth=6, 
                criterion='gini', 
                random_state=seed
            )

        clf_dfbs.fit(np.concatenate([x_train, x_oversampled], axis=0),
                    np.concatenate([y_train, np.ones((target_num_samples))], axis=0))

        y_pred_clf_oversample = clf_dfbs.predict_proba(x_test)[:,1]

        clf_dfbs_auc = metrics.roc_auc_score(y_test, y_pred_clf_oversample)
        clf_dfbs_auprc = metrics.average_precision_score(y_test, y_pred_clf_oversample, average='macro')
        
        clf_dfbs_f1 = metrics.f1_score(y_test, clf_dfbs.predict(x_test), average='macro')
        clf_dfbs_gm = geometric_mean_score(y_test, clf_dfbs.predict(x_test), average='macro')
        
        dfbs_auc_list.append(clf_dfbs_auc)
        dfbs_auprc_list.append(clf_dfbs_auprc)
        dfbs_f1_list.append(clf_dfbs_f1)
        dfbs_gm_list.append(clf_dfbs_gm)

        x_oversampled = dsmote_sampling(dsmote, target_num_samples, config, dataloader_for_benchmark, seed)

        clf_dsmote =  RandomForestClassifier(
                n_estimators=10, 
                max_depth=6, 
                criterion='gini', 
                random_state=seed
            )
        
        clf_dsmote.fit(np.concatenate([x_train, x_oversampled], axis=0),
                    np.concatenate([y_train, np.ones((target_num_samples))], axis=0))

        y_pred_clf_oversample = clf_dsmote.predict_proba(x_test)[:,1]

        clf_dsmote_auc = metrics.roc_auc_score(y_test, y_pred_clf_oversample)
        clf_dsmote_auprc = metrics.average_precision_score(y_test, y_pred_clf_oversample, average='macro')

        clf_dsmote_f1 = metrics.f1_score(y_test, clf_dsmote.predict(x_test), average='macro')
        clf_dsmote_gm = geometric_mean_score(y_test, clf_dsmote.predict(x_test), average='macro')

        dsmote_auc_list.append(clf_dsmote_auc)
        dsmote_auprc_list.append(clf_dsmote_auprc)
        dsmote_f1_list.append(clf_dsmote_f1)
        dsmote_gm_list.append(clf_dsmote_gm)

print("BASE AUPRC MEAN: {}, STD: {}".format(round(np.mean(base_auprc_list), 3), round(np.std(base_auprc_list), 3)))
print("BASE AUC MEAN: {}, STD: {}".format(round(np.mean(base_auc_list), 3), round(np.std(base_auc_list), 3)))
print("BASE F1 MEAN: {}, STD: {}".format(round(np.mean(base_f1_list), 3), round(np.std(base_f1_list), 3)))
print("BASE GMEAN MEAN: {}, STD: {}".format(round(np.mean(base_gm_list), 3), round(np.std(base_gm_list), 3)))

print("SMOTE AUPRC MEAN: {}, STD: {}".format(round(np.mean(smote_auprc_list), 3), round(np.std(smote_auprc_list), 3)))
print("SMOTE AUC MEAN: {}, STD: {}".format(round(np.mean(smote_auc_list), 3), round(np.std(smote_auc_list), 3)))
print("SMOTE F1 MEAN: {}, STD: {}".format(round(np.mean(smote_f1_list), 3), round(np.std(smote_f1_list), 3)))
print("SMOTE GMEAN MEAN: {}, STD: {}".format(round(np.mean(smote_gm_list), 3), round(np.std(smote_gm_list), 3)))

print("BSMOTE AUPRC MEAN: {}, STD: {}".format(round(np.mean(bsmote_auprc_list), 3), round(np.std(bsmote_auprc_list), 3)))
print("BSMOTE AUC MEAN: {}, STD: {}".format(round(np.mean(bsmote_auc_list), 3), round(np.std(bsmote_auc_list), 3)))
print("BSMOTE F1 MEAN: {}, STD: {}".format(round(np.mean(bsmote_f1_list), 3), round(np.std(bsmote_f1_list), 3)))
print("BSMOTE GMEAN MEAN: {}, STD: {}".format(round(np.mean(bsmote_gm_list), 3), round(np.std(bsmote_gm_list), 3)))

print("SMOTE_ENN AUPRC MEAN: {}, STD: {}".format(round(np.mean(sme_auprc_list), 3), round(np.std(sme_auprc_list), 3)))
print("SMOTE_ENN AUC MEAN: {}, STD: {}".format(round(np.mean(sme_auc_list), 3), round(np.std(sme_auc_list), 3)))
print("SMOTE_ENN F1 MEAN: {}, STD: {}".format(round(np.mean(sme_f1_list), 3), round(np.std(sme_f1_list), 3)))
print("SMOTE_ENN GMEAN MEAN: {}, STD: {}".format(round(np.mean(sme_gm_list), 3), round(np.std(sme_gm_list), 3)))

print("KMEANS_SMOTE AUPRC MEAN: {}, STD: {}".format(round(np.mean(ksmote_auprc_list), 3), round(np.std(ksmote_auprc_list), 3)))
print("KMEANS_SMOTE AUC MEAN: {}, STD: {}".format(round(np.mean(ksmote_auc_list), 3), round(np.std(ksmote_auc_list), 3)))
print("KMEANS_SMOTE F1 MEAN: {}, STD: {}".format(round(np.mean(ksmote_f1_list), 3), round(np.std(ksmote_f1_list), 3)))
print("KMEANS_SMOTE GMEAN MEAN: {}, STD: {}".format(round(np.mean(ksmote_gm_list), 3), round(np.std(ksmote_gm_list), 3)))

print("DFBS AUPRC MEAN: {}, STD: {}".format(round(np.mean(dfbs_auprc_list), 3), round(np.std(dfbs_auprc_list), 3)))
print("DFBS AUC MEAN: {}, STD: {}".format(round(np.mean(dfbs_auc_list), 3), round(np.std(dfbs_auc_list), 3)))
print("DFBS F1 MEAN: {}, STD: {}".format(round(np.mean(dfbs_f1_list), 3), round(np.std(dfbs_f1_list), 3)))
print("DFBS GMEAN MEAN: {}, STD: {}".format(round(np.mean(dfbs_gm_list), 3), round(np.std(dfbs_gm_list), 3)))

print("CVAE AUPRC MEAN: {}, STD: {}".format(round(np.mean(cvae_auprc_list), 3), round(np.std(cvae_auprc_list), 3)))
print("CVAE AUC MEAN: {}, STD: {}".format(round(np.mean(cvae_auc_list), 3), round(np.std(cvae_auc_list), 3)))
print("CVAE F1 MEAN: {}, STD: {}".format(round(np.mean(cvae_f1_list), 3), round(np.std(cvae_f1_list), 3)))
print("CVAE GMEAN MEAN: {}, STD: {}".format(round(np.mean(cvae_gm_list), 3), round(np.std(cvae_gm_list), 3)))

print("DEEP_SMOTE AUPRC MEAN: {}, STD: {}".format(round(np.mean(dsmote_auprc_list), 3), round(np.std(dsmote_auprc_list), 3)))
print("DEEP_SMOTE AUC MEAN: {}, STD: {}".format(round(np.mean(dsmote_auc_list), 3), round(np.std(dsmote_auc_list), 3)))
print("DEEP_SMOTE F1 MEAN: {}, STD: {}".format(round(np.mean(dsmote_f1_list), 3), round(np.std(dsmote_f1_list), 3)))
print("DEEP_SMOTE GMEAN MEAN: {}, STD: {}".format(round(np.mean(dsmote_gm_list), 3), round(np.std(dsmote_gm_list), 3)))

print("DDHS AUPRC MEAN: {}, STD: {}".format(round(np.mean(ddhs_auprc_list), 3), round(np.std(ddhs_auprc_list), 3)))
print("DDHS AUC MEAN: {}, STD: {}".format(round(np.mean(ddhs_auc_list), 3), round(np.std(ddhs_auc_list), 3)))
print("DDHS F1 MEAN: {}, STD: {}".format(round(np.mean(ddhs_f1_list), 3), round(np.std(ddhs_f1_list), 3)))
print("DDHS GMEAN MEAN: {}, STD: {}".format(round(np.mean(ddhs_gm_list), 3), round(np.std(ddhs_gm_list), 3)))

print("SMOTE_CLS AUPRC MEAN: {}, STD: {}".format(round(np.mean(smote_cls_auprc_list), 3), round(np.std(smote_cls_auprc_list), 3)))
print("SMOTE_CLS AUC MEAN: {}, STD: {}".format(round(np.mean(smote_cls_auc_list), 3), round(np.std(smote_cls_auc_list), 3)))
print("SMOTE_CLS F1 MEAN: {}, STD: {}".format(round(np.mean(smote_cls_f1_list), 3), round(np.std(smote_cls_f1_list), 3)))
print("SMOTE_CLS GMEAN MEAN: {}, STD: {}".format(round(np.mean(smote_cls_gm_list), 3), round(np.std(smote_cls_gm_list), 3)))