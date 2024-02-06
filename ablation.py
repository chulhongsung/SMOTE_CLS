from tqdm import tqdm
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib as mpl
from ing_theme_matplotlib import mpl_style
mpl_style(False)
mpl.rcParams["figure.figsize"] = (6,6)
plt.rcParams['figure.dpi'] = 200
import torch
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

from xgboost.sklearn import XGBClassifier

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, KernelDensity

from model import * 
from utils import *
from train import * 

import seaborn as sns

np.random.seed(10)
s1 = 80
s2 = 20
n_noise = 50
minor1 = np.random.multivariate_normal(mean=(-0.3, 0), cov=[[0.01, 0.0], [0.0, 0.01]], size=s1)
minor2 = np.random.multivariate_normal(mean=(0.3, 0), cov=[[0.01, 0.0], [0.0, 0.01]], size=s2)
major = np.random.random((1500, 2)) * 2 - 1
major = major[((major[:, 0] + 0.3) ** 2 + major[:, 1] ** 2 >= 0.008) & ((major[:, 0] - 0.3) ** 2 + major[:, 1] ** 2 >= 0.008)]

tmp_df = pd.DataFrame(np.concatenate([major, minor1, minor2], axis=0), columns=["x1", "x2"])
tmp_df["label"] = np.concatenate((np.zeros(major.shape[0], dtype=np.int32), np.ones(s1 + s2, dtype=np.int32)), axis=0)

noise_idx = np.random.choice(list(range(len(major))), n_noise)
tmp_df.iloc[noise_idx, -1] = 1

sns.scatterplot(data=tmp_df, x="x1", y="x2", hue="label")
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])
plt.yticks(np.linspace(-1.0, 1.0, 5))


minor_data = pd.DataFrame(np.concatenate([minor1, minor2], axis=0), columns=["x1", "x2"])
g=sns.jointplot(data=tmp_df, x="x1", y="x2", kind="kde", hue="label", palette={0: "#66b2ff", 1:"#ff8000", 2:"k"}, threshold=0.1, marginal_ticks=False, alpha=0.9, levels=10, label="Density")
g.ax_marg_x.remove()
g.ax_marg_y.remove()
g.set_axis_labels("","")
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])
mark1 = mlines.Line2D([], [], c="#66b2ff", lw=2,label="Major")
mark2 = mlines.Line2D([], [], c="#ff8000", lw=2,label="Minor")

plt.legend(handles=[mark1, mark2], fontsize="15", loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=2, fancybox=True, shadow=True)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(tmp_df.iloc[:, :-1], tmp_df["label"])
y_pred = neigh.predict(tmp_df.iloc[:, :-1])
confusion_matrix(tmp_df["label"], y_pred)

y_tmp = tmp_df["label"].copy()
y_tmp[(tmp_df["label"] == 0) & (y_pred == 1)] = 2
y_tmp[(tmp_df["label"] == 1) & (y_pred == 0)] = 3

confusion_matrix(y_tmp, y_pred)

clf2 = XGBClassifier(
        n_estimators=30, 
        max_depth=15, 
        gamma = 0.3, 
        importance_type='gain', 
        reg_lambda = 0.1, 
        random_state=0
    )

clf2.fit(tmp_df.iloc[:, :-1], y_tmp)
y_pred = clf2.predict(tmp_df.iloc[:, :-1])
confusion_matrix(y_tmp, y_pred)

tmp_df["knn_label"] = y_tmp
tmp_df["true_label"] = tmp_df["label"]
tmp_df.iloc[noise_idx, -1] = 2

g=sns.scatterplot(data=tmp_df,
                  x="x1", y="x2",
                  hue="true_label",
                  s=60, alpha=0.8,
                  palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})

plt.xlabel("")
plt.ylabel("")
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])

mark1 = mlines.Line2D([], [], c="#66b2ff", lw=0, marker='o', markersize=8, label="Major")
mark2 = mlines.Line2D([], [], c="#ff8000", lw=0, marker='o', markersize=8, label="Minor")
mark3 = mlines.Line2D([], [], c="k",  lw=0, marker='o', markersize=8, label="Noise")
plt.legend('',frameon=False) 

def smote_cls_mlpc_train(model, train_loader, optimizer, config, device):
    model.train()
    
    '''prior design'''
    prior_means = np.zeros((4, config['latent_dim']))
    prior_means[0, 0] = -1 * config['dist']
    prior_means[0, 1] = 1 * config['dist']
    prior_means[1, 0] = 1 * config['dist']
    prior_means[1, 1] = -1 * config['dist']
    prior_means[2, 0] = 1 * config['dist']
    prior_means[2, 1] = 1 * config['dist']
    prior_means[3, 0] = -1 * config['dist']
    prior_means[3, 1] = -1 * config['dist']
    prior_means = torch.tensor(prior_means[np.newaxis, :, :], dtype=torch.float32).to(device)

    sigma_vector = np.ones((1, config['latent_dim'])) 
    sigma_vector[0, :config['latent_dim']] = config['sigma1']
    sigma_vector[0, config['latent_dim']:] = config['sigma2']
    sigma_vector = torch.tensor(sigma_vector, dtype=torch.float32).to(device)
    
    
    logs = {
        "elbo": [],
        "recon": [],
        "kl": [],
        "cce": [],
    }
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        loss_ = []

        optimizer.zero_grad()

        '''ELBO'''
        mean, logvar, probs, y, z, z_tilde, xhat = model(batch_x)
        
        recon = torch.abs(batch_x - xhat).sum()
        loss_.append(('recon', recon))
        
        kl1 = torch.log(probs) + torch.log(torch.tensor(2))
        kl1 = (probs * kl1).sum(axis=1).sum()

        kl2 = torch.pow(mean - prior_means, 2) / sigma_vector
        kl2 -= 1
        kl2 += torch.log(sigma_vector)
        kl2 += torch.exp(logvar) / sigma_vector
        kl2 -= logvar
        kl2 = probs * (0.5 * kl2).sum(axis=-1)
        kl2 = kl2.sum()

        probL = model.classify(batch_x)
        cce = F.nll_loss(torch.log(probL), batch_y.squeeze().type(torch.long),
                        reduction='none').sum()

        elbo = recon + 30 * cce + config["beta"] * (kl1 + kl2)

        loss_.append(('elbo', elbo))
        loss_.append(('recon', recon))
        loss_.append(('kl', kl1 + kl2))
        loss_.append(('cce', cce))

        # encoder and decoder
        elbo.backward()
        optimizer.step()
        
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
        
    return logs
# %%
config = {
    "dataset": "simul",
    "latent_dim": 2,
    "input_dim" : 2,
    "batch_size": 512,
    "epochs": 300,
    "recon": "l1",
    "activation": "identity",
    "beta": 1.0,
    "sigma1" : 0.1,
    "sigma2" : 1.0,
    "dist": 1.0,
} 

config["method"] = "smote_cls"     
config["classifier"] = "rf"
#%%
dataset = TensorDataset(torch.from_numpy(tmp_df.iloc[:,:2].values.astype('float32')), torch.Tensor(y_tmp.values[:, np.newaxis]))
dataloader = DataLoader(dataset, batch_size=512, shuffle=False, drop_last=False)
#%%
smote_cls_mlpc = MixtureMLPVAE(config, class_num=4, classifier=None)

"""optimizer"""
optimizer = torch.optim.Adam(
        list(smote_cls_mlpc.encoder.parameters()) + list(smote_cls_mlpc.decoder.parameters())+ list(smote_cls_mlpc.classifier.parameters()), 
        lr=0.001
    )
config["beta"] = 0.1

pbar = tqdm(range(400))

for epoch in pbar:
    logs = smote_cls_mlpc_train(smote_cls_mlpc, dataloader, optimizer, config, device)
    pbar.set_description('====> Seed: {} Epoch: {} ELBO: {:.4f} KL {:.4f} Recon {:.4f} CCE {:.4f}'.format(
        0, epoch, logs['elbo'][0], logs['kl'][0], logs['recon'][0], logs['cce'][0] ))
# %%
samples = []
true_y = []
z_tildes = []
predicted_y = []
xhats = []

for batch_x, batch_y in dataloader:
    mean, logvar, probs, y_, z, z_tilde, xhat = smote_cls_mlpc(batch_x, sampling=False)
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
#%%
z = pd.DataFrame(z_tildes[:, :2], columns=["Z1", "Z2"])
z["label"] = tmp_df["label"]
z["true_label"] = tmp_df["true_label"]
g=sns.jointplot(data=z, x="Z1", y="Z2", kind="kde", hue="label", marginal_ticks=False, thresh=0.05, alpha=0.5, levels=10, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"}, label="Density")
g.ax_marg_x.remove()
g.ax_marg_y.remove()
g.set_axis_labels("","")
g1=sns.scatterplot(x="Z1", y="Z2", hue="true_label", data=z, s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
g1.add_patch(
    patches.Rectangle((-2, 0), 2, 2,  alpha=0.15, color="#66b2ff", label="Easy Major"))
g1.add_patch(
    patches.Rectangle((0, 0), 2, 2,  alpha=0.15, color="#7f00ff", label="Hard Major"))
g1.add_patch(
    patches.Rectangle((-2, -2), 2, 2,  alpha=0.15, color="#ff6666", label="Hard Minor"))
g1.add_patch(
    patches.Rectangle((0, -2), 2, 2,  alpha=0.15, color="#ff8000", label="Easy Minor"))
plt.xlim([-3.0, 3.0])  
plt.ylim([-3.0, 3.0])
plt.text(-2.3, 2.1, 'Easy Major', fontsize=15)
plt.text(1.0, 2.1, 'Hard Major', fontsize=15)
plt.text(-2.3, -2.3, 'Hard Minor', fontsize=15)
plt.text(1.0, -2.3, 'Easy Minor', fontsize=15)
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
#%%
z_train = z_tildes.detach().numpy()
y_train = true_y.detach().numpy()

z_train_easy_minor = z_train[y_train == 1, :]
z_train_hard_minor = z_train[y_train == 3, :]

z_train_major = z_train[y_train == 0, :]

minor_easy_kde = KernelDensity(kernel='gaussian').fit(z_train_easy_minor)
minor_hard_kde = KernelDensity(kernel='gaussian').fit(z_train_hard_minor)

minor_easy_score = minor_easy_kde.score_samples(z_train_easy_minor)
minor_hard_score = minor_hard_kde.score_samples(z_train_hard_minor)

minor_easy_threshold = np.quantile(minor_easy_score, 0.2)
minor_hard_threshold = np.quantile(minor_hard_score, 0.6)

z_train_filtered_easy_minor = z.loc[y_train == 1, :].loc[minor_easy_score >= minor_easy_threshold, :]
z_train_filtered_hard_minor = z.loc[y_train == 3, :].loc[minor_hard_score > minor_hard_threshold, :]
#%%
g1=sns.scatterplot(x="Z1", y="Z2", data=z.loc[(y_train == 0) | (y_train == 2), :], hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
sns.scatterplot(x="Z1", y="Z2", data=z_train_filtered_easy_minor, hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
sns.scatterplot(x="Z1", y="Z2", data=z_train_filtered_hard_minor, hue="true_label", s=40, alpha=0.8, palette={1: "#ff8000", 2:"k"})

g1.add_patch(
    patches.Rectangle((-2, 0), 2, 2,  alpha=0.15, color="#66b2ff", label="Easy Major"))
g1.add_patch(
    patches.Rectangle((0, 0), 2, 2,  alpha=0.15, color="#7f00ff", label="Hard Major"))
g1.add_patch(
    patches.Rectangle((-2, -2), 2, 2,  alpha=0.15, color="#ff6666", label="Hard Minor"))
g1.add_patch(
    patches.Rectangle((0, -2), 2, 2,  alpha=0.15, color="#ff8000", label="Easy Minor"))
plt.xlim([-3.0, 3.0])  
plt.ylim([-3.0, 3.0])
plt.text(-2.3, 2.1, 'Easy Major', fontsize=15)
plt.text(1.0, 2.1, 'Hard Major', fontsize=15)
plt.text(-2.3, -2.3, 'Hard Minor', fontsize=15)
plt.text(1.0, -2.3, 'Easy Minor', fontsize=15)
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
#%%
x_train_filtered_easy_minor = tmp_df.loc[tmp_df["knn_label"] == 1, :].loc[minor_easy_score >= minor_easy_threshold, :]
x_train_filtered_hard_minor = tmp_df.loc[tmp_df["knn_label"] == 3, :].loc[minor_hard_score > minor_hard_threshold, :]
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])
sns.scatterplot(x="x1", y="x2", data=x_train_filtered_easy_minor, hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
sns.scatterplot(x="x1", y="x2", data=x_train_filtered_hard_minor, hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
#%%
dataset_for_benchmark = TensorDataset(torch.from_numpy(tmp_df.iloc[:, :2].values.astype('float32')), torch.Tensor(tmp_df["label"].values[:, np.newaxis]))
dataloader_for_benchmark = DataLoader(dataset_for_benchmark, batch_size=512, shuffle=False, drop_last=False)

def smote_cls_nosplit_train(model, train_loader, optimizer, config, device):
    model.train()
    
    '''prior design'''
    prior_means = np.zeros((2, config['latent_dim']))
    prior_means[0, 0] = 1 * config['dist']
    prior_means[0, 1] = 1 * config['dist']
    prior_means[1, 0] = -1 * config['dist']
    prior_means[1, 1] = -1 * config['dist']
    prior_means = torch.tensor(prior_means[np.newaxis, :, :], dtype=torch.float32).to(device)

    sigma_vector = np.ones((1, config['latent_dim'])) 
    sigma_vector[0, :config['latent_dim']] = 0.1
    sigma_vector[0, config['latent_dim']:] = config['sigma2']
    sigma_vector = torch.tensor(sigma_vector, dtype=torch.float32).to(device)
    
    
    logs = {
        "elbo": [],
        "recon": [],
        "kl": [],
        "cce": [],
    }
    
    # output_info_list = config["transformer"].output_info_list
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        loss_ = []

        optimizer.zero_grad()

        '''ELBO'''
        mean, logvar, probs, y, z, z_tilde, xhat = model(batch_x)
        
        recon = torch.abs(batch_x - xhat).sum()
        loss_.append(('recon', recon))
        
        kl1 = torch.log(probs) + torch.log(torch.tensor(2))
        kl1 = (probs * kl1).sum(axis=1).sum()

        kl2 = torch.pow(mean - prior_means, 2) / sigma_vector
        kl2 -= 1
        kl2 += torch.log(sigma_vector)
        kl2 += torch.exp(logvar) / sigma_vector
        kl2 -= logvar
        kl2 = probs * (0.5 * kl2).sum(axis=-1)
        kl2 = kl2.sum()

        probL = model.classify(batch_x)
        cce = F.nll_loss(torch.log(probL), batch_y.squeeze().type(torch.long),
                        reduction='none').sum()

        elbo = recon + config["beta"] * (kl1 + kl2)

        loss_.append(('elbo', elbo))
        loss_.append(('recon', recon))
        loss_.append(('kl', kl1 + kl2))
        loss_.append(('cce', cce))

        # encoder and decoder
        elbo.backward()
        optimizer.step()
        
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
        
    return logs


clf2 = XGBClassifier(
        n_estimators=15, 
        max_depth=8, 
        gamma = 0.3, 
        importance_type='gain', 
        reg_lambda = 0.1, 
        # subsample=0.8, 
        # scale_pos_weigh = scale_pos_weight,
        random_state=0
    )

clf2.fit(tmp_df.iloc[:, :2].values, tmp_df["label"].values)
y_pred = clf2.predict(tmp_df.iloc[:, :2].values)
confusion_matrix(tmp_df["label"], y_pred)
#%%
no_split_model = MixtureMLPVAE(config, class_num=2, classifier=clf2)
# %%
"""optimizer"""
optimizer = torch.optim.Adam(
        list(no_split_model.encoder.parameters()) + list(no_split_model.decoder.parameters()), 
        lr=0.001
    )
config["beta"] = 0.1

pbar = tqdm(range(400))

for epoch in pbar:
    logs = smote_cls_nosplit_train(no_split_model, dataloader_for_benchmark, optimizer, config, device)
    pbar.set_description('====> Seed: {} Epoch: {} ELBO: {:.4f} KL {:.4f} Recon {:.4f} CCE {:.4f}'.format(
        0, epoch, logs['elbo'][0], logs['kl'][0], logs['recon'][0], logs['cce'][0] ))
# %%
samples = []
true_y = []
z_tildes = []
predicted_y = []
xhats = []

for batch_x, batch_y in dataloader_for_benchmark:
    mean, logvar, probs, y_, z, z_tilde, xhat = no_split_model(batch_x, sampling=False)
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
#%%
mpl.rcParams["figure.figsize"] = (6,6)
plt.rcParams['figure.dpi'] = 200
#%%
plt.scatter(samples[:, 0, 0], samples[:, 0, 1], alpha=0.3, c="#66b2ff", label="Major")
plt.scatter(samples[:, 1, 0], samples[:, 1, 1], alpha=0.3, c="#ff8000", label="Minor")
plt.legend(fontsize=16)
plt.legend('',frameon=False)
# %%
mpl.rcParams["figure.figsize"] = (6,6)
plt.rcParams['figure.dpi'] = 200
z = pd.DataFrame(z_tildes[:, :2], columns=["Z1", "Z2"])
z["label"] = tmp_df["true_label"]
g=sns.jointplot(data=z, x="Z1", y="Z2", kind="kde", hue="label", marginal_ticks=False, thresh=0.05, alpha=0.5, levels=10, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"}, label="Density")
g.ax_marg_x.remove()
g.ax_marg_y.remove()
g.set_axis_labels("","")
g1=sns.scatterplot(x="Z1", y="Z2", hue="label", data=z, s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
g1.add_patch(
    patches.Rectangle((0, 0), 2, 2,  alpha=0.15, color="#66b2ff", label="Major"))
g1.add_patch(
    patches.Rectangle((-2, -2), 2, 2,  alpha=0.15, color="#ff8000", label="Minor"))
plt.xlim([-3.0, 3.0])  
plt.ylim([-3.0, 3.0])
plt.text(1.3, 2.1, 'Major', fontsize=15)
plt.text(-2., -2.3, 'Minor', fontsize=15)
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
# %%
z_train = z_tildes.detach().numpy()
y_train = true_y.detach().numpy()

z_train_minor = z_train[(y_train == 3) | (y_train == 1), :]

minor_kde = KernelDensity(kernel='gaussian').fit(z_train_minor)

minor_score = minor_kde.score_samples(z_train_minor)
treshold = np.quantile(minor_score, 0.35)
z_train_filtered_minor = z_train_minor[minor_score > treshold, :]
#%%
g1=sns.scatterplot(x="Z1", y="Z2", data=z.loc[(y_train == 0) | (y_train == 2), :], hue="label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
sns.scatterplot(x="Z1", y="Z2", data=z.loc[(y_train == 1) | (y_train == 3), :].loc[minor_score > treshold, :], hue="label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlim([-3.0, 3.0])  
plt.ylim([-3.0, 3.0])
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
# %%
x_train_filtered_minor = tmp_df.loc[(y_train == 1) | (y_train == 3), :].loc[minor_score >= treshold, :]
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])
sns.scatterplot(x="x1", y="x2", data=x_train_filtered_minor, hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlabel("")
plt.ylabel("")

plt.legend('',frameon=False)
# %% Selection not Filtering
def smote_cls_train(model, train_loader, optimizer, config, device):
    model.train()
    
    '''prior design'''
    prior_means = np.zeros((4, config['latent_dim']))
    prior_means[0, 0] = -1 * config['dist']
    prior_means[0, 1] = 1 * config['dist']
    prior_means[1, 0] = 1 * config['dist']
    prior_means[1, 1] = -1 * config['dist']
    prior_means[2, 0] = 1 * config['dist']
    prior_means[2, 1] = 1 * config['dist']
    prior_means[3, 0] = -1 * config['dist']
    prior_means[3, 1] = -1 * config['dist']
    prior_means = torch.tensor(prior_means[np.newaxis, :, :], dtype=torch.float32).to(device)

    sigma_vector = np.ones((1, config['latent_dim'])) 
    sigma_vector[0, :config['latent_dim']] = config['sigma1']
    sigma_vector[0, config['latent_dim']:] = config['sigma2']
    sigma_vector = torch.tensor(sigma_vector, dtype=torch.float32).to(device)
    
    
    logs = {
        "elbo": [],
        "recon": [],
        "kl": [],
        "cce": [],
    }
    
    # output_info_list = config["transformer"].output_info_list
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        loss_ = []

        optimizer.zero_grad()

        '''ELBO'''
        mean, logvar, probs, y, z, z_tilde, xhat = model(batch_x)
        
        recon = torch.abs(batch_x - xhat).sum()
        loss_.append(('recon', recon))
        
        kl1 = torch.log(probs) + torch.log(torch.tensor(2))
        kl1 = (probs * kl1).sum(axis=1).sum()

        kl2 = torch.pow(mean - prior_means, 2) / sigma_vector
        kl2 -= 1
        kl2 += torch.log(sigma_vector)
        kl2 += torch.exp(logvar) / sigma_vector
        kl2 -= logvar
        kl2 = probs * (0.5 * kl2).sum(axis=-1)
        kl2 = kl2.sum()

        probL = model.classify(batch_x)
        cce = F.nll_loss(torch.log(probL), batch_y.squeeze().type(torch.long),
                        reduction='none').sum()

        elbo = recon + config["beta"] * (kl1 + kl2)

        loss_.append(('elbo', elbo))
        loss_.append(('recon', recon))
        loss_.append(('kl', kl1 + kl2))
        loss_.append(('cce', cce))

        # encoder and decoder
        elbo.backward()
        optimizer.step()
        
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
        
    return logs

config = {
    "dataset": "simul",
    "latent_dim": 2,
    "input_dim" : 2,
    "batch_size": 512,
    "epochs": 300,
    "recon": "l1",
    "activation": "identity",
    "beta": 1.0,
    "sigma1" : 0.1,
    "sigma2" : 1.0,
    "dist": 1.0,
} 
config["method"] = "smote_cls"     
config["classifier"] = "rf"
#%%
clf2 = XGBClassifier(
        n_estimators=30, 
        max_depth=15, 
        gamma = 0.3, 
        importance_type='gain', 
        reg_lambda = 0.1, 
        random_state=0
    )

clf2.fit(tmp_df.iloc[:, :2], y_tmp)
y_pred = clf2.predict(tmp_df.iloc[:, :-1])
confusion_matrix(y_tmp, y_pred)
#%%
dataset = TensorDataset(torch.from_numpy(tmp_df.iloc[:,:2].values.astype('float32')), torch.Tensor(y_tmp.values[:, np.newaxis]))
dataloader = DataLoader(dataset, batch_size=512, shuffle=False, drop_last=False)

config["clf"] = clf2
no_filter_model = MixtureMLPVAE(config, class_num=4, classifier=clf2)


"""optimizer"""
optimizer = torch.optim.Adam(
        list(no_filter_model.encoder.parameters()) + list(no_filter_model.decoder.parameters()), 
        lr=0.001
    )
config["beta"] = 0.1

pbar = tqdm(range(400))

for epoch in pbar:
    logs = smote_cls_train(no_filter_model, dataloader, optimizer, config, device)
    pbar.set_description('====> Seed: {} Epoch: {} ELBO: {:.4f} KL {:.4f} Recon {:.4f} CCE {:.4f}'.format(
        0, epoch, logs['elbo'][0], logs['kl'][0], logs['recon'][0], logs['cce'][0] ))
#%%
samples = []
true_y = []
z_tildes = []
predicted_y = []
xhats = []

for batch_x, batch_y in dataloader:
    mean, logvar, probs, y_, z, z_tilde, xhat = no_filter_model(batch_x, sampling=False)
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

plt.scatter(z_tildes[tmp_df["true_label"].values==0, 0], z_tildes[tmp_df["true_label"].values==0, 1], alpha=0.3, c="#66b2ff",  label="Major")
plt.scatter(z_tildes[tmp_df["true_label"].values==1, 0], z_tildes[tmp_df["true_label"].values==1, 1], alpha=0.3, c="#ff8000", label="Minor")
plt.scatter(z_tildes[tmp_df["true_label"].values==2, 0], z_tildes[tmp_df["true_label"].values==2, 1], alpha=0.3, c="k", label="Noise")
mark1 = mlines.Line2D([], [], c="#66b2ff", lw=0, marker='o', label="Major")
mark2 = mlines.Line2D([], [], c="#ff8000", lw=0, marker='o', label="Minor")
mark3 = mlines.Line2D([], [], c="k", lw=0, marker='o', label="Noise")
plt.legend(handles=[mark1, mark2, mark3], loc="upper right")
#%%
z = pd.DataFrame(z_tildes[:, :2], columns=["Z1", "Z2"])
z["label"] = tmp_df["true_label"]
g=sns.jointplot(data=z, x="Z1", y="Z2", kind="kde", hue="label", marginal_ticks=False, thresh=0.05, alpha=0.5, levels=10, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"}, label="Density")
g.ax_marg_x.remove()
g.ax_marg_y.remove()
g.set_axis_labels("","")
g1=sns.scatterplot(x="Z1", y="Z2", hue="label", data=z, s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})

g1.add_patch(
    patches.Rectangle((-2, 0), 2, 2,  alpha=0.15, color="#66b2ff", label="Easy Major"))
g1.add_patch(
    patches.Rectangle((0, 0), 2, 2,  alpha=0.15, color="#7f00ff", label="Hard Major"))
g1.add_patch(
    patches.Rectangle((-2, -2), 2, 2,  alpha=0.15, color="#ff6666", label="Hard Minor"))
g1.add_patch(
    patches.Rectangle((0, -2), 2, 2,  alpha=0.15, color="#ff8000", label="Easy Minor"))
plt.xlim([-3.0, 3.0])  
plt.ylim([-3.0, 3.0])
plt.text(-2.3, 2.1, 'Easy Major', fontsize=15)
plt.text(1.0, 2.1, 'Hard Major', fontsize=15)
plt.text(-2.3, -2.3, 'Hard Minor', fontsize=15)
plt.text(1.0, -2.3, 'Easy Minor', fontsize=15)
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
#%% REAL METHOD 
z_train = z_tildes.detach().numpy()
y_train = true_y.detach().numpy()

z_train_minor = z_train[(y_train == 3) | (y_train == 1), :]

minor_kde = KernelDensity(kernel='gaussian').fit(z_train_minor)

minor_score = minor_kde.score_samples(z_train_minor)
treshold = np.quantile(minor_score, 0.0)
z_train_filtered_minor = z_train_minor[minor_score > treshold, :]
#%%
g1=sns.scatterplot(x="Z1", y="Z2", data=z.loc[(y_train == 0) | (y_train == 2), :], hue="label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
sns.scatterplot(x="Z1", y="Z2", data=z.loc[(y_train == 1) | (y_train == 3), :].loc[minor_score > treshold, :], hue="label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
#sns.scatterplot(x="Z1", y="Z2", data=z_train_filtered_hard_minor, hue="label", s=40, alpha=0.8, palette={1: "#ff8000", 2:"k"})
plt.xlim([-3.0, 3.0])  
plt.ylim([-3.0, 3.0])
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
#%%
x_train_filtered_minor = tmp_df.loc[(y_train == 1) | (y_train == 3), :].loc[minor_score >= treshold, :]
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])
# g1=sns.scatterplot(x="x1", y="x2", data=tmp_df.loc[(y_train == 0) | (y_train == 2), :], hue="label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
sns.scatterplot(x="x1", y="x2", data=x_train_filtered_minor, hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
# sns.scatterplot(x="x1", y="x2", data=x_train_filtered_hard_minor, hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
# %%
####################################################################
####################################################################
############################# CVAE #################################
####################################################################
####################################################################

cvae = CVAE(config).to(device)

optimizer = torch.optim.Adam(cvae.parameters(), lr=0.01)

pbar = tqdm(range(500))

for epoch in pbar:
    logs = cvae_train(cvae, dataloader_for_benchmark, optimizer, device)

#%%
samples = []
true_y = []
z_tildes = []
predicted_y = []
xhats = []

for batch_x, batch_y in dataloader_for_benchmark:
    xhat, mean, logvar = cvae(batch_x, batch_y)
    samples.append(mean)
    true_y.append(batch_y)
    z_tildes.append(mean)
    xhats.append(xhat)

samples = torch.cat(samples, dim=0)
samples = samples.detach()
true_y = torch.cat(true_y, dim=0)
true_y = true_y.detach().squeeze()
z_tildes = torch.cat(z_tildes, dim=0)
z_tildes = z_tildes.detach().squeeze()
xhats = torch.cat(xhats, dim=0)
# %%
z = pd.DataFrame(z_tildes[:, :2], columns=["Z1", "Z2"])
z["label"] = tmp_df["true_label"]
g=sns.jointplot(data=z, x="Z1", y="Z2", kind="kde", hue="label", marginal_ticks=False, thresh=0.05, alpha=0.5, levels=10, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"}, label="Density")
g.ax_marg_x.remove()
g.ax_marg_y.remove()
g.set_axis_labels("","")
g1=sns.scatterplot(x="Z1", y="Z2", hue="label", data=z, s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlim([-3.0, 3.0])  
plt.ylim([-3.0, 3.0])
plt.xlabel("")
plt.ylabel("")
mark1 = mlines.Line2D([], [], c="#66b2ff", lw=0, marker='o', label="Major")
mark2 = mlines.Line2D([], [], c="#ff8000", lw=0, marker='o', label="Minor")
mark3 = mlines.Line2D([], [], c="k", lw=0, marker='o', label="Noise")
plt.legend(handles=[mark1, mark2, mark3], fontsize="15", loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
# %%
z_train = z_tildes.detach().numpy()
y_train = true_y.detach().numpy()
z_train_minor = z_train[y_train == 1, :]
minor_kde = KernelDensity(kernel='gaussian', bandwidth="scott").fit(z_train_minor)
minor_score = minor_kde.score_samples(z_train_minor)
treshold = np.quantile(minor_score, 0.4)
z_train_filtered_minor = z_train_minor[minor_score > treshold, :]
#%%
g1=sns.scatterplot(x="Z1", y="Z2", data=z.loc[(y_train == 0), :], hue="label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
sns.scatterplot(x="Z1", y="Z2", data=z.loc[(y_train == 1), :].loc[minor_score > treshold, :], hue="label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlim([-3.0, 3.0])  
plt.ylim([-3.0, 3.0])
plt.xlabel("")
plt.ylabel("")
mark1 = mlines.Line2D([], [], c="#66b2ff", lw=0, marker='o', label="Major")
mark2 = mlines.Line2D([], [], c="#ff8000", lw=0, marker='o', label="Minor")
mark3 = mlines.Line2D([], [], c="k", lw=0, marker='o', label="Noise")
plt.legend(handles=[mark1, mark2, mark3], fontsize="15", loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
# %%
x_train_filtered_minor = tmp_df.loc[(y_train == 1), :].loc[minor_score >= treshold, :]
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])
sns.scatterplot(x="x1", y="x2", data=x_train_filtered_minor, hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlabel("")
plt.ylabel("")

mark1 = mlines.Line2D([], [], c="#66b2ff", lw=0, marker='o', label="Major")
mark2 = mlines.Line2D([], [], c="#ff8000", lw=0, marker='o', label="Minor")
mark3 = mlines.Line2D([], [], c="k", lw=0, marker='o', label="Noise")
# mark4 = mlines.Line2D([], [], c="#ff6666", lw=0, marker='o', label="Hard Minor Label")
plt.legend(handles=[mark1, mark2, mark3], fontsize="15", loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)