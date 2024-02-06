from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib as mpl
from ing_theme_matplotlib import mpl_style
mpl_style(False)
mpl.rcParams["figure.figsize"] = (6,6)
plt.rcParams['figure.dpi'] = 200
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, DataLoader

from imblearn.over_sampling import SMOTE

from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.inspection import DecisionBoundaryDisplay

from model import * 
from utils import *
from train import * 

import seaborn as sns
#%%
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
#%%
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
#%% Naive Bayes Decision Boundary 
cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["#66b2ff","#ff8000"])

xlim = (-1, 1)
ylim = (-1, 1)
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500),
                     np.linspace(ylim[0], ylim[1], 500))
#%%
mpl.rcParams["figure.figsize"] = (6,6)
plt.rcParams['figure.dpi'] = 200
#%%
G1 = multivariate_normal(mean=(-0.3, 0), cov=[[0.01, 0.0], [0.0, 0.01]])
G2 = multivariate_normal(mean=(0.3, 0), cov=[[0.01, 0.0], [0.0, 0.01]])

px = 11/48 + (1/15) * G1.pdf(np.c_[xx.ravel(), yy.ravel()]) + (1/60) * G2.pdf(np.c_[xx.ravel(), yy.ravel()])
py_1_x = (11/48) / px
sum((1 - py_1_x) > 0.5)
bayes_classifier = np.where((1 - py_1_x) >= 0.5, 1, 0)

# display = DecisionBoundaryDisplay(xx0=xx, xx1=yy, response=np.reshape(1 - py_1_x, xx.shape))
display = DecisionBoundaryDisplay(xx0=xx, xx1=yy, response=np.reshape(bayes_classifier, xx.shape))
display.plot(alpha=0.95, cmap=cmap)
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])

display.ax_.scatter(tmp_df.loc[:, "x1"], tmp_df.loc[:, "x2"], c=tmp_df['label'],
                 s=40, alpha=0.3,
                 edgecolor="w",
                 cmap=cmap)

labels = ["Major", "Minor"]
colors = ["#66b2ff", "#ff8000"]
handles = [
    Patch(facecolor=color, label=label) 
    for label, color in zip(labels, colors)
]

plt.legend(handles=handles, fontsize="15", loc='upper center', bbox_to_anchor=(0.5, 1.05), 
           ncol=2, fancybox=True, shadow=True)
#%%
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(tmp_df.iloc[:, :-1], tmp_df["label"])
y_pred = neigh.predict(tmp_df.iloc[:, :-1])

y_tmp = tmp_df["label"].copy()
y_tmp[(tmp_df["label"] == 0) & (y_pred == 1)] = 2
y_tmp[(tmp_df["label"] == 1) & (y_pred == 0)] = 3

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
#%%
tmp_df["knn_label"] = y_tmp
tmp_df["true_label"] = tmp_df["label"]
tmp_df.iloc[noise_idx, -1] = 2
# %%
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

model = MixtureMLPVAE(config, class_num=4, classifier=clf2)
config["clf"] = clf2

"""optimizer"""
optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()), 
        lr=0.001
    )
config["beta"] = 0.1

pbar = tqdm(range(500))

for epoch in pbar:
    logs = smote_cls_train(model, dataloader, optimizer, config, device)
    pbar.set_description('====> Seed: {} Epoch: {} ELBO: {:.4f} KL {:.4f} Recon {:.4f} CCE {:.4f}'.format(
        0, epoch, logs['elbo'][0], logs['kl'][0], logs['recon'][0], logs['cce'][0] ))
#%%
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

plt.scatter(z_tildes[tmp_df["true_label"].values==0, 0], z_tildes[tmp_df["true_label"].values==0, 1], alpha=0.3, c="#66b2ff",  label="Major")
plt.scatter(z_tildes[tmp_df["true_label"].values==1, 0], z_tildes[tmp_df["true_label"].values==1, 1], alpha=0.3, c="#ff8000", label="Minor")
plt.scatter(z_tildes[tmp_df["true_label"].values==2, 0], z_tildes[tmp_df["true_label"].values==2, 1], alpha=0.3, c="k", label="Noise")
mark1 = mlines.Line2D([], [], c="#66b2ff", lw=0, marker='o', label="Major")
mark2 = mlines.Line2D([], [], c="#ff8000", lw=0, marker='o', label="Minor")
mark3 = mlines.Line2D([], [], c="k", lw=0, marker='o', label="Noise")
plt.legend(handles=[mark1, mark2, mark3], loc="upper right")
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
# major_kde = KernelDensity(kernel='gaussian').fit(z_train_major)

minor_easy_score = minor_easy_kde.score_samples(z_train_easy_minor)
minor_hard_score = minor_hard_kde.score_samples(z_train_hard_minor)

minor_easy_threshold = np.quantile(minor_easy_score, 0.1)
minor_hard_threshold = np.quantile(minor_hard_score, 0.7)

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
# g1=sns.scatterplot(x="x1", y="x2", data=tmp_df.loc[(y_train == 0) | (y_train == 2), :], hue="label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
sns.scatterplot(x="x1", y="x2", data=x_train_filtered_easy_minor, hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
sns.scatterplot(x="x1", y="x2", data=x_train_filtered_hard_minor, hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
#%% One KDE
z_train = z_tildes.detach().numpy()
y_train = true_y.detach().numpy()

z_train_minor = z_train[(y_train == 3) | (y_train == 1), :]

minor_kde = KernelDensity(kernel='gaussian').fit(z_train_minor)

minor_score = minor_kde.score_samples(z_train_minor)
treshold = np.quantile(minor_score, 0.35)
z_train_filtered_minor = z_train_minor[minor_score > treshold, :]
#%%
g1=sns.scatterplot(x="Z1", y="Z2", data=z.loc[(y_train == 0) | (y_train == 2), :], hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
sns.scatterplot(x="Z1", y="Z2", data=z.loc[(y_train == 1) | (y_train == 3), :].loc[minor_score > treshold, :], hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlim([-3.0, 3.0])  
plt.ylim([-3.0, 3.0])
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
#%%
x_train_filtered_minor = tmp_df.loc[(y_train == 1) | (y_train == 3), :].loc[minor_score >= treshold, :]
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])
sns.scatterplot(x="x1", y="x2", data=x_train_filtered_minor, hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
#%%
x_train_filtered_minor = tmp_df.loc[(y_train == 1) | (y_train == 3), :].loc[minor_score >= treshold, :]
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])
g1=sns.scatterplot(x="x1", y="x2", data=tmp_df.loc[(y_train == 0) | (y_train == 2), :], hue="label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
sns.scatterplot(x="x1", y="x2", data=x_train_filtered_minor, hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
#%%
def smote_cls_sampling(model, target_num_samples, config, train_loader1, train_loader2, seed):
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

    z_train = z_tildes.detach().numpy()
    y_train = true_y.detach().numpy()

    z_train_easy_minor = z_train[y_train == 1, :]
    z_train_hard_minor = z_train[y_train == 3, :]

    minor_easy_kde = KernelDensity(kernel='gaussian').fit(z_train_easy_minor)
    minor_hard_kde = KernelDensity(kernel='gaussian').fit(z_train_hard_minor)

    minor_easy_score = minor_easy_kde.score_samples(z_train_easy_minor)
    minor_hard_score = minor_hard_kde.score_samples(z_train_hard_minor)

    minor_easy_threshold = np.quantile(minor_easy_score, 0.1)
    minor_hard_threshold = np.quantile(minor_hard_score, 0.7)

    x_train_filtered_easy_minor = x_train[y_train == 1, :][minor_easy_score >= minor_easy_threshold, :]
    x_train_filtered_hard_minor = x_train[y_train == 3, :][minor_hard_score > minor_hard_threshold, :]

    x_train_minor = np.concatenate([x_train_filtered_easy_minor, x_train_filtered_hard_minor], axis=0)
    x_train_major = x_train[(y_train== 0)| (y_train == 2), :]
    
    x_train_filtered = np.concatenate([x_train_major, x_train_minor], axis=0)
    y_train_filtered = np.concatenate([np.zeros((x_train_major.shape[0])), np.ones((x_train_minor.shape[0]))], axis=0)

    sm = SMOTE(random_state=seed, sampling_strategy={1: target_num_samples})
    X_res, y_res = sm.fit_resample(x_train_filtered, y_train_filtered)

    return X_res, y_res
#%%
dataset_for_benchmark = TensorDataset(torch.from_numpy(tmp_df.iloc[:, :2].values.astype('float32')), torch.Tensor(tmp_df["label"].values[:, np.newaxis]))
dataloader_for_benchmark = DataLoader(dataset_for_benchmark, batch_size=512, shuffle=False, drop_last=False)
#%%
oversampled_x, oversampled_y = smote_cls_sampling(model, 1500, config, dataloader, dataloader_for_benchmark, 15)
#%%
oversampled_data = pd.DataFrame(oversampled_x, columns=["x1", "x2"])
oversampled_data["label"] = oversampled_y

#%%
# g1=sns.scatterplot(x="x1", y="x2", data=tmp_df.loc[(y_train == 0) | (y_train == 2), :], hue="label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
sns.scatterplot(x="x1", y="x2", data=oversampled_data, hue="label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
# sns.scatterplot(x="x1", y="x2", data=x_train_filtered_hard_minor, hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
#%%
z_train = z_tildes.detach().numpy()
y_train = true_y.detach().numpy()

z_train_minor = z_train[(y_train == 3) | (y_train == 1), :]

minor_kde = KernelDensity(kernel='gaussian').fit(z_train_minor)

minor_score = minor_kde.score_samples(z_train_minor)
treshold = np.quantile(minor_score, 0.7)
#%%
z_train_filtered_minor = z_train_minor[minor_score > treshold, :]
g1=sns.scatterplot(x="Z1", y="Z2", data=z.loc[(y_train == 0) | (y_train == 2), :], hue="label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
sns.scatterplot(x="Z1", y="Z2", data=z.loc[(y_train == 1) | (y_train == 3), :].loc[minor_score > treshold, :], hue="label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlim([-3.0, 3.0])  
plt.ylim([-3.0, 3.0])
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
#%%
x_train_filtered_minor = tmp_df.loc[(y_train == 1) | (y_train == 3), :].loc[minor_score >= treshold, :]
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])
sns.scatterplot(x="x1", y="x2", data=x_train_filtered_minor, hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
# %%
config["input_dim"] = 2

ddhs = DDHS(config, device) 

optimizer = torch.optim.Adam(
        ddhs.parameters(), 
        lr=0.001
    )

pbar = tqdm(range(1000))

for epoch in pbar:
    logs = ddhs_train(ddhs, dataloader_for_benchmark, optimizer, config, device)
#%%
samples = []
true_y = []
z_tildes = []
predicted_y = []
train_x = []
for batch_x, batch_y in dataloader_for_benchmark:
    xhat, yhat = ddhs(batch_x)
    z_tilde = ddhs.encode(batch_x)
    samples.append(xhat)
    true_y.append(batch_y)
    z_tildes.append(z_tilde)
    predicted_y.append(yhat)
    train_x.append(batch_x)

train_x = torch.cat(train_x, dim=0).detach().numpy()
samples = torch.cat(samples, dim=0)
samples = samples.detach()
true_y = torch.cat(true_y, dim=0)
true_y = true_y.detach().squeeze()
z_tildes = torch.cat(z_tildes, dim=0)
z_tildes = z_tildes.detach().squeeze()
predicted_y = torch.cat(predicted_y, dim=0)
predicted_y = predicted_y.detach().squeeze()

ddhs_z = pd.DataFrame(z_tildes[:, :2], columns=["Z1", "Z2"])
ddhs_z["label"] = tmp_df["true_label"]
#%%
g=sns.jointplot(data=ddhs_z, x="Z1", y="Z2", kind="kde", hue="label", marginal_ticks=False, thresh=0.1, alpha=0.3, levels=4, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"}, label="Density")
g.ax_marg_x.remove()
g.ax_marg_y.remove()
g.set_axis_labels("","")
sns.scatterplot(x=ddhs_z.loc[ddhs_z["label"] == 0, "Z1"],y=ddhs_z.loc[ddhs_z["label"] == 0, "Z2"], s=40, alpha=0.8, c="#66b2ff", label="Major")
sns.scatterplot(x=ddhs_z.loc[ddhs_z["label"] == 1, "Z1"],y=ddhs_z.loc[ddhs_z["label"] == 1, "Z2"], s=40, alpha=0.8, c="#ff8000", label="Minor")
sns.scatterplot(x=ddhs_z.loc[ddhs_z["label"] == 2, "Z1"],y=ddhs_z.loc[ddhs_z["label"] == 2, "Z2"], s=40, alpha=0.8, c="k", label="Noise")
plt.scatter(ddhs_z.loc[tmp_df["label"] == 1, "Z1"].mean(), ddhs_z.loc[tmp_df["label"] == 1, "Z2"].mean(), s=100, c='r')
plt.scatter(ddhs_z.loc[tmp_df["label"] == 0, "Z1"].mean(), ddhs_z.loc[tmp_df["label"] == 0, "Z2"].mean(), s=100, c='g')
plt.xlabel("")
plt.ylabel("")
# plt.xlim(0.0, 0.35)
# plt.ylim(-.52, -.16)
plt.legend('',frameon=False)
#%%
minor_kde = KernelDensity(kernel='gaussian').fit(z_tildes[true_y == 1, :])
major_kde = KernelDensity(kernel='gaussian').fit(z_tildes[true_y == 0, :])
samples.shape
minor_score = minor_kde.score_samples(z_tildes[true_y == 1, :])
major_score = major_kde.score_samples(z_tildes[true_y == 0, :])
minor_score_q3 = np.quantile(minor_score, 0.75)
major_score_q2 = np.quantile(major_score, 0.5)

selected_minor_z = ddhs_z.loc[((ddhs_z["label"] == 1) | (ddhs_z["label"] == 2)), :].loc[(minor_score > minor_score_q3), :]
selected_major_z = ddhs_z.loc[(ddhs_z["label"] == 0), :].loc[(major_score > major_score_q2), : ]
#%%
sns.scatterplot(x="Z1", y="Z2", data=selected_major_z, s=40, alpha=0.5, c="#66b2ff")
sns.scatterplot(x="Z1", y="Z2", data=selected_minor_z, hue="label", s=40, alpha=0.8, palette={1: "#ff8000", 2:"k"})
plt.scatter(ddhs_z.loc[tmp_df["label"] == 1, "Z1"].mean(), ddhs_z.loc[tmp_df["label"] == 1, "Z2"].mean(), s=100, c='r')
plt.scatter(ddhs_z.loc[tmp_df["label"] == 0, "Z1"].mean(), ddhs_z.loc[tmp_df["label"] == 0, "Z2"].mean(), s=100, c='g')
plt.xlim(0.17, 0.45)
plt.ylim(-.63, -.31)
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
#%%
ddhs_minor_x_trained = tmp_df.loc[((ddhs_z["label"] == 1) | (ddhs_z["label"] == 2)), :].loc[(minor_score > minor_score_q3), :]
g1=sns.scatterplot(x="x1", y="x2", data=ddhs_minor_x_trained, hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])
plt.xlabel("")
plt.ylabel("")
plt.legend('',frameon=False)
#%%
x_oversampled, minor_x, selected_major_x = ddhs_sampling(ddhs, 1400, config, dataloader_for_benchmark, 0)
classifier = DecisionTreeClassifier(max_depth=5, 
                                    random_state=0).fit(np.concatenate([selected_major_x, minor_x, x_oversampled], axis=0),
                                                        np.concatenate([np.zeros((len(selected_major_x))), np.ones((len(minor_x))),np.ones((1400))], axis=0))
#%%
cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["#66b2ff","#ff8000"])

disp = DecisionBoundaryDisplay.from_estimator(classifier, 
                                              np.concatenate([selected_major_x, minor_x, x_oversampled], axis=0), 
                                              response_method="predict",
                                              alpha=0.3, 
                                              cmap=cmap)

disp.ax_.scatter(np.concatenate([selected_major_x, minor_x, x_oversampled], axis=0)[:, 0],
                 np.concatenate([selected_major_x, minor_x, x_oversampled], axis=0)[:, 1], 
                 s=40, alpha=0.7,
                 c=np.concatenate([np.zeros((len(selected_major_x))), np.ones((len(minor_x))),np.ones((1400))], axis=0), edgecolor="w",
                 cmap=cmap)
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])

#%%
dfbs = AutoEncoder(config, device)

optimizer = torch.optim.Adam(
        dfbs.parameters(), 
        lr=0.001
    )

pbar = tqdm(range(1000))

for epoch in pbar:
    logs = dfbs_train(dfbs, dataloader_for_benchmark, optimizer, config, device)

#%%
samples = []
true_y = []
z_tildes = []

for batch_x, batch_y in dataloader_for_benchmark:
    xhat = dfbs(batch_x)
    z_tilde = dfbs.encode(batch_x)
    samples.append(xhat)
    true_y.append(batch_y)
    z_tildes.append(z_tilde)

samples = torch.cat(samples, dim=0)
samples = samples.detach()
true_y = torch.cat(true_y, dim=0)
true_y = true_y.detach().squeeze()
z_tildes = torch.cat(z_tildes, dim=0)
z_tildes = z_tildes.detach().squeeze()

#%%
dfbs_z = pd.DataFrame(z_tildes[:, :2], columns=["Z1", "Z2"])
dfbs_z["label"] = tmp_df["true_label"]

sns.scatterplot(x=dfbs_z.loc[dfbs_z["label"] == 0, "Z1"],y=dfbs_z.loc[dfbs_z["label"] == 0, "Z2"], s=40, alpha=0.8, c="#66b2ff", label="Major")
sns.scatterplot(x=dfbs_z.loc[dfbs_z["label"] == 1, "Z1"],y=dfbs_z.loc[dfbs_z["label"] == 1, "Z2"], s=40, alpha=0.8, c="#ff8000", label="Minor")
sns.scatterplot(x=dfbs_z.loc[dfbs_z["label"] == 2, "Z1"],y=dfbs_z.loc[dfbs_z["label"] == 2, "Z2"], s=40, alpha=0.8, c="k", label="Noise")
plt.scatter(dfbs_z.loc[tmp_df["label"] == 1, "Z1"].mean(), dfbs_z.loc[tmp_df["label"] == 1, "Z2"].mean(), s=100, c='r')
plt.scatter(dfbs_z.loc[tmp_df["label"] == 0, "Z1"].mean(), dfbs_z.loc[tmp_df["label"] == 0, "Z2"].mean(), s=100, c='g')

plt.xlabel("")
plt.ylabel("")
mark1 = mlines.Line2D([], [], c="#66b2ff", lw=0, marker='o', label="Major")
mark2 = mlines.Line2D([], [], c="#ff8000", lw=0, marker='o', label="Minor")
mark3 = mlines.Line2D([], [], c="k", lw=0, marker='o', label="Noise")
plt.legend(handles=[mark1, mark2, mark3], fontsize="15", loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
#%%
major_center = z_tildes[true_y == 0, :].mean(dim=0).detach().numpy()
minor_center = z_tildes[true_y == 1, :].mean(dim=0).detach().numpy()
all_center = z_tildes.mean(dim=0).detach().numpy()

candidate_z = dfbs_z.loc[(dfbs_z["label"] == 1) | (dfbs_z["label"] == 2), :]
dist_M = np.power(candidate_z.iloc[:, :2].values - major_center, 2).sum(axis=1) 
dist_A = np.power(candidate_z.iloc[:, :2].values - all_center, 2).sum(axis=1)
dist_m = np.power(candidate_z.iloc[:, :2].values - minor_center, 2).sum(axis=1) 
#%%
final_candidate = candidate_z.loc[((dist_M > dist_A) & (dist_A > dist_m)), :]
sns.scatterplot(x=dfbs_z.loc[dfbs_z["label"] == 0, "Z1"],y=dfbs_z.loc[dfbs_z["label"] == 0, "Z2"], s=40, alpha=0.8, c="#66b2ff", label="Major")
sns.scatterplot(x="Z1",y="Z2", data=final_candidate, hue="label", s=40, alpha=0.8, palette={1: "#ff8000", 2:"k"})
plt.scatter(dfbs_z.loc[tmp_df["label"] == 1, "Z1"].mean(), dfbs_z.loc[tmp_df["label"] == 1, "Z2"].mean(), s=100, c='r')
plt.scatter(dfbs_z.loc[tmp_df["label"] == 0, "Z1"].mean(), dfbs_z.loc[tmp_df["label"] == 0, "Z2"].mean(), s=100, c='g')

plt.xlabel("")
plt.ylabel("")

mark1 = mlines.Line2D([], [], c="#66b2ff", lw=0, marker='o', label="Major")
mark2 = mlines.Line2D([], [], c="#ff8000", lw=0, marker='o', label="Minor")
mark3 = mlines.Line2D([], [], c="k", lw=0, marker='o', label="Noise")
plt.legend(handles=[mark1, mark2, mark3], fontsize="15", loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
#%%
dfbs_x_trained = tmp_df.loc[(dfbs_z["label"] == 1) | (dfbs_z["label"] == 2), :].loc[((dist_M > dist_A) & (dist_A > dist_m)), :]
g1=sns.scatterplot(x="x1", y="x2", data=dfbs_x_trained, hue="true_label", s=40, alpha=0.8, palette={0: "#66b2ff", 1:"#ff8000", 2:"k"})
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])
plt.xlabel("")
plt.ylabel("")
mark2 = mlines.Line2D([], [], c="#ff8000", lw=0, marker='o', label="Minor")
mark3 = mlines.Line2D([], [], c="k", lw=0, marker='o', label="Noise")
plt.legend(handles=[mark2, mark3], fontsize="15", loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=2, fancybox=True, shadow=True)
#%%
x_oversampled = dfbs_sampling(ddhs, 1400, config, dataloader_for_benchmark, 0)
#%%
classifier = DecisionTreeClassifier(max_depth=5, 
                                    random_state=0).fit(np.concatenate([train_x, x_oversampled], axis=0),
                                                        np.concatenate([true_y, np.ones((1400))], axis=0))
#%%
disp = DecisionBoundaryDisplay.from_estimator(classifier, 
                                              np.concatenate([train_x, x_oversampled], axis=0), 
                                              response_method="predict",
                                              alpha=0.3, 
                                              cmap=cmap)

disp.ax_.scatter(np.concatenate([train_x, x_oversampled], axis=0)[:, 0],
                 np.concatenate([train_x, x_oversampled], axis=0)[:, 1], 
                 s=40, alpha=0.7,
                 c=np.concatenate([true_y, np.ones((1400))], axis=0), edgecolor="w",
                 cmap=cmap)
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])
mark1 = mlines.Line2D([], [], c="#66b2ff", lw=0, marker='o', label="Major")
mark2 = mlines.Line2D([], [], c="#ff8000", lw=0, marker='o', label="Minor")
# mark4 = mlines.Line2D([], [], c="#ff6666", lw=0, marker='o', label="Hard Minor Label")
plt.legend(handles=[mark1, mark2], fontsize="15", loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
#%%
dsmote = AutoEncoder(config, device)

optimizer = torch.optim.Adam(
        dsmote.parameters(), 
        lr=0.001
    )

pbar = tqdm(range(500))

for epoch in pbar:
    logs = dsmote_train(dsmote, dataloader_for_benchmark, optimizer, config, device)

#%%
samples = []
true_y = []
z_tildes = []

for batch_x, batch_y in dataloader_for_benchmark:
    xhat = dsmote(batch_x)
    z_tilde = dsmote.encode(batch_x)
    samples.append(xhat)
    true_y.append(batch_y)
    z_tildes.append(z_tilde)

samples = torch.cat(samples, dim=0)
samples = samples.detach()
true_y = torch.cat(true_y, dim=0)
true_y = true_y.detach().squeeze()
z_tildes = torch.cat(z_tildes, dim=0)
z_tildes = z_tildes.detach().squeeze()
#%%
dsmote_z = pd.DataFrame(z_tildes[:, :2], columns=["Z1", "Z2"])
dsmote_z["label"] = tmp_df["true_label"]
#%%
sns.scatterplot(x=dsmote_z.loc[dsmote_z["label"] == 0, "Z1"],y=dsmote_z.loc[dsmote_z["label"] == 0, "Z2"], s=40, alpha=0.8, c="#66b2ff", label="Major")
sns.scatterplot(x=dsmote_z.loc[dsmote_z["label"] == 1, "Z1"],y=dsmote_z.loc[dsmote_z["label"] == 1, "Z2"], s=40, alpha=0.8, c="#ff8000", label="Minor")
sns.scatterplot(x=dsmote_z.loc[dsmote_z["label"] == 2, "Z1"],y=dsmote_z.loc[dsmote_z["label"] == 2, "Z2"], s=40, alpha=0.8, c="k", label="Noise")
plt.xlabel("")
plt.ylabel("")       
plt.xlim([-0.7, -.2])  
plt.ylim([-.11, .12])
plt.legend('',frameon=False)
#%%
x_oversampled = dsmote_sampling(ddhs, 1400, config, dataloader_for_benchmark, 0)
#%%
classifier = DecisionTreeClassifier(max_depth=5, 
                                    random_state=0).fit(np.concatenate([train_x, x_oversampled], axis=0),
                                                        np.concatenate([true_y, np.ones((1400))], axis=0))

disp = DecisionBoundaryDisplay.from_estimator(classifier, 
                                              np.concatenate([train_x, x_oversampled], axis=0), 
                                              response_method="predict",
                                              alpha=0.3, 
                                              cmap=cmap)

disp.ax_.scatter(np.concatenate([train_x, x_oversampled], axis=0)[:, 0],
                 np.concatenate([train_x, x_oversampled], axis=0)[:, 1], 
                 s=40, alpha=0.7,
                 c=np.concatenate([true_y, np.ones((1400))], axis=0), edgecolor="w",
                 cmap=cmap)
plt.xlim([-1.5, 1.5])  
plt.ylim([-1.5, 1.5])