import torch as torch
import numpy as np
import random 

from sklearn.neighbors import NearestNeighbors, KernelDensity

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
# def smote_cls_sampling(model, target_num_samples, config, train_loader, seed):
#     set_random_seed(seed)
    
#     z_train = []
#     y_train = []
#     transformer = config["transformer"]
#     model.eval()
    
#     for batch_x, batch_y in train_loader:
#         # batch_x, batch_y = next(iter(dataloader))
#         _, _, _, _, _, z_tilde, _ = model(batch_x)
#         z_train.append(z_tilde)
#         y_train.append(batch_y)
    
#     z_train = torch.cat(z_train, dim=0).detach().numpy()
#     y_train = torch.cat(y_train, dim=0).squeeze().detach().numpy()
    
#     minor_z = z_train[(y_train == 1) | (y_train == 3), :]
#     # minor_z = minor_z_[(minor_z_[:, 1] < 0), :]
#     # minor_z = minor_z_[minor_z_[:, 1] < 0, :]
    
#     n_neigh = 4
#     knn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
#     knn.fit(minor_z)
#     dist, ind = knn.kneighbors(minor_z)

#     # generating samples
#     base_indices = np.random.choice(list(range(len(minor_z))),target_num_samples)
#     neighbor_indices = np.random.choice(list(range(1, n_neigh)),target_num_samples)

#     minor_z_base = minor_z[base_indices]
#     minor_z_neighbor = minor_z[ind[base_indices, neighbor_indices]]

#     smote_z = minor_z_base + np.multiply(np.random.rand(target_num_samples,1),
#             minor_z_neighbor - minor_z_base)
    
#     oversampled = model.decoder(torch.Tensor(smote_z)).detach().cpu().numpy()
    
#     samples = transformer.inverse_transform(oversampled, model.sigma.detach().cpu().numpy())
    
#     return samples


def dsmote_sampling(model, target_num_samples, config, train_loader, seed):
    set_random_seed(seed)
    
    model.eval()
    
    # x_train = []
    y_train = []
    z_train = []
    
    
    for batch_x, batch_y in train_loader:
        z = model.encode(batch_x)
        y_train.append(batch_y)
        z_train.append(z)
        
    # x_train = torch.cat(x_train, dim=0).detach().numpy()
    y_train = torch.cat(y_train, dim=0).squeeze().detach().numpy()
    z_train = torch.cat(z_train, dim=0).detach().numpy()

    minor_label = 1        
    minor_z = z_train[y_train == minor_label, :]
    
    n_neigh = 3
    knn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    knn.fit(minor_z)
    dist, ind = knn.kneighbors(minor_z)

    # generating samples
    base_indices = np.random.choice(list(range(len(minor_z))),target_num_samples)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)),target_num_samples)

    minor_z_base = minor_z[base_indices]
    minor_z_neighbor = minor_z[ind[base_indices, neighbor_indices]]

    samples = minor_z_base + np.multiply(np.random.rand(target_num_samples,1),
            minor_z_neighbor - minor_z_base)
    
    oversampled = model.decode(torch.Tensor(samples)).detach().cpu().numpy()
    
    return oversampled

def dfbs_sampling(model, target_num_samples, config, train_loader, seed):
    set_random_seed(seed)
    
    train_x = []
    samples = []
    train_y = []
    
    for batch_x, batch_y in train_loader:
        z = model.encode(batch_x)
        
        train_x.append(batch_x)
        samples.append(z)
        train_y.append(batch_y)
        
    train_x = torch.cat(train_x, dim=0).detach()
    samples = torch.cat(samples, dim=0)
    samples = samples.detach()
    train_y = torch.cat(train_y, dim=0)
    train_y = train_y.detach().squeeze()
    
    minor_x = train_x[train_y == 1, :].detach().cpu().numpy()
    
    major_center = samples[train_y == 0, :].mean(dim=0)
    minor_center = samples[train_y == 1, :].mean(dim=0)
    all_center = samples.mean(dim=0)
    
    total_gen_sample = 0
    oversampled = []
    model.eval()
    
    while total_gen_sample < target_num_samples:
        candidate = np.zeros((3000, minor_x.shape[1]))
        for i in range(minor_x.shape[1]):
            candidate[:, i] = minor_x[np.random.choice(list(range(len(minor_x))), 3000, replace=True), i]
        
        candidate_z = model.encode(torch.Tensor(candidate))
        dist_M = torch.pow(candidate_z - major_center, 2).sum(dim=1) 
        dist_A = torch.pow(candidate_z - all_center, 2).sum(dim=1)
        dist_m = torch.pow(candidate_z - minor_center, 2).sum(dim=1) 

        x_upsampled = candidate[((dist_M > dist_A) & (dist_A > dist_m)).detach().cpu().numpy(), :]

        if x_upsampled.shape[0] + total_gen_sample > target_num_samples:
            x_upsampled_ = x_upsampled[np.random.choice(x_upsampled.shape[0], target_num_samples - total_gen_sample, replace=True), :]
            oversampled.append(x_upsampled_)
            total_gen_sample += x_upsampled_.shape[0]
        else:
            oversampled.append(x_upsampled)
            total_gen_sample += x_upsampled.shape[0]
            
    return np.concatenate(oversampled, axis=0)

def ddhs_sampling(model, target_num_samples, config, train_loader, seed):
    set_random_seed(seed)
    
    train_x = []
    samples = []
    train_y = []
    
    for batch_x, batch_y in train_loader:
        z = model.encode(batch_x)
        train_x.append(batch_x)
        samples.append(z)
        train_y.append(batch_y)
        
    train_x = torch.cat(train_x, dim=0).detach().cpu().numpy()
    samples = torch.cat(samples, dim=0)
    samples = samples.detach()
    train_y = torch.cat(train_y, dim=0)
    train_y = train_y.detach().squeeze().cpu().numpy()
    

    minor_label = 1
    major_label = 0    
    
    minor_kde = KernelDensity(kernel='gaussian').fit(samples[train_y == minor_label, :])
    major_kde = KernelDensity(kernel='gaussian').fit(samples[train_y == major_label, :])

    minor_score = minor_kde.score_samples(samples[train_y == minor_label, :])
    major_score = major_kde.score_samples(samples[train_y == major_label, :])

    minor_score_q3 = np.quantile(minor_score, 0.75)
    major_score_q2 = np.quantile(major_score, 0.5)

    selected_minor_z = samples[train_y == minor_label, :][minor_score > minor_score_q3]
    selected_major_z = samples[train_y == major_label, :][major_score > major_score_q2]

    selected_minor_x = train_x[train_y == minor_label, :][minor_score > minor_score_q3]

    minor_center = selected_minor_z.mean(dim=0)
    major_center = selected_major_z.mean(dim=0)    
    
    total_gen_sample = 0
    
    oversampled = []
    squared_radius = torch.pow(minor_center - selected_minor_z, 2).sum(dim=1).max()
    model.eval()
    
    while total_gen_sample < target_num_samples:
        candidate = np.zeros((3000, selected_minor_x.shape[1]))
        for i in range(selected_minor_x.shape[1]):
            candidate[:, i] = selected_minor_x[np.random.choice(list(range(len(selected_minor_x))), 3000, replace=True), i]
                    
        candidate_z = model.encode(torch.Tensor(candidate))
        dist_M = torch.pow(candidate_z - major_center, 2).sum(dim=1) 
        dist_m = torch.pow(candidate_z - minor_center, 2).sum(dim=1) 

        x_upsampled = candidate[((dist_M > dist_m) & (squared_radius > dist_m)).detach().cpu().numpy(), :]

        if x_upsampled.shape[0] + total_gen_sample > target_num_samples:
            x_upsampled_ = x_upsampled[np.random.choice(x_upsampled.shape[0], target_num_samples - total_gen_sample, replace=True), :]
            oversampled.append(x_upsampled_)
            total_gen_sample += x_upsampled_.shape[0]
        else:
            oversampled.append(x_upsampled)
            total_gen_sample += x_upsampled.shape[0]
            
    return np.concatenate(oversampled, axis=0)

def cvae_sampling(model, target_num_samples, config, train_loader, seed):
    set_random_seed(seed)
    
    train_x = []
    samples = []
    train_y = []
    
    for batch_x, batch_y in train_loader:
        z, _ = model.encode(batch_x, batch_y)
        train_x.append(batch_x)
        samples.append(z)
        train_y.append(batch_y)
        
    train_x = torch.cat(train_x, dim=0).detach().cpu().numpy()
    samples = torch.cat(samples, dim=0)
    samples = samples.detach()
    train_y = torch.cat(train_y, dim=0)
    train_y = train_y.detach().squeeze().cpu().numpy()

    minor_z = samples[train_y == 1, :]

    n_neigh = 4
    knn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    knn.fit(minor_z)
    dist, ind = knn.kneighbors(minor_z)

    # generating samples
    base_indices = np.random.choice(list(range(len(minor_z))),target_num_samples)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)),target_num_samples)

    minor_z_base = minor_z[base_indices]
    minor_z_neighbor = minor_z[ind[base_indices, neighbor_indices]]

    smote_z = minor_z_base + np.multiply(np.random.rand(target_num_samples,1),
            minor_z_neighbor - minor_z_base)
    
    oversampled = model.decode(torch.tensor(smote_z, dtype=torch.float32), torch.ones(smote_z.shape[0], 1, dtype=torch.float32)).detach().cpu().numpy()
    
    return oversampled