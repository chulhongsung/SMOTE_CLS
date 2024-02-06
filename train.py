import numpy as np 

import torch
import torch.nn.functional as F
from torch.nn.functional import cross_entropy

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
    
    output_info_list = config["transformer"].output_info_list
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        loss_ = []

        optimizer.zero_grad()

        
        '''ELBO'''
        mean, logvar, probs, y, z, z_tilde, xhat = model(batch_x)

        # if config["recon"] == "l2":
        #     error = (0.5 * torch.pow(batch_x - xhat, 2)).sum()
        # elif config["recon"] == "l1":
        #     error = torch.abs(batch_x - xhat).sum()
        # elif config["recon"] == "bce":
        #     error = F.binary_cross_entropy(xhat, batch_x, reduction='sum')
        
        start = 0
        recon = 0
        for column_info in output_info_list:
            for span_info in column_info:
                if span_info.activation_fn != 'softmax':
                    end = start + span_info.dim
                    std = model.sigma[start]
                    residual = batch_x[:, start] - torch.tanh(xhat[:, start])
                    recon += (residual ** 2 / 2 / (std ** 2)).sum()
                    recon += torch.log(std)
                    start = end
                else:
                    end = start + span_info.dim
                    recon += cross_entropy(
                        xhat[:, start:end], torch.argmax(batch_x[:, start:end], dim=-1), reduction='sum')
                    start = end
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

def dfbs_train(model, train_loader, optimizer, config, device):
    model.train()
    
    logs = {
        "recon": [],
        "center": []
    }
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        loss_ = []
        optimizer.zero_grad()        
        xhat = model(batch_x)
  
        error = (0.5 * torch.pow(batch_x - xhat, 2)).sum()
     
        major_emb = model.encode(batch_x[batch_y.squeeze() == 0, :])
        minor_emb = model.encode(batch_x[batch_y.squeeze() == 1, :])
        major_center = major_emb.mean(dim=0)
        minor_center = minor_emb.mean(dim=0)
        center_loss = (0.5 * torch.pow(major_emb - major_center, 2)).sum() + (0.5 * torch.pow(minor_emb - minor_center, 2)).sum()
        
        loss = error + center_loss 
        
        loss_.append(('recon', error))
        loss_.append(('center', center_loss))
        # encoder and decoder
        loss.backward()
        optimizer.step()
        
    for x, y in loss_:
        logs[x] = logs.get(x) + [y.item()]
        
    return logs 


def dsmote_train(model, train_loader, optimizer, config, device):
    model.train()
    logs = {
        "recon": [],
        "penalty": [],
        "total_loss": []
        
    }
    
    x_train = []
    y_train = []
    
    for batch_x, batch_y in train_loader:
        x_train.append(batch_x)
        y_train.append(batch_y)
    
    x_train = torch.cat(x_train, dim=0).detach().numpy()
    y_train = torch.cat(y_train, dim=0).squeeze().detach().numpy()
    
    minor_idx, = np.where(y_train == 1)
    major_idx, = np.where(y_train == 0)
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        loss_ = []

        optimizer.zero_grad()        
        xhat = model(batch_x)

        error = (0.5 * torch.pow(batch_x - xhat, 2)).mean()
        
        if np.random.binomial(1, 0.5, 1).item() == 0:
            samples = x_train[np.random.choice(major_idx, config["batch_size"], replace=True), :]
        else:
            samples = x_train[np.random.choice(minor_idx, config["batch_size"], replace=True), :]
        
        E_s = model.encode(torch.Tensor(samples))
        P_E = E_s[torch.randperm(E_s.size()[0])]
        
        penalty = (0.5 * torch.pow(E_s - P_E, 2)).mean()
        total_loss = error + penalty
        loss_.append(('recon', error))
        loss_.append(('penalty', penalty))
        loss_.append(('total_loss', total_loss))

        # encoder and decoder
        total_loss.backward()
        optimizer.step()
        
    for x, y in loss_:
        logs[x] = logs.get(x) + [y.item()]
        
    return logs 

def ddhs_train(model, train_loader, optimizer, config, device):
    model.train()
    
    logs = {
        "recon": [],
        "cce": [],
        "center": []
    }
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        loss_ = []

        optimizer.zero_grad()        
        xhat, yhat = model(batch_x)

        error = (0.5 * torch.pow(batch_x - xhat, 2)).sum()

        major_emb = model.encode(batch_x[batch_y.squeeze() == 0, :])
        minor_emb = model.encode(batch_x[batch_y.squeeze() == 1, :])
        major_center = major_emb.mean(dim=0)
        minor_center = minor_emb.mean(dim=0)
        center_loss = (0.5 * torch.pow(major_emb - major_center, 2)).sum() + (0.5 * torch.pow(minor_emb - minor_center, 2)).sum()
        
        bce = F.binary_cross_entropy(yhat, batch_y, reduction='sum')
        
        loss = error + center_loss + bce 
        
        loss_.append(('recon', error))
        loss_.append(('center', center_loss))
        loss_.append(('cce', bce))
        # encoder and decoder
        loss.backward()
        optimizer.step()
        
    for x, y in loss_:
        logs[x] = logs.get(x) + [y.item()]
        
    return logs 


def cvae_train(model, train_loader, optimizer, device):
    def loss_function(xhat, batch_x, mu, logvar):
        error = (0.5 * torch.pow(batch_x - xhat, 2)).sum()
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return error + 0.1 * KLD
    
    model.train()
    
    train_loss = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        xhat, mu, logvar = model(batch_x, batch_y)
        
        optimizer.zero_grad()
        loss = loss_function(xhat, batch_x, mu, logvar)
        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()
