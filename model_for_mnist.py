import torch
import torch.nn as nn
import torch.nn.functional as F

# import xgboost

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, output_dim=4):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, latent_dim, class_num, device=device):
        super(MLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 16 * self.latent_dim),
            nn.ELU(),
            nn.Linear(16 * self.latent_dim, 8 * self.latent_dim),
            nn.ELU(),
            nn.Linear(8 * self.latent_dim, class_num),
            nn.Softmax()
        ).to(device)

    def forward(self, input):
        yhat = self.net(input)
        return yhat

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, class_num, device=device):
        super(MLPEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.class_num = class_num
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 16 * self.latent_dim * self.class_num),
            nn.ELU(),
            nn.Linear(16 * self.latent_dim * self.class_num, 8 * self.latent_dim * self.class_num),
            nn.ELU(),
            nn.Linear(8 * self.latent_dim * self.class_num, 2 * self.latent_dim * self.class_num),
        ).to(device)
        
    def forward(self, input):
        h = self.net(input)
        mean, logvar = torch.split(h, split_size_or_sections=self.latent_dim * self.class_num, dim=-1)
        mean = torch.split(mean, split_size_or_sections=self.latent_dim, dim=-1)
        logvar = torch.split(logvar, split_size_or_sections=self.latent_dim, dim=-1)
        return torch.stack(mean, dim=1), torch.stack(logvar, dim=1)

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, activation="identity", device=device):
        super(MLPDecoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation
        
        if self.activation == "identity":
            self.act = nn.Identity()
        elif self.activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif self.activation == "tanh":
            self.act = nn.Tanh()
        
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, 8 * self.latent_dim),
            nn.ELU(),
            nn.Linear(8 * self.latent_dim, 16 * self.latent_dim),
            nn.ELU(),
            nn.Linear(16 * self.latent_dim, self.input_dim)
        ).to(device)
        
    def forward(self, input):
        h = self.net(input)
        output = self.act(h) 
        return output

class MixtureMLPVAE(nn.Module):
    def __init__(self, config, class_num, classifier, device=device):
        super(MixtureMLPVAE, self).__init__()
        self.config = config
        self.device = device
        self.hard = True
        
        if not config["activation"]:
            config["activation"] = "identity"
        
        if classifier == None:
            self.classifier = MLPClassifier(config["input_dim"], config["latent_dim"], class_num, device=device)
        else:
            self.classifier = classifier
        
        self.encoder = MLPEncoder(config["input_dim"], config["latent_dim"], class_num, device=device)
        self.decoder = MLPDecoder(config["input_dim"], config["latent_dim"], config["activation"], device=device)
        
        self.sigma = nn.Parameter(torch.ones(config["input_dim"]) * 0.1)
    
    def sample_gumbel(self, shape):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + 1e-8) + 1e-8)

    def gumbel_max_sample(self, probs):
        y = torch.log(probs + 1e-8) + self.sample_gumbel(probs.shape).to(self.device)
        if self.hard:
            y_hard = (y == torch.max(y, 1, keepdim=True)[0]).type(y.dtype)
            y = (y_hard - y).detach() + y
        return y
    
    def encode(self, input):
        mean, logvar = self.encoder(input)
        return mean, logvar
    
    def classify(self, input):
    
        probs = self.classifier(input.reshape(-1, 1, 28, 28))
        # else:
        #     probs = self.classifier(input)
                
        return probs
    
    def decode(self, input):
        xhat = self.decoder(input)
        return xhat
    
    def forward(self, input, sampling=True):
        mean, logvar = self.encoder(input)
        
        if sampling:
            epsilon = torch.randn(mean.shape).to(self.device)
            z = mean + epsilon * torch.exp(logvar / 2)
        else:
            z = mean
            
        probs = self.classify(input)
        y = self.gumbel_max_sample(probs)
        
        z_tilde = torch.matmul(y[:, None, :], z).squeeze(1)
        
        xhat = self.decoder(z_tilde)
        
        return mean, logvar, probs, y, z, z_tilde, xhat

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, device=device):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 16 * self.latent_dim),
            nn.ELU(),
            nn.Linear(16 * self.latent_dim, 8 * self.latent_dim),
            nn.ELU(),
            nn.Linear(8 * self.latent_dim, self.latent_dim),
        ).to(device)
        
    def forward(self, input):
        z = self.net(input)
        return z

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, activation="identity", device=device):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation
        
        if self.activation == "identity":
            self.act = nn.Identity()
        elif self.activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif self.activation == "tanh":
            self.act = nn.Tanh()
        
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, 8 * self.latent_dim),
            nn.ELU(),
            nn.Linear(8 * self.latent_dim, 16 * self.latent_dim),
            nn.ELU(),
            nn.Linear(16 * self.latent_dim, self.input_dim)
        ).to(device)
        
    def forward(self, input):
        h = self.net(input)
        output = self.act(h) 
        return output
    
class AutoEncoder(nn.Module):
    def __init__(self, config, device=device):
        super(AutoEncoder, self).__init__()
        self.config = config
        self.device = device
        
        self.encoder = Encoder(config["input_dim"], config["latent_dim"], device=device)
        self.decoder = Decoder(config["input_dim"], config["latent_dim"], config["activation"], device=device)
        
    def encode(self, input):
        z = self.encoder(input)
        return z

    def decode(self, input):
        xhat = self.decoder(input)
        return xhat
    
    def forward(self, input):
        z = self.encoder(input)                    
        xhat = self.decoder(z)

        return xhat

class MLPClassifier_DDHS(nn.Module):
    def __init__(self, latent_dim, device=device):
        super(MLPClassifier_DDHS, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, 8 * self.latent_dim),
            nn.ELU(),
            nn.Linear(8 * self.latent_dim, 16 * self.latent_dim),
            nn.ELU(),
            nn.Linear(16 * self.latent_dim, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, input):
        yhat = self.net(input)
        return yhat
        
class DDHS(nn.Module):
    def __init__(self, config, device=device):
        super(DDHS, self).__init__()
        self.config = config
        self.device = device
        
        self.encoder = Encoder(config["input_dim"], config["latent_dim"], device=device)
        self.decoder = Decoder(config["input_dim"], config["latent_dim"], config["activation"], device=device)
        self.classifier = MLPClassifier_DDHS(config["latent_dim"], device=device)
        
    def encode(self, input):
        z = self.encoder(input)
        return z

    def decode(self, input):
        xhat = self.decoder(input)
        return xhat
    
    def forward(self, input):
        z = self.encoder(input)                    
        yhat = self.classifier(z)
        xhat = self.decoder(z)

        return xhat, yhat

class CVAE(nn.Module):
    def __init__(self, config):
        super(CVAE, self).__init__()

        self.class_size = 1
        
        # encoder
        self.fc1  = nn.Linear(config["input_dim"] + self.class_size, 8 * config["latent_dim"])
        self.fc2  = nn.Linear(8 * config["latent_dim"], 4 * config["latent_dim"])
        self.fc3  = nn.Linear(4 * config["latent_dim"], 2 * config["latent_dim"])
        self.fc31 = nn.Linear(2 * config["latent_dim"], config["latent_dim"])
        self.fc32 = nn.Linear(2 * config["latent_dim"], config["latent_dim"])

        # decoder
        self.fc4 = nn.Linear(config["latent_dim"] + self.class_size, 2 * config["latent_dim"])
        self.fc5 = nn.Linear(2 * config["latent_dim"], 4 * config["latent_dim"])
        self.fc6 = nn.Linear(4 * config["latent_dim"], 8 * config["latent_dim"])
        self.fc7 = nn.Linear(8 * config["latent_dim"], config["input_dim"])
        
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1) 
        h1 = self.elu(self.fc3(self.fc2(self.fc1(inputs))))
        z_mu = self.fc31(h1)
        z_var = self.fc32(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): 
        inputs = torch.cat([z, c], 1) 
        output = self.fc7(self.fc6(self.fc5(self.fc4(inputs))))
        return output

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
