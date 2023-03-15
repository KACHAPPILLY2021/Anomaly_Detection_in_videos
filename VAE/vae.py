from typing import List, Tuple, Dict
import torch
from torch import nn
from torch.functional import Tensor
from torch.nn import functional as F

class VAE(nn.Module):
  def __init__(self, 
               in_channels: int,
               latent_dim: int,
               hidden_dims: List,
               dropout: float = 0.5):
    super(VAE, self).__init__()

    self.latent_dim = latent_dim

    out_channels = in_channels

    modules = []
    for h_dim in hidden_dims:
      modules.append(
        nn.Sequential(
          nn.Dropout(dropout),
          nn.Linear(in_channels, h_dim),
          nn.LeakyReLU()
        )
      )
      in_channels = h_dim

    self.encoder = nn.Sequential(*modules)
    self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
    self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

    
    hidden_dims.reverse()
    modules = [
      nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(latent_dim, hidden_dims[0]),
      nn.LeakyReLU()
    )]
    

    for i in range(len(hidden_dims) - 1):
      modules.append(
        nn.Sequential(
          nn.Linear(hidden_dims[i], hidden_dims[i+1]),
          nn.LeakyReLU()
        )
      )
    self.decoder = nn.Sequential(*modules)

    self.final_layer = nn.Sequential(
      nn.Linear(hidden_dims[-1], out_channels),
      
    )
  def encode(self, x: Tensor) -> Tuple[Tensor]:
    x = self.encoder(x)

    mu = self.fc_mu(x)
    log_var = self.fc_var(x)
    return (mu, log_var)
  def decode(self, z: Tensor) -> Tensor:
    res = self.decoder(z)
    res = self.final_layer(res)
    return res
  
  def reparameterize(self, mu: Tensor, logvar: Tensor)-> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

  def forward(self, x: Tensor) -> Dict:
    mu, log_var = self.encode(x)
    z = self.reparameterize(mu, log_var)
    res = self.decode(z)
    out_dict = {
      'result': res,
      'input': x,
      'mu': mu,
      'log_var': log_var
    }
    return out_dict
  def sample(self,
             num_samples: int,
             cur_device: int) -> Tensor:
    z = torch.randn(num_samples, self.latent_dim)
    z = z.to(cur_device)

    samples = self.decode(z)
    return samples

class ShareDecoderVae(nn.Module):
  def __init__(self, 
               in_channels: int,
               latent_dim: int,
               hidden_dims: List,
               dropout: float = 0.5):

    super(ShareDecoderVae, self).__init__()

    self.latent_dim = latent_dim

    out_channels = in_channels

    modules = []
    for h_dim in hidden_dims:
      modules.append(
        nn.Sequential(
          nn.Dropout(dropout),
          nn.Linear(in_channels, h_dim),
          nn.LeakyReLU()
        )
      )
      in_channels = h_dim

    self.encoder = nn.Sequential(*modules)
    self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
    self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

    
    hidden_dims.reverse()
    modules = [
      nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(latent_dim, hidden_dims[0]),
      nn.LeakyReLU()
    )]
    

    for i in range(len(hidden_dims) - 1):
      modules.append(
        nn.Sequential(
          nn.Linear(hidden_dims[i], hidden_dims[i+1]),
          nn.LeakyReLU()
        )
      )
    self.decoder1 = nn.Sequential(*modules)
    self.decoder2 = nn.Sequential(*modules)

    self.final_layer1 = nn.Sequential(
      nn.Linear(hidden_dims[-1], out_channels))

    self.final_layer2 = nn.Sequential(
      nn.Linear(hidden_dims[-1], out_channels))

  def encode(self, x: Tensor) -> Tuple[Tensor]:
    x = self.encoder(x)

    mu = self.fc_mu(x)
    log_var = self.fc_var(x)
    return (mu, log_var)
  

  def decode(self, z: Tensor) -> Tensor:
    res1 = self.decoder1(z)
    res1 = self.final_layer1(res1)
    res2 = self.decoder2(z)
    res2 = self.final_layer2(res2)
    return res1, res2
  
  def reparameterize(self, mu: Tensor, logvar: Tensor)-> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

  def forward(self, x: Tensor) -> Dict:
    mu, log_var = self.encode(x)
    z = self.reparameterize(mu, log_var)
    res1, res2 = self.decode(z)
    out_dict = {
      'result': [res1, res2],
      'input': x,
      'mu': mu,
      'log_var': log_var
    }
    return out_dict
