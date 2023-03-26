import numpy as np
import matplotlib.pyplot as plt
import csv
import os

from dataset import Dataset, Dataset_v2
from torch.utils.data import DataLoader

from vae import *
import torch
from tqdm import tqdm

from test import validation
import json

dataset_name = 'ucf-crime'
dataset_name = 'shanghaitech'

# Specify path to save the generated pseudo features
save_root = '{}/augment/'.format(dataset_name)

train_dataset = Dataset_v2('./{}_i3d'.format(dataset_name), 
                          './S3R/data/{}'.format(dataset_name), 
                           split='all', transform=False)

augment = 1

exp_dir = 'exps/{}_sVAE_{}'.format(dataset_name, 2)

with open(os.path.join(exp_dir, 'config.json')) as f:
    config = json.load(f)


hidden_dims = config['hidden_dims']
latent_dim = config['latent_dim']
#hidden_dims.reverse()

sVAE = ShareDecoderVae(2048, latent_dim, hidden_dims, dropout=0.0)

sVAE.load_state_dict(torch.load(os.path.join(exp_dir, 'epoch_100.pth')))
sVAE = sVAE.cuda()
sVAE.eval()

n_cnt = 0
a_cnt = 0


save_dir = os.path.join(exp_dir, 'augment')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

pseudo_gt = dict()

for _ in range(augment):
  #for feature, label in zip(features, labels):
  for feature, label in tqdm(train_dataset):
    T, N, C = feature.shape
    if label > 0:

      # tmp_feature = feature.transpose(1,0,2) # T, N, C
      # sample_feature = feature[rand_i] # T/2, N, C

      # print(sample_feature.shape)
      sample_feature = torch.from_numpy(feature).reshape(-1, C)
      sample_feature = sample_feature.cuda().float()
      with torch.no_grad():
        #mu, log_var = abnormal_vae.encode(sample_feature)
        #z = abnormal_vae.reparameterize(mu, log_var)
        #nor_feat = normal_vae.decode(z)
        res = sVAE(sample_feature)
        nor_feat = res['result'][0]
      nor_feat = nor_feat.cpu().numpy().reshape(T, N, -1)

      # tmp_feature = tmp_feature.transpose(1, 0, 2)
      save_pth = os.path.join(save_dir, 'normal_{}'.format(n_cnt))
      np.save(save_pth, nor_feat)

      frame_lbl = np.zeros(T)

      pseudo_gt['normal_{}'.format(n_cnt)] = frame_lbl.tolist()

      n_cnt += 1


    else:
      if T > 5000: continue
      rand_i = np.random.choice(T, T//2, replace=False)

      # tmp_feature = feature.transpose(1,0,2) # T, N, C
      sample_feature = feature[rand_i] # T/2, N, C

      # print(sample_feature.shape)
      sample_feature = torch.from_numpy(sample_feature).reshape(-1, C)
      sample_feature = sample_feature.cuda().float()
      with torch.no_grad():
        #mu, log_var = normal_vae.encode(sample_feature)
        #z = normal_vae.reparameterize(mu, log_var)
        #abn_feat = abnormal_vae.decode(z)
        res = sVAE(sample_feature)
        abn_feat = res['result'][1]
      abn_feat = abn_feat.cpu().numpy().reshape(T//2, N, -1)

      tmp_feature = feature.copy()
      tmp_feature[rand_i] = abn_feat
      # tmp_feature = tmp_feature.transpose(1, 0, 2)
      
      save_pth = os.path.join(save_dir, 'abnormal_{}'.format(a_cnt))
      np.save(save_pth, tmp_feature)

      frame_lbl = np.zeros(T)
      frame_lbl[rand_i] = 1

      pseudo_gt['abnormal_{}'.format(a_cnt)] = frame_lbl.tolist()

      a_cnt += 1

import json

save_pth = os.path.join(save_dir, '{}_pseudo_gt.json'.format(dataset_name))
with open(save_pth, 'w') as f:
    json.dump(pseudo_gt, f)
