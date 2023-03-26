import numpy as np
import matplotlib.pyplot as plt
import csv

from dataset import Dataset, Dataset_v2
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss import *
import torch
from vae import VAE, ShareDecoderVae
import os
from glob import glob
import json

split = 'all'

#dataset_name = 'ucf-crime'
dataset_name = 'shanghaitech'
assert dataset_name in ['shanghaitech', 'ucf-crime']

dataset_root = ''

abnormal_res_dict = None
with open(os.path.join(dataset_root, '{0}/pred-{0}.json'.format(dataset_name))) as f:
    abnormal_res_dict = json.load(f)

# Please specify your project root path
proj_root = ''
i3d_pth = os.path.join(proj_root, 'dataset', dataset_name, 'i3d')
meta_pth = os.path.join(proj_root, 'dataset', dataset_name)

train_dataset = Dataset_v2(i3d_pth,
                           meta_pth,
                           split=split, pseudo_score_dict=abnormal_res_dict)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)


save_dir = "" # Put your path to save the configurations and checkpoints
os.mkdir(save_dir)


bs = 1024
lr = 1e-4
ttl_epoch = 100
kld_w= 0.00025


hidden_dims = [1024,512]
latent_dim = 256

# Save the configurations of your model
config = dict(hidden_dims=hidden_dims, latent_dim=latent_dim)
with open(os.path.join(save_dir, 'config.json'), 'w') as f:
    json.dump(config, f)

model = ShareDecoderVae(2048, latent_dim, hidden_dims, dropout=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


start_epoch = 0
model = model.cuda()

abnormal_res_dict = None

for epoch in range(start_epoch, ttl_epoch):

    losses = 0.0
    cnt = 0
    for i, (features, label) in enumerate(tqdm(train_dataloader)):


        feature_flat = features.reshape(-1, features.shape[-1]).cpu()

        if feature_flat.shape[0] == 0: continue
        elif feature_flat.shape[0] > bs:
            rand_idx = np.random.choice(feature_flat.shape[0], bs, replace=False)
            feature_flat = feature_flat[rand_idx]
        
        optimizer.zero_grad()
        output = model(feature_flat.cuda())
        output['result'] = output['result'][int(label)]
        loss = vae_loss(output, kld_w)
        if torch.isnan(loss):
            #print(label, feature_flat.shape)
            continue
        loss.backward()
        optimizer.step()
        losses += loss.item()
        cnt += 1


    print_text = 'Epoch: {}'.format(epoch+1)
    print_text += ' Loss: {}'.format(losses/cnt)

    print(print_text)
    if (epoch+1)%5 == 0:
        scheduler.step()
        torch.save(model.cpu().state_dict(), os.path.join(save_dir, 'epoch_{}.pth'.format(epoch+1)))
        model = model.cuda()
