import numpy as np
import matplotlib.pyplot as plt
import csv

from dataset import Dataset, Dataset_v2
from torch.utils.data import DataLoader

from vae import VAE
import torch
from tqdm import tqdm

from test import validation

from model import rtfm_model, baseline_model
import torch.nn as nn
import torch
from loss import rtfm_loss, smooth, sparsity
import torch.nn.functional as F
import json


def update_lr(optim, epoch, lr):
  if epoch >= 50 and (epoch < 100):
    for g in optim.param_groups:
      g['lr'] = lr*0.1
  elif epoch >= 100:
    for g in optim.param_groups:
      g['lr'] = lr*0.01

def train(model, optimizer, train_dataloader, val_dataloader, total_epoch):
    best_score = 0.0
    best_res = None
    best_model = None
    
    for epoch in range(total_epoch):
      ttl_loss = 0
      #update_lr(optimizer, epoch, lr)
      for batch in tqdm(train_dataloader):
    
    
        optimizer.zero_grad()
        if model_type == 'RTFM':
            output = model(batch['features'].float().cuda(), batch['video_lbl'].cuda())
        else:
            output = model(batch['features'].float().cuda())
        video_score = output['snippet_score']
        topk_score = output['topk_score']
    
    
        batch_labels = batch['video_lbl'].type(video_score.dtype).to(video_score.device)
    
        anomaly_mask = batch_labels == 1
    
        if anomaly_mask.sum() > 0:
          abn_scores = video_score[anomaly_mask]
          loss_sparse = sparsity(abn_scores, 8e-3)
          loss_smooth = smooth(abn_scores, 8e-4)
        else:
          loss_sparse = 0.0
          loss_smooth = 0.0
    
        mask = batch['have_frame_lbl'].bool().cuda()
    
        if model_type == 'baseline':
            loss = F.binary_cross_entropy(topk_score, batch_labels)
        elif model_type == 'RTFM':
            loss = rtfm_loss(topk_score,
                            batch_labels,
                            output['select_abn_feature'],
                            output['select_nor_feature'],
                            alpha=0.0001,
                            )
    
        # loss += (loss_sparse + loss_smooth)
    
        if mask.sum() > 0:
          loss += 0.1*F.binary_cross_entropy(video_score[mask], batch['frame_lbl'][mask].cuda().float())
    
    
        loss.backward()
        optimizer.step()
        ttl_loss += loss.item()
    
      ttl_loss /= len(train_dataloader)
      print('Epoch {}'.format(epoch+1), ttl_loss)
    
      model.eval()
      res = validation(val_dataloader, model, val_dataset.ground_truths)
      model.train()
      score = res['score']
      print('Validation score:', score)
      if score > best_score:
        best_score = score
        best_res = res
        best_model = model.cpu().state_dict()
        model = model.cuda()
    
    return best_res, best_model



#dataset_name = 'ucf-crime'
dataset_name = 'shanghaitech'
assert dataset_name in ['shanghaitech', 'ucf-crime']

augment_type = 0

#augment_path = './exps/{}_sVAE_{}/augment'.format(dataset_name, augment_type)

# Please specify your project root path
proj_root = ''
i3d_pth = os.path.join(proj_root, 'dataset', dataset_name, 'i3d')
meta_pth = os.path.join(proj_root, 'dataset', dataset_name)

# Specify your experiment name
exp_name = ''
exp_pth = os.path.join(proj_root, 'exp', exp_name)
if not os.path.isdir(exp_pth):
    os.mkdir(exp_pth)

# Your path to the generated psuedo features
augment_pth = os.path.join(proj_root, 'dataset', 'augment')



train_dataset = Dataset_v2(i3d_pth, 
                           meta_pth, 
                           split='all', transform=True, augment_pth=augment_pth)
 


val_dataset = Dataset_v2(i3d_pth,
                         meta_pth, 'all',
                         '{}_ground_truth.testing.json'.format(dataset_name), 'test')


#model_type = 'RTFM'
model_type = 'baseline'
assert model_type in ['baseline', 'RTFM']


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)


total_epoch = 200
lr = 1e-3


if model_type == 'baseline':
    model = baseline_model()
elif model_type == 'RTFM':
    model = rtfm_model(use_mst=True)
else:
    raise Exception('No model called {}'.format(model_type))

# model.load_state_dict(torch.load('models/bl_pretrain.pth'))

model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


best_res, best_model = train(model, optimizer, train_dataloader, val_dataloader, total_epoch)

print('Best Score', best_res['score'])

with open(os.path.join(exp_pth, 'best_res.json'), 'w') as f:
    json.dump(best_res, f)

torch.save(best_model, os.path.join(exp_pth, 'best_ckpt.pth'))
