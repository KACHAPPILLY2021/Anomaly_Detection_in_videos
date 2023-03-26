import torch
import torch.utils.data as data
import numpy as np
import cv2
import csv
import os
from collections import defaultdict
from tqdm import tqdm
from scipy.ndimage import interpolation
from glob import glob
import json


class Dataset(data.Dataset):
  def __init__(self, data_root_pth, meta_root_pth, split='abnormal', ann_file=None, 
              mode='train', transform=False, pos_encoding=False, augment_pth=None, 
              pseudo_score_dict=None):
    assert mode in ['train', 'test']
    dataset_name = os.path.split(data_root_pth)[-1].split('_')[0]
    assert dataset_name in ['shanghaitech', 'ucf-crime']
    assert split in ['all', 'normal', 'abnormal']
    if mode == 'test': split = 'all'
    self.data_root_pth = data_root_pth
    self.split = split
    self.mode = mode
    self.transform = transform
    self.pos_encoding = pos_encoding
    self.dataset_name = dataset_name
    self.quantize_size = 32
    self._load_meta_files(meta_root_pth)
    if augment_pth is not None: 
        self.augment_pth = augment_pth
        self._augment(augment_pth)
    #self._load_pretrain_features(data_root_pth)
    if self.mode == 'test':
      assert ann_file is not None
      self._prepare_frame_level_labels(os.path.join(meta_root_pth, ann_file))

    self.pseudo_score_dict = pseudo_score_dict
    

  def _load_meta_files(self, meta_root_pth):
    self.video_ids = list()
    self.video_lbls = list()
    if self.dataset_name == 'shanghaitech': norm_start_id = 63
    elif self.dataset_name == 'ucf-crime': norm_start_id = 810

    with open(os.path.join(meta_root_pth, self.dataset_name+'.{}ing.csv'.format(self.mode))) as f:
      csvreader= csv.reader(f, delimiter=' ', quotechar='|')
      for i, row in enumerate(csvreader):
        if i == 0: continue
        video_id = row[0]
        feat_pth = os.path.join(self.data_root_pth, self.mode, '{}_i3d.npy'.format(video_id))
        if not os.path.exists(feat_pth): continue
        if self.split == 'all' or\
           (self.split == 'normal' and i > norm_start_id) or\
           (self.split == 'abnormal' and i <= norm_start_id):
            self.video_ids.append(video_id)

            if self.mode == 'train':
                if(i > norm_start_id): self.video_lbls.append(0)
                else: self.video_lbls.append(1)

    if self.mode == 'train':
        self.have_frame_lbl = [0] * len(self.video_lbls)
        self.frame_lbls = [np.zeros(self.quantize_size)] * len(self.video_lbls)  
  
  def _augment(self, augment_root):
    #augment_root = os.path.join(self.data_root_pth, 'augment/') 
    print('Use augmentation: ', augment_root)

    with open(os.path.join(augment_root, '{}_pseudo_gt.json'.format(self.dataset_name))) as fin:
      pseudo_gt = json.load(fin)

    l = glob(augment_root + '/*npy')
    for pth in l:
      vid = os.path.split(pth)[-1].split('.')[0]
      self.video_ids.append(vid)
      if vid[0] == 'n':
        self.video_lbls.append(0) 
      else:
        self.video_lbls.append(1) 

      self.have_frame_lbl.append(1)
      frame_lbl = pseudo_gt[vid]

      self.frame_lbls.append(np.array(frame_lbl))


      
  def _load_pretrain_features(self, data_root_pth):
    self.features = list()
    print('Load {} features'.format(self.mode))
    for vid_id in tqdm(self.video_ids):
      
      feat = np.load(os.path.join(data_root_pth, self.mode, '{}_i3d.npy'.format(vid_id)))
      self.features.append(feat)

  def _prepare_frame_level_labels(self, ann_file):
    import json
    with open(ann_file, 'r') as fin:
      db = json.load(fin)

    self.ground_truths = list()
    for video_id in self.video_ids:
      labels = db[video_id]['labels']
      self.ground_truths.append(np.array(labels))
    # self.ground_truths = np.concatenate(ground_truths)
    # self.ground_truths = ground_truths
  
  def __getitem__(self, index):
    features = self.features[index]

    if self.pos_encoding:
      T, _, C = features.shape
      pe = np.zeros([T, C])
      position = np.arange(T).reshape(-1,1)
      div_term = np.exp(np.arange(0, C, 2) * -(np.log(10000.0)/C))
      pe[:, 0::2] = np.sin(position * div_term)
      pe[:, 1::2] = np.cos(position * div_term)
      pe = np.expand_dims(pe, 1).astype(float)
      features = pe + features

    if self.mode == 'test':
      return features, self.ground_truths[index]
    if not self.transform:
      return features, self.video_lbls[index]
    t, n_group, channels = features.shape

    features = np.transpose(features, (2, 0, 1)) # C x tau x N

    # quantize each video to 32-snippet-length video
    width, height = self.quantize_size, channels
    features = cv2.resize(features, (width, height),
        interpolation=cv2.INTER_LINEAR) # CxTxN

    video = np.transpose(features, (2, 1, 0)) # NxTxC

    # global video statistics
    regular_labels = torch.tensor(0.0) # being normal video
    label = self.video_lbls[index]

    return video, label
  def __len__(self):
    return len(self.video_ids)

class Dataset_v2(Dataset):
  def __getitem__(self, index):
    #features = self.features[index]
    vid_id = self.video_ids[index]
    if vid_id[:6] == 'normal' or (vid_id[:8] == 'abnormal'): 
      features = np.load(os.path.join(self.augment_pth, '{}.npy'.format(vid_id)))
    else:
      features = np.load(os.path.join(self.data_root_pth, self.mode, '{}_i3d.npy'.format(vid_id)))

    if self.pos_encoding:
      T, _, C = features.shape
      pe = np.zeros([T, C])
      position = np.arange(T).reshape(-1,1)
      div_term = np.exp(np.arange(0, C, 2) * -(np.log(10000.0)/C))
      pe[:, 0::2] = np.sin(position * div_term)
      pe[:, 1::2] = np.cos(position * div_term)
      pe = np.expand_dims(pe, 1).astype(float)
      features = pe + features

    if self.mode == 'test':
      output = dict(
        features=features,
        gt=self.ground_truths[index]
      )
      return output

    if not self.transform:
      lbl = self.video_lbls[index]
      if self.pseudo_score_dict is not None and (lbl > 0):
          score = self.pseudo_score_dict[self.video_ids[index]]
          # You can change the threshold here
          mask = np.array(score) > 0.5
          features = features[mask]

      return features, lbl
    t, n_group, channels = features.shape

    features = np.transpose(features, (2, 0, 1)) # C x tau x N

    # quantize each video to 32-snippet-length video
    width, height = self.quantize_size, channels
    features = cv2.resize(features, (width, height),
        interpolation=cv2.INTER_LINEAR) # CxTxN

    video = np.transpose(features, (2, 1, 0)) # NxTxC

    # global video statistics
    regular_labels = torch.tensor(0.0) # being normal video
    label = self.video_lbls[index]

    frame_lbl = self.frame_lbls[index]
    have_frame_lbl = self.have_frame_lbl[index]
    if have_frame_lbl:
      z = self.quantize_size/len(frame_lbl)
      res_frame_lbl = interpolation.zoom(frame_lbl, z, order=1)
      assert len(res_frame_lbl) == self.quantize_size
      frame_lbl = res_frame_lbl

    output = dict(
      features=video,
      video_lbl=label,
      frame_lbl=frame_lbl,
      have_frame_lbl=have_frame_lbl
    )

    return output

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_dataset = Dataset_v2('./ucf-crime_i3d',
                               './S3R/data/ucf-crime',
                               transform=True)
    val_dataset = Dataset_v2('./ucf-crime_i3d',
                               './S3R/data/ucf-crime',
                               'ucf-crime_ground_truth.testing.json', 'test')
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for batch in train_dataloader:
        print(batch['features'].shape, batch['frame_lbl'].shape)
