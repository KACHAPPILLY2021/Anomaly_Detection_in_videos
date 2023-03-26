import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch import Tensor, select
from typing import Union

class _NonLocalBlockND(nn.Module):
  def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
    super(_NonLocalBlockND, self).__init__()

    assert dimension in [1, 2, 3]

    self.dimension = dimension
    self.sub_sample = sub_sample

    self.in_channels = in_channels
    self.inter_channels = inter_channels

    if self.inter_channels is None:
      self.inter_channels = in_channels // 2
      if self.inter_channels == 0:
        self.inter_channels = 1

    if dimension == 3:
      conv_nd = nn.Conv3d
      max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
      bn = nn.BatchNorm3d
    elif dimension == 2:
      conv_nd = nn.Conv2d
      max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
      bn = nn.BatchNorm2d
    else:
      conv_nd = nn.Conv1d
      max_pool_layer = nn.MaxPool1d(kernel_size=(2))
      bn = nn.BatchNorm1d

    self.value = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0) # value

    if bn_layer:
      self.alter = nn.Sequential(
          conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                  kernel_size=1, stride=1, padding=0),
          bn(self.in_channels)
      ) # output
      nn.init.constant_(self.alter[1].weight, 0)
      nn.init.constant_(self.alter[1].bias, 0)
    else:
      self.alter = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0)
      nn.init.constant_(self.alter.weight, 0)
      nn.init.constant_(self.alter.bias, 0)

    self.query = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                          kernel_size=1, stride=1, padding=0) # query

    self.key = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0) # key

    if sub_sample: # default = False
      self.value = nn.Sequential(self.value, max_pool_layer)
      self.key = nn.Sequential(self.key, max_pool_layer)

  def forward(self, x: Tensor, return_nl_map: bool=False):
    """
    :param x: BCT
    :param return_nl_map: if True return z, nl_map, else only return z.
    :return:
    """

    identity = x

    B, C, T = x.shape
    D = self.inter_channels

    value = self.value(x).view(B, D, -1) # BDT
    value = value.transpose(-2, -1) # BTD

    query = self.query(x).view(B, D, -1) # BDT
    query = query.transpose(-2, -1) # BTD
    key = self.key(x).view(B, D, -1) # BDT

    attn = query @ key # BTT
    attn = attn / T

    out = torch.matmul(attn, value)
    out = out.permute(0, 2, 1).contiguous()
    out = out.view(B, self.inter_channels, *x.size()[2:])
    out = self.alter(out)
    out = out + identity

    if return_nl_map:
      return out, attn
    return out


class NonLocalBlock1D(_NonLocalBlockND):
  def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
    super(NonLocalBlock1D, self).__init__(in_channels,
                                          inter_channels=inter_channels,
                                          dimension=1, sub_sample=sub_sample,
                                          bn_layer=bn_layer)


def weight_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 or classname.find('Linear') != -1:
    torch_init.xavier_uniform_(m.weight)
    if m.bias is not None:
      m.bias.data.fill_(0)


class Aggregate(nn.Module):
  def __init__(
      self,
      dim: int=2048,
      reduction: int=4,
  ):
    super(Aggregate, self).__init__()

    dim_inner = dim // reduction

    bn = nn.BatchNorm1d
    self.dim = dim

    self.conv_1 = nn.Sequential(
        nn.Conv1d(dim, dim_inner, kernel_size = 3,
            stride = 1, dilation = 1, padding = 1),
        nn.GroupNorm(num_groups = 8, num_channels = dim_inner, eps=1e-05),
        nn.ReLU())
    self.conv_2 = nn.Sequential(
        nn.Conv1d(dim, dim_inner, kernel_size=3,
            stride = 1, dilation = 2, padding = 2),
        nn.GroupNorm(num_groups = 8, num_channels = dim_inner, eps=1e-05),
        nn.ReLU())
    self.conv_3 = nn.Sequential(
        nn.Conv1d(dim, dim_inner, kernel_size = 3,
            stride = 1, dilation = 4, padding = 4),
        nn.GroupNorm(num_groups = 8, num_channels = dim_inner, eps=1e-05),
        nn.ReLU())
    self.conv_4 = nn.Sequential(
        nn.Conv1d(dim, dim_inner, kernel_size = 1,
            stride = 1, padding = 0, bias = False),
        nn.ReLU(),
    )
    self.conv_5 = nn.Sequential(
        nn.Conv1d(dim, dim, kernel_size = 3,
            stride = 1, padding = 1, bias = False), # TODO: should we keep the bias?
        nn.GroupNorm(num_groups = 8, num_channels = dim, eps=1e-05),
        nn.ReLU())

    self.non_local = NonLocalBlock1D(dim_inner, sub_sample=False, bn_layer=True)


  def forward(self, x: Tensor):

    x: Tensor # input feature of shape BTC

    out = x.transpose(-2, -1) # BCT
    residual = out

    out1 = self.conv_1(out)
    out2 = self.conv_2(out)

    out3 = self.conv_3(out)
    out_d = torch.cat((out1, out2, out3), dim = 1)
    out = self.conv_4(out)
    out = self.non_local(out)
    out = torch.cat((out_d, out), dim=1)
    out = self.conv_5(out)   # fuse all the features together
    out = out + residual # BCT

    out = out.transpose(-2, -1) # BTC

    return out


class baseline_model(nn.Module):
  def __init__(self, dim: int = 2048, 
                dropout: float = 0.7, k: int =3, 
                use_mst: bool = False):
    super(baseline_model, self).__init__()

    self.video_encoder = None
    if use_mst:
      self.video_encoder = nn.Sequential(
        Aggregate(dim), nn.Dropout(dropout))
    self.video_classifier = nn.Sequential(
      nn.Linear(dim, dim // 4),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(dim // 4, dim // 16),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(dim // 16, 1),
      nn.Sigmoid())
    self.k = k
  def forward(self, video: Tensor):
    B, N, T, C = video.shape
    video = video.reshape(-1, T, C)
    if self.video_encoder is not None:
      video = self.video_encoder(video)
    video_score = self.video_classifier(video)
    video_score = video_score.reshape(B, N, -1).mean(1) # B, T
    video_max_topk_score = video_score.topk(self.k, -1)[0].mean(-1)
    # video_score_ = video_score.sort(-1)[0] # B, T
    # video_min_topk_score = video_score_[:,:self.k].mean(-1) # B, 
    # video_max_topk_score = video_score_[:,-self.k:].mean(-1) # B,
    output = {
      'snippet_score': video_score,
      'topk_score': video_max_topk_score,
    }
    return output

class conv_baseline_model(nn.Module):
  def __init__(self, dim: int = 2048, 
                dropout: float = 0.7, k: int =3, 
                use_mst: bool = False):
    super(conv_baseline_model, self).__init__()

    self.video_encoder = None
    if use_mst:
      self.video_encoder = nn.Sequential(
        Aggregate(dim), nn.Dropout(dropout))
    # self.video_classifier = nn.Sequential(
    #   nn.Linear(dim, dim // 4),
    #   nn.ReLU(),
    #   nn.Dropout(dropout),
    #   nn.Linear(dim // 4, dim // 16),
    #   nn.ReLU(),
    #   nn.Dropout(dropout),
    #   nn.Linear(dim // 16, 1),
    #   nn.Sigmoid())
    self.video_classifier = nn.Sequential(
      nn.BatchNorm1d(dim),
      nn.Conv1d(dim, dim//4, 3, padding=1),
      nn.ReLU(),
      nn.BatchNorm1d(dim//4),
      nn.Conv1d(dim//4, dim//16, 3, padding=1),
      nn.ReLU(),
  
    )
    self.score_pred = nn.Sequential(
      nn.Linear(dim //16, 1),
      nn.Sigmoid()
    )
    self.k = k
  def forward(self, video: Tensor):
    B, N, T, C = video.shape
    video = video.reshape(-1, T, C)
    if self.video_encoder is not None:
      video = self.video_encoder(video)

    video = video.permute(0,2,1)
    video_score = self.video_classifier(video)
    video_score = self.score_pred(video_score.permute(0,2,1))
    
    video_score = video_score.reshape(B, N, -1).mean(1) # B, T
    # video_max_topk_score = video_score.topk(self.k, -1)[0].mean(-1)
    video_score_ = video_score.sort(-1)[0] # B, T
    video_min_topk_score = video_score[:,:self.k].mean(-1) # B, 
    video_max_topk_score = video_score[:,-self.k:].mean(-1) # B,
    output = {
      'snippet_score': video_score,
      'topk_score': video_max_topk_score,
    }
    return output



class rtfm_model(nn.Module):
  def __init__(self, dim: int = 2048, 
                dropout: float = 0.7, k: int =3, 
                use_mst: bool = False):
    super(rtfm_model, self).__init__()

    self.video_encoder = None
    self.dropout = dropout
    if use_mst:
      self.video_embedding = Aggregate(dim)
    
    else:
      self.video_embedding = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim, dim)
      )

    self.video_classifier = nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(dim, dim // 4),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(dim // 4, dim // 16),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(dim // 16, 1),
      nn.Sigmoid())
    self.k = k

  def forward(self, 
              video_feat: Tensor, 
              labels: Union[Tensor, None] = None):
    output = {}
    B, N, T, C = video_feat.shape
    video_feat = video_feat.reshape(-1, T, C)
    # if self.video_encoder is not None:
    #   video_feat = self.video_encoder(video_feat)
    
    video_feat = self.video_embedding(video_feat) # B*N, T, C
    video_score = self.video_classifier(video_feat)
    video_score = video_score.view(B, N, -1).mean(1) # B, T
    output['snippet_score'] = video_score 
    if labels is None: return output # Testing

    video_feat = video_feat.view(B, N, T, -1)
    feat_mag = torch.norm(video_feat, p=2, dim=-1) # B, N, T
    feat_mag = feat_mag.mean(1) # B, T

    rand_mask = (torch.rand_like(feat_mag) < self.dropout).float()
    feat_mag = feat_mag * rand_mask
    _, topk_id = feat_mag.topk(self.k, -1) # B, k

    topk_score = torch.gather(video_score, -1, topk_id).mean(-1)
    output['topk_score'] = topk_score

    video_feat = video_feat.view(B, N, T, -1)
    video_feat = video_feat.permute(1,0,2,3) # N, B, T, C

    abn_mask = labels > 0 # B
    nor_mask = ~abn_mask

    topk_id = topk_id.unsqueeze(2).expand(-1, -1, C)

    select_abn_feature = torch.zeros(0, device=video_feat.device)
    select_nor_feature = torch.zeros(0, device=video_feat.device)
    for feat in video_feat:

      select_feat = torch.gather(feat, 1, topk_id) # B, K, C
      if abn_mask.sum() > 0:
        abn_feat = select_feat[abn_mask] 
        select_abn_feature = torch.cat([select_abn_feature, abn_feat])
      if nor_mask.sum() > 0:
        nor_feat = select_feat[nor_mask] 
        select_nor_feature = torch.cat([select_nor_feature, nor_feat])
    
    if select_abn_feature.shape[0] > 0: 
      select_abn_feature = select_abn_feature.mean(1) 
    if select_nor_feature.shape[0] > 0:
      select_nor_feature = select_nor_feature.mean(1)
    
  
    
    output['select_abn_feature'] = select_abn_feature
    output['select_nor_feature'] = select_nor_feature
    
    return output
