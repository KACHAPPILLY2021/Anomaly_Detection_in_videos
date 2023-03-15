import torch.nn.functional as F
import torch

def sparsity(arr, lamda2):
  # loss = torch.mean(torch.norm(arr, dim=0))
  loss = arr.mean()
  return lamda2*loss


def smooth(arr, lamda1):
  arr2 = torch.zeros_like(arr)
  arr2[:, :-1] = arr[:, 1:]
  arr2[:, -1] = arr[:, -1]

  # loss = torch.sum((arr2-arr)**2)
  loss = ((arr2-arr)**2).sum(-1).mean()

  return lamda1*loss



def vae_loss(input_dict, w=1.0):
  x = input_dict['input']
  re_x = input_dict['result']
  mu = input_dict['mu']
  log_var = input_dict['log_var']
  recons_loss = F.mse_loss(x, re_x)
  kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
  return recons_loss + w * kld_loss

def rtfm_loss(score, label, abn_feat, nor_feat, margin=100, alpha=0.0001):

    abnormal_mask = label > 0
    normal_mask = ~abnormal_mask
    loss_cls = 0.0
    if abnormal_mask.sum() > 0:
        loss_cls += F.binary_cross_entropy(score[abnormal_mask], label[abnormal_mask])
    if normal_mask.sum() > 0:
        loss_cls += F.binary_cross_entropy(score[normal_mask], label[normal_mask])
    
    loss_abn = torch.abs(margin - torch.norm(abn_feat, p=2, dim=-1))**2
    loss_nor = torch.norm(nor_feat, p=2, dim=-1)**2
    
    loss_rtfm = loss_nor.mean() + loss_abn.mean()
    
    loss_total = loss_cls + alpha * loss_rtfm
    
    return loss_total

