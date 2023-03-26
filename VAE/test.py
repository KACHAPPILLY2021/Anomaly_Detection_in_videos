from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from dataset import Dataset, Dataset_v2
from torch.utils.data import DataLoader
from model import baseline_model

def validation(val_dataloader, model, id_list=None):

  pred_video_scores = []
  gts = []

  res_dict = {}

  for i, batch in enumerate(tqdm(val_dataloader)):
    features = batch['features']
    labels = batch['gt']

    features = features.permute(0, 2, 1, 3)
    # print(features.shape)
    with torch.no_grad():
      output = model(features.cuda())
      video_score = output['snippet_score']

    video_score = video_score[0].cpu().detach().numpy()
    video_score = np.repeat(video_score, 16)
    pred_video_scores.append(video_score)

    if id_list is not None:
        res_dict[id_list[i]] = video_score

    gts.append(labels[0].numpy())

  preds = np.concatenate(pred_video_scores, 0)
  # pred = np.repeat(pred_video_scores, 16)
  gts = np.concatenate(gts, 0)
  fpr, tpr, threshold = roc_curve(gts, preds)

  rec_auc = auc(fpr, tpr)
  score = rec_auc
  # print(score)
  out = dict(score=score, fpr=fpr.tolist(), tpr=tpr.tolist(), pred=res_dict)

  return out


def vae_validation(val_dataloader, model, gt):

  pred_video_scores = []
  gts = []
  # bl_model.cuda()
  for batch in tqdm(val_dataloader):
    features = batch['features']
    labels = batch['gt']

    features = features.squeeze()
    ori_shape = features.shape

    features = features.view(-1, ori_shape[-1]).cuda()

    with torch.no_grad():
      output = model(features.cuda())

    res1 = output['result'][0]
    res2 = output['result'][1]

    res1 = res1.view(ori_shape)
    res2 = res2.view(ori_shape)

    features = features.view(ori_shape)

    normal_score = torch.norm(features - res1, p=2, dim=-1).mean(-1).unsqueeze(-1)
    abnormal_score = torch.norm(features - res2, p=2, dim=-1).mean(-1).unsqueeze(-1)
    score = torch.cat([normal_score, abnormal_score], -1)
    score = F.softmax(-score, -1)[:,-1]


    video_score = score.cpu().detach().numpy()
    video_score = np.repeat(video_score, 16)
    pred_video_scores.append(video_score)


    gts.append(labels[0].numpy())

  preds = np.concatenate(pred_video_scores, 0)
  # pred = np.repeat(pred_video_scores, 16)
  gts = np.concatenate(gts, 0)
  fpr, tpr, threshold = roc_curve(gts, preds)

  rec_auc = auc(fpr, tpr)
  score = rec_auc
  # print(score)
  return score


if __name__ == '__main__':
    
    #dataset_name = 'ucf-crime'
    dataset_name = 'shanghaitech'
    assert dataset_name in ['shanghaitech', 'ucf-crime']
    
    # Please specify your project root path
    proj_root = ''
    i3d_pth = os.path.join(proj_root, 'dataset', dataset_name, 'i3d')
    meta_pth = os.path.join(proj_root, 'dataset', dataset_name)
    
    
    # Load trained model
    # Specify checkpoint path here
    model_pth = ''

    model = baseline_model()
    model.load_state_dict(torch.load(model_pth))
    
    train_dataset = Dataset_v2(i3d_pth, 
                               meta_pth, 
                               split='all')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

     
    id_list = train_dataset.video_ids  
    out = validation(train_dataloader, model, id_list) 

    pred = out['pred']
    
    # Save the psuedo 'snippet-level' score
    save_dir = ''
    with open(os.path.join(save_dir, 'pred-{}.json'.format(dataset_name)), 'w') as f:
        json.dump(pred, f)

    
