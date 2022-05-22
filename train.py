# -*- coding: utf-8 -*-
from comet_ml import Experiment
experiment = Experiment(
    api_key="KfIQxtLQwFBi7wahbWN9aCeav",
    project_name="FCD_Minkowski",
    workspace="grebenkovao", log_code = False)


import argparse
import numpy as np

import os
import sys
import torch
import random
import torch.optim as optim
from sklearn.metrics import recall_score
from urllib.request import urlretrieve
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join("/home/gbobrovskih/neurodata/Minkowski"))
from Minkowski.examples.minkunet import MinkUNet34C
import MinkowskiEngine as ME
import nibabel as nib
import pandas as pd
import json


# 1) Inference
# 4) Использовать фичи, внести в эксперимент
# 5) Использовать координаты как фичи, внести в эксперимент

# Plans for week:
# 1) Нормализовать интенсиваность
# 2) Проверить облака точек до и после функции хеширования

# Plans for future
# 1) Информация по мозгам. Соблюдение распределений. Имеет ли смысл?
#


def load_nii_to_array(nii_path):
    """
    Function returns np.array data from the *.nii file
    :params nii_path: str, path to *.nii file with data
    :outputs data: np.array,  data obtained from nii_path
    """

    try:
        data = np.asanyarray(nib.load(nii_path).dataobj)
        return (data)

    except OSError:
        print(FileNotFoundError(f'No such file or no access: {nii_path}'))
        return('')

class Brains(Dataset):
    def __init__(
        self, 
        task = 'train',
        data_dict = None):
        self.task = task
        if task == "train":
            self.data_dict = data_dict 
            
    def __len__(self):
        return len(self.data_dict['t1_brains'])
    
    def __getitem__(self, idx):
        single_data_dict = {key: self.data_dict[key][idx] for key in self.data_dict}
        for key in single_data_dict:
            single_data_dict[key] = load_nii_to_array(
                single_data_dict[key])
        size = single_data_dict['t1_brains'].shape
        grid_x, grid_y, grid_z = np.meshgrid(
            np.array(range(size[0])),
            np.array(range(size[1])),
            np.array(range(size[2])),
            indexing = 'ij'
        )
        point_cloud = np.concatenate(
            [
                np.expand_dims(grid_x,-1),
                np.expand_dims(grid_y,-1),
                np.expand_dims(grid_z,-1),
            ]
            + [
                np.expand_dims(single_data_dict[key],-1)
                for key in single_data_dict
                if key != "labels"
            ],
            -1,
        )
    
        point_cloud_fcd = point_cloud[
            (single_data_dict["labels"] == 1) & (single_data_dict["t1_brains"] > 0.01), :]
        
        pc_brain_without_fcd_noair = point_cloud[
            (single_data_dict["labels"] == 0) & (single_data_dict["t1_brains"] > 0.01), :]

        without_fcd_shape = pc_brain_without_fcd_noair.shape[0]
        without_fcd_noair_shape = pc_brain_without_fcd_noair.shape[0]

        pcd = np.concatenate([point_cloud_fcd, pc_brain_without_fcd_noair])
        
        random_idxs = np.random.choice(pcd.shape[0], size = 200000, replace = False)
        coords = pcd[:,:3][random_idxs] #n * 3
        feats = pcd[:,3:][random_idxs] #n * 4
        labels = np.array([1] * point_cloud_fcd.shape[0] + [0] * without_fcd_shape)[random_idxs] # n 
        
        coords, feats, labels = ME.utils.sparse_quantize(
             coordinates=coords,
             features=feats,
             labels=labels, 
            quantization_size = 1 
            )   
        
        '''mapping = ME.utils.sparse_quantize(
            coordinates=coords,
            return_index=True)'''
     
        return coords, feats, labels

def contrast_f(pred,label):
    p_in = np.sum(pred*label) / np.sum(label) # fcd true pred/ fcd real 
    p_out = np.sum(pred*(1 - label)) / np.sum(1-label) # fcd false / no fcd
    return (p_in - p_out) / (p_in + p_out)

def top10_f(pred,label,coords, crop_size = 32):
    """
        pred - (N,)
        label - (N,)
        coords = (N, 3)
    """
    df = pd.DataFrame({'pred':pred,'label':label,'coord1':coords[:,0],'coord2':coords[:,1],'coord3':coords[:,2]})
    
    for i in range(1,4):
        df[f'coord{i}'] = df[f'coord{i}'] // crop_size

    df = df.groupby([f'coord{i}' for i in range(1,4)]).mean().reset_index()
    df.label = 1 - df.label

    df = df.sort_values(['pred','label'], ascending = False).reset_index(drop = True).head(10)
    df = df[df.label!=1]
    if df.shape[0] == 0:
        top10 = 0.0
    else:
        top10 = 1 - float(df.head(1).index.values[0]) / 10
    return top10


def main(config, train_dict, test_dict):
    # 4 features, 3 coordinates, 2 outputs (binary segmentation)
    device = torch.device(config.device)
    net = MinkUNet34C(2, 2, D = 3) ### try 1 feature and all points 
    net = net.to(device)

    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    loader = Brains(data_dict = train_dict)
    feats_sum = np.zeros(len(FEATURES))
    feats_squared_sum =  np.zeros(len(FEATURES))
    num_batches = 0
    for data in loader:
        _, feats, _ = data
        feats_sum += np.mean(feats, axis=0)
        feats_squared_sum += np.mean(feats**2,  axis=0)
        num_batches += 1
    mean = feats_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (feats_squared_sum / num_batches - mean ** 2) ** 0.5
        
    '''train_iter = iter(loader)
    torch.mean(data, dim=[0,2,3])
    
    for i, data in enumerate(train_iter):
        coords, feats, labels = data
        max_temp = np.max(feats, axis = 1)
        min_temp = np.min(feats, axis = 1)
        for i in range(len(FEATURES)):
            if max_temp[i] > max_f[i]:
                max_f[i] = max_temp[i]
            if min_temp[i] < min_f[i]:
                min_f[i] = min_temp[i]
    
    print('max', max_f)
    print('min', min_f)'''
    print('MEAN', mean)
    print('STD', std)
   
    # Dataset, data loader
    train_dataset = Brains(data_dict = train_dict)
    
    # get weights for crossentropy
    labelss = []
    for data in train_dataset:
        _, _, labels = data
        labelss+=list(labels)
    weight1 = np.mean(labelss)
    weights = [weight1, 1 - weight1]
    print('Weights are', weights)
    criterion = torch.nn.CrossEntropyLoss(torch.tensor(weights).float().to(device))
    
    test_dataset = Brains(data_dict = test_dict)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=ME.utils.batch_sparse_collate,
        #num_workers=6
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=ME.utils.batch_sparse_collate,
        #num_workers=6
    )


    for epoch in range(config.max_epochs):
        accum_loss, accum_iter = 0, 0
        np.random.seed()
        train_iter = iter(train_dataloader)
        with experiment.train():
            # Training
            net.train()
            for i, data in enumerate(train_iter):
                coords, feats, labels = data
                coords = coords.to(device)
                labels = labels.to(device)
                '''for j in range(len(FEATURES)):
                    feats[j] = (feats[j] - min_f[j]) / (max_f[j] - min_f[j])'''
                for j in range(len(FEATURES)):
                    feats[j] = (feats[j] - mean[j]) / std[j]
                feats = feats.to(device)
                out = net(ME.SparseTensor(feats.float(), coords, device=device))
                optimizer.zero_grad()
                loss = criterion(out.F.squeeze(), labels.long().to(device))
                loss.backward()
                optimizer.step()

                accum_loss += loss.item()
                accum_iter += 1


            experiment.log_metric(name = 'loss', value = accum_loss / accum_iter, epoch=epoch)
        
        with experiment.test():
            #validation
            if epoch % 5 == 0:
                torch.save(net.state_dict(), 'saved_models/Model_t1_t2_numbers_200k_test')#'saved_models/Model_10brains_test_mapping')
                experiment.log_model("Model_full",  'saved_models/Model_t1_t2_numbers_200k_test')#'saved_models/Model_10brains_test_mapping')
                np.random.seed(42)
                accum_loss, accum_iter = 0, 0
                test_iter = iter(test_dataloader)
                net.eval()
                preds = []
                labelss = []
                coordss = []
                for i, data in enumerate(test_iter):
                    coords, feats, labels = data
                    coords = coords.to(device)
                    labels = labels.to(device)
                    '''for j in range(len(FEATURES)):
                        feats[j] = (feats[j] - min_f[j]) / (max_f[j] - min_f[j])'''
                    
                    for j in range(len(FEATURES)):
                        feats[j] = (feats[j] - mean[j]) / std[j]
                    feats = feats.to(device)
                    with torch.no_grad():
                        out = net(ME.SparseTensor(feats.float(), coords, device=device))
                        preds.append(out.F.softmax(dim = 1)[:,1].detach().cpu().numpy().reshape(-1))                       
                        labelss.append(labels.detach().cpu().numpy().reshape(-1))
                        coordss.append(coords.detach().cpu().numpy())
                        loss = criterion(out.F.squeeze(), labels.long().to(device))
                        '''if epoch % 395:
                            
                            out1 = out.F.softmax(dim = 1)[:,1].detach().cpu().numpy().reshape(-1).tolist()
                            coords1 = coords[:,1:].detach().cpu().numpy().tolist()
                            labels1 = labels.reshape(-1).detach().cpu().numpy().tolist()
                            result = {'coordinates': coords1,
                                      'predictions': out1,
                                      'labels': labels1
                                                         }
                            with open(f'predictions/noquant/mapping{i}-{epoch}.json', 'w') as file:
                                json.dump(result, file)'''
                        
                        
                        

                    accum_loss += loss.item()
                    accum_iter += 1
                experiment.log_metric(name = 'loss', value = accum_loss / accum_iter, epoch=epoch)
                num_test_brains = len(test_dataset)
                preds = np.concatenate(preds)
                labelss = np.concatenate(labelss)
#                 print(np.argwhere(labelss == 1)[0:3])
    #             coordss = np.concatenate(coordss, axis = 0)
                points_in_test_brains = len(preds)//num_test_brains
                top10 = [top10_f(preds[i*points_in_test_brains:(i+1)*points_in_test_brains],
                                       labelss[i*points_in_test_brains:(i+1)*points_in_test_brains],coordss[i]) for i in range(num_test_brains)] 
                contrast = contrast_f(preds,labelss)
                recall = recall_score(labelss, [1 if x > 0.5 else 0 for x in preds])
                contrasts = [contrast_f(preds[i*points_in_test_brains:(i+1)*points_in_test_brains],
                                        labelss[i*points_in_test_brains:(i+1)*points_in_test_brains]) for i in range(num_test_brains)]
                recalls = [recall_score(labelss[i*points_in_test_brains:(i+1)*points_in_test_brains],
                                        [1 if x > 0.5 else 0 for x in preds[i*points_in_test_brains:(i+1)*points_in_test_brains]]) for i in range(num_test_brains)]
                experiment.log_metric(name = 'top10 mean',value = np.mean(top10), epoch=epoch)
                experiment.log_metric(name = 'contrast', value = contrast, epoch=epoch)
                experiment.log_metric(name = 'contrast mean', value = np.nanmean(contrasts), epoch=epoch)
                experiment.log_metric(name = 'recall', value = recall, epoch=epoch)
                experiment.log_metric(name = 'recall mean', value = np.mean(recalls), epoch=epoch)
        
        if epoch % 50:
            torch.save(net.state_dict(), 'saved_models/Model_t1_t2_numbers_200k_test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
       



    config = parser.parse_args()
    
    PREFIX = '/code/PointCloudResNet/'
    BRAIN_TYPE = 'full'
    FEATURES = ['t1_brains', 't2_brains']
    path_to_data = f"{PREFIX}data"
    


    
    
    #path_to_allowed_subjects = f'{PREFIX}data/table_data/valid_preprocessed_data.csv'
    #allowed_subjects = np.load(path_to_allowed_subjects, allow_pickle=True).tolist()
    
   
    allowed_subjects = ['1', '2', '4', '5', '7', '8', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '26', '27', '28',
                         '29', '30', '31', '32', '34', '35', '36', '38', '39', '40', '41', '42', '44', '45', '46', '48', '49', '50', '51', '52', '53', '59',
                        '60', '61', '76', '82', '83']
    #allowed_subjects = ['15', '22', '31', '35', '44', '52', '59', '60', '61', '76']
    #allowed_subjects = ['60']
    train_dict = {}
    test_dict = {}
    for feature in FEATURES:
        train_dict[feature] = [f"{path_to_data}/{feature}/{subject}.nii.gz" for subject in allowed_subjects[:-10]]
        test_dict[feature] = [f"{path_to_data}/{feature}/{subject}.nii.gz" for subject in allowed_subjects[-10:]]
    train_dict['labels'] = [f"{path_to_data}/labels/{subject}.nii.gz" for subject in allowed_subjects[:-10]]
    test_dict['labels'] = [f"{path_to_data}/labels/{subject}.nii.gz"  for subject in allowed_subjects[-10:]]
    print(test_dict)
    
    '''for feature in FEATURES:
        train_dict[feature] = [f"{path_to_data}/{feature}/n21.nii"]
        test_dict[feature] = [f"{path_to_data}/{feature}/n21.nii" ]
    train_dict['labels'] = [f"{path_to_data}/labels/n21.nii"]
    test_dict['labels'] = [f"{path_to_data}/labels/n21.nii"]'''

    main(config, train_dict,test_dict)
