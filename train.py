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

sys.path.append(os.path.join('/home/neurodata/Minkowski'))
# from examples.minkunet import MinkUNet34C
from model import MinkUNet34CAttention
import MinkowskiEngine as ME
import nibabel as nib

from dataset import Brains
from utils import top10_f, contrast_f, get_statistics, get_model_size

# full dataset precomputed statistics
MEAN2 = [304.884, 290.758]
MEAN3 = [304.86774503, 290.75134592, 142.4693862]
STD2 = [86.335, 127.093]
STD3 = [86.35006108, 127.1242876, 48.30014252]
WEIGHTS = [0.0015, 0.998]

def main(config, train_dict, test_dict):
    # 4 features, 3 coordinates, 2 outputs (binary segmentation)
    device = torch.device(config.device)
    
    net = MinkUNet34CAttention(len(FEATURES), 2, D = 3, attention=bool(config.attention)) ### try 1 feature and all points 
    net = net.to(device)
    print(get_model_size(net))

    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    train_dataset = Brains(data_dict=train_dict, num_points=config.num_points)
    # get mean, std and weights for crossentropy
    if not config.compute_statistics:
        if len(FEATURES) == 2:
            mean = torch.Tensor(MEAN2).float()
            std = torch.Tensor(STD2).float()
            weights = torch.Tensor(WEIGHTS).float()
        elif len(FEATURES) == 3:
            mean = torch.Tensor(MEAN3).float()
            std = torch.Tensor(STD3).float()
    else:
        mean, std, weights = get_statistics(train_dataset, num_features=len(FEATURES))
    
    print('Weights are', weights)
    criterion = torch.nn.CrossEntropyLoss(weights.to(device))
    
    # Dataset, data loader
    test_dataset = Brains(data_dict=test_dict)
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

        #TODO: random seed: what for?
        np.random.seed()
        train_iter = iter(train_dataloader)
        with experiment.train():
            # Training
            net.train()
            for i, data in enumerate(train_dataloader):
                coords, feats, labels = data
                coords = coords.to(device)
                labels = labels.to(device)
                
                # normalize features
                feats -= mean.view(1, len(FEATURES)) / std.view(1, len(FEATURES))
                
                feats = feats.to(device)
                out = net(ME.SparseTensor(feats.float(), coords, device=device))
                optimizer.zero_grad()
                loss = criterion(out.F.squeeze(), labels.long().to(device))
                
                if i % config.print_each_step == 0:
                    print(f"Epoch {epoch} Step {i}/{len(train_dataloader)}: Train Loss is {loss.item()}")
                loss.backward()
                optimizer.step()

                accum_loss += loss.item()
                accum_iter += 1
                
            if config.log:
                experiment.log_metric(name = 'loss', value = accum_loss / accum_iter, epoch=epoch)
            print(f'Epoch {epoch}: Mean Train loss is {accum_loss / accum_iter}')
            
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
                    
                    # normalize features
                    feats -= mean.view(1, len(FEATURES)) / std.view(1, len(FEATURES))
                    
                    feats = feats.to(device)
                    with torch.no_grad():
                        out = net(ME.SparseTensor(feats.float(), coords, device=device))
                        preds.append(out.F.softmax(dim = 1)[:,1].detach().cpu().numpy().reshape(-1))                       
                        labelss.append(labels.detach().cpu().numpy().reshape(-1))
                        coordss.append(coords.detach().cpu().numpy())
                        loss = criterion(out.F.squeeze(), labels.long().to(device))
                    
                    accum_loss += loss.item()
                    accum_iter += 1
                if config.log:
                    experiment.log_metric(name = 'loss', value = accum_loss / accum_iter, epoch=epoch)
                print(f'Epoch {epoch}: Mean Validation loss is {accum_loss / accum_iter}')
                
                num_test_brains = len(test_dataset)
                preds = np.concatenate(preds)
                labelss = np.concatenate(labelss)
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
    parser.add_argument('--num_points', type=int, default=200000)
    parser.add_argument('--compute_statistics', type=int, default=0)
    parser.add_argument('--print_each_step', type=int, default=5)
    parser.add_argument('--attention', type=int, default=0)
    parser.add_argument('--log', type=int, default=1) 

    config = parser.parse_args()
    
    PREFIX = '/home/neurodata/'
    BRAIN_TYPE = 'full'
    FEATURES = ['t1_brains', 't2_brains', 'flair_brains']
    path_to_data = f"{PREFIX}data"
    
    #path_to_allowed_subjects = f'{PREFIX}data/table_data/valid_preprocessed_data.csv'
    #allowed_subjects = np.load(path_to_allowed_subjects, allow_pickle=True).tolist()
   
    allowed_subjects = ["1", "4", "5", "6", "7", "8", "9", "10", "11", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "25", "26", "27", "28", "29", "32", "35", "36", "37", "39", "40", "41", "42", "45", "47", "48", "49", "50", "51", "52", "53", "54", "55", "57", "58", "59", "60", "61", "61", "76", "83", "2", "3", "23", "24", "30", "31", "33", "34", "38", "43", "44", "46", "82"]
    train_dict = {}
    test_dict = {}
    for feature in FEATURES:
        train_dict[feature] = [f"{path_to_data}/{feature}/{subject}.nii.gz" for subject in allowed_subjects[:-13]]
        test_dict[feature] = [f"{path_to_data}/{feature}/{subject}.nii.gz" for subject in allowed_subjects[-13:]]
    train_dict['labels'] = [f"{path_to_data}/labels/{subject}.nii.gz" for subject in allowed_subjects[:-13]]
    test_dict['labels'] = [f"{path_to_data}/labels/{subject}.nii.gz"  for subject in allowed_subjects[-13:]]
    print(test_dict)
    print(f'Features: {FEATURES}')

    main(config, train_dict,test_dict)