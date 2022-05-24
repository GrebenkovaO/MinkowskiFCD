import numpy as np
import torch

#TODO: comment on purpose of this function
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

def get_statistics(dataset, num_features):
    feats_sum = np.zeros((num_features))
    feats_squared_sum = np.zeros((num_features))
    labelss = []
    
    for data in dataset:
        _, feats, labels = data
        feats_sum += np.mean(feats, axis=0)
        feats_squared_sum += np.mean(feats**2,  axis=0)
        labelss += list(labels)
        
    mean = feats_sum / len(dataset)
    # std = sqrt(E[X^2] - (E[X])^2)
    std = (feats_squared_sum / len(dataset) - mean ** 2) ** 0.5
    print(f'Mean={mean}, std={std}')
    
    weight1 = np.mean(labelss)
    weights = [weight1, 1 - weight1]
    print(f'Weights are {weights}')
    
    return torch.Tensor(mean).float(), torch.Tensor(std).float(), torch.Tensor(weights).float()