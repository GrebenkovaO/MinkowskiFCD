import os
import sys
import numpy as np
import nibabel as nib
sys.path.append(os.path.join('/home/neurodata/Minkowski'))
import MinkowskiEngine as ME
from torch.utils.data import Dataset, DataLoader

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
        data_dict = None,
        num_points=200000):
        self.task = task
        if task == 'train':
            self.data_dict = data_dict 
        self.num_points = num_points
            
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
        
        mask = lambda a: (single_data_dict["labels"] == a) & (single_data_dict["t1_brains"] > 0.01)
        point_cloud_fcd = point_cloud[mask(1), :]
        
        pc_brain_without_fcd_noair = point_cloud[mask(0), :]

        without_fcd_shape = pc_brain_without_fcd_noair.shape[0]
        without_fcd_noair_shape = pc_brain_without_fcd_noair.shape[0]

        pcd = np.concatenate([point_cloud_fcd, pc_brain_without_fcd_noair])
        
        random_idxs = np.random.choice(pcd.shape[0], size = self.num_points, replace = False)
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
