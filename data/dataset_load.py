import os
from torch.utils.data import Dataset
from data.preprocessing import load, augmentation, augmentation_test, load_fast
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class BraTSDataset(Dataset):
    """BraTS Dataset""" 
    def __init__(self,
                 dataset_path,
                 mode='train',
                 debug_mode=False,
                 ):

        self.mode = mode
        self.dataset_path = os.path.join(dataset_path, mode)
        img_names = [d for d in os.listdir(self.dataset_path) if d.endswith('h5')]
        # print(img_names)
        img_names.sort()
        self.data_path = [[os.path.join(self.dataset_path, name), name] for name in img_names]
        # self.data_path = self.data_path[:2]
        if debug_mode:
            self.data_path = self.data_path[:2]
        # print(self.data_path)
        # input()
        # exit()
        total_files = len(self.data_path)
        slices_list = [None]*total_files

        with ProcessPoolExecutor(max_workers=4) as executor:
            for idx, batch in enumerate(tqdm(
                executor.map(self.load_and_move, self.data_path),
                total=total_files,
                desc=f"Loading {mode}",
                colour='yellow'
            )):
                slices_list[idx] = batch
        total_slices_number = sum([s.shape[0] for s in slices_list])
        print(f"Total slices number: {total_slices_number}")
        self.slices = [None]*total_slices_number
        start_idx = 0
        for idx, batch in enumerate(slices_list):
            batch_size = batch.shape[0]
            self.slices[start_idx:start_idx+batch_size] = batch
            start_idx += batch_size


    def load_and_move(self, path):
        arr = load(path)
        return np.moveaxis(arr, -1, 0)    
    
    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
    
        sample = {'image':self.slices[idx]}

        if self.mode == 'test':
            output = augmentation_test(sample)
        else:
            output = augmentation(sample)  
        return output


class IXIDataset(Dataset):
    """IXI Dataset""" 
    def __init__(self,
                 dataset_path,
                 mode='train',
                 debug_mode=False,
                 ):

        self.mode = mode
        self.dataset_path = os.path.join(dataset_path, mode)
        img_names = [d for d in os.listdir(self.dataset_path) if d.endswith('h5')]
        img_names.sort()
        self.data_path = [[os.path.join(self.dataset_path, name), name] for name in img_names]
        if debug_mode:
            self.data_path = self.data_path[:2]
        total_files = len(self.data_path)
        slices_list = [None]*total_files
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            for idx, batch in enumerate(tqdm(
                executor.map(self.load_and_move, self.data_path),
                total=total_files,
                desc=f"Loading {mode}",
                colour='yellow'
            )):
                slices_list[idx] = batch
        total_slices_number = sum([s.shape[0] for s in slices_list])
        print(f"Total slices number: {total_slices_number}")
        self.slices = [None]*total_slices_number
        start_idx = 0
        for idx, batch in enumerate(slices_list):
            batch_size = batch.shape[0]
            self.slices[start_idx:start_idx+batch_size] = batch
            start_idx += batch_size
            
    def __len__(self):
        return len(self.slices)
    
    def load_and_move(self, path):
        arr = load(path)
        return np.moveaxis(arr, -1, 0)

    def __getitem__(self, idx):
        
        sample = {'image':self.slices[idx]}
        
        if self.mode == 'test':
            output = augmentation_test(sample, ixi=True)
        else:
            output = augmentation(sample, ixi=True)
        return output