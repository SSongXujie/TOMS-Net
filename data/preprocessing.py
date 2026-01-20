import hdf5plugin  
import numpy as np
import h5py
import torch
import random
from torchvision.transforms import transforms

def load(data_path):
    image_path = data_path[0]

    if 'h5' in image_path:
        f = h5py.File(image_path, 'r')
        image_numpy = np.array(f.get('vol'), dtype = np.float32)
    return image_numpy

def load_fast(path, use_chunk_cache_bytes=256*1024*1024):
    """Efficiently read /vol/ dataset from HDF5 file as float32 ndarray."""
    path = path[0]
    with h5py.File(path, 'r', libver='latest', rdcc_nbytes=use_chunk_cache_bytes) as f:
        ds = f['vol']
        if ds.dtype == np.float32:
            return ds[...]           
        out = np.empty(ds.shape, dtype=np.float32)
        ds.read_direct(out)         
        return out
    
class Random_select(object):
    def __init__(self, mode='train', ixi=False):
        self.mode = mode
        self.ixi = ixi
    def __call__(self, sample):
        image = sample['image']       
        if self.mode == 'test':
                return {'image': image}

        image_masked = np.copy(image)
        if self.ixi:
            modalitiy = random.randint(0,2)
        else:
            modalitiy = random.randint(0,3)
            # modalitiy = random.sample(range(4), 2)
        image_masked[modalitiy,...] = -1 #zero value but we normalized to -1~1
        target = image[modalitiy:modalitiy+1,...]
        # print(target)
        # exit()
        return {'image': image,'image_masked': image_masked, 'target': target, 'modalitiy': modalitiy}
        # images_masked is a copy of image with one modality masked; target is the masked modality image; modalitiy is the num of the masked modality
class Random_mask_any_nums_Modalitiy(object):
    def __init__(
        self,
        mode='train',
        num_modalities=4,
        mask_value=-1,
        ixi=False,
        min_mask=1,        
        max_mask=None,     
        fixed_mask_num=None, 
    ):
        self.mode = mode
        self.num_modalities = 3 if ixi else num_modalities
        self.mask_value = mask_value
        self.ixi = ixi
        if fixed_mask_num is not None:
            self.min_mask = self.max_mask = fixed_mask_num
        else:
            self.min_mask = min_mask
            self.max_mask = max_mask if max_mask is not None else self.num_modalities

        assert 1 <= self.min_mask <= self.max_mask <= self.num_modalities, \
            f"Invalid mask range: min={self.min_mask}, max={self.max_mask}, total={self.num_modalities}"
    
    def __call__(self, sample):
        image = sample['image']
        
        if self.mode == 'test':
            return {'image': image}
        
        image_masked = np.copy(image)
        num_to_mask = random.randint(self.min_mask, self.max_mask)
        masked_modalities = sorted(random.sample(range(self.num_modalities), num_to_mask))
        
        for modality in masked_modalities:
            image_masked[modality, ...] = self.mask_value
        target = image[masked_modalities, ...]
        
        return {
            'image': image,
            'image_masked': image_masked,
            'target': target,
            'masked_modalities': np.array(masked_modalities),  
        }
        
class ToTensor_new(object):
    """Convert ndarrays in sample to Tensors with proper types and dimensions."""
    def __init__(self, mode: str = 'train', ixi: bool = False):
        """
        Args:
            mode (str): 'train' or 'test'
            ixi (bool): Whether using IXI dataset (affects modality handling)
        """
        self.mode = mode
        self.ixi = ixi

    def __call__(self, sample) :
        # Convert main image
        image = self._convert_to_tensor(sample['image'])
        
        if self.mode == 'test':
            return {'image': image}

        # Convert other components
        target = self._convert_to_tensor(sample['target'])
        
        # Handle both old ('modalitiy') and new ('masked_modalities') naming
        if 'masked_modalities' in sample:
            masked_modalities = self._convert_to_tensor(sample['masked_modalities'], dtype=torch.long)
        else:
            masked_modalities = self._convert_to_tensor(sample['modalitiy'], dtype=torch.long)
        
        image_masked = self._convert_to_tensor(sample['image_masked'])
        
        return {
            'image': image,
            'image_masked': image_masked,
            'target': target,
            'masked_modalities': masked_modalities  # Unified output name
        }

    def _convert_to_tensor(self, array: np.ndarray, dtype: torch.dtype = torch.float) -> torch.Tensor:
        """Helper function to convert numpy array to contiguous torch tensor"""
        array = np.ascontiguousarray(array)
        return torch.from_numpy(array).to(dtype)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, mode='train', ixi=False):
        self.mode = mode
        self.ixi = ixi
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float()

        if self.mode == 'test':
            return {'image': image}

        target = sample['target']
        target = np.ascontiguousarray(target)
        target = torch.from_numpy(target).float()

        modalitiy = sample['modalitiy']
        modalitiy = np.ascontiguousarray(modalitiy)
        modalitiy = torch.from_numpy(modalitiy).long()
        
        image_masked = sample['image_masked']
        image_masked = np.ascontiguousarray(image_masked)
        image_masked = torch.from_numpy(image_masked).float()

        return {'image': image,'image_masked': image_masked, 'target': target, 'modalitiy': modalitiy}
    
def augmentation(sample, ixi=False):
    trans = transforms.Compose([
        Random_select(ixi=ixi),
        ToTensor(ixi=ixi)
        # Random_mask_any_nums_Modalitiy(ixi=ixi, max_mask=max_mask, fixed_mask_num=fixed_mask_num),
        # ToTensor_new(ixi=ixi)
    ])

    return trans(sample)

def augmentation_test(sample, ixi=False):
    trans = transforms.Compose([
        Random_select(mode='test', ixi=ixi),
        ToTensor(mode='test', ixi=ixi)
        # Random_mask_any_nums_Modalitiy(mode='test',ixi=ixi, max_mask=max_mask, fixed_mask_num=fixed_mask_num),
        # ToTensor_new(mode='test',ixi=ixi)
    ])

    return trans(sample)