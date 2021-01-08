import glob
import random
import os

from torch.utils.data import Dataset
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
import random
import numpy as np
import torch

class ImageDataset(Dataset):
    def __init__(self, root, unaligned=False, mode='train'):
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/train_A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/train_C_fixed_official' % mode) + '/*.*'))

    def __getitem__(self, index):
        i = random.randint(0, 48)
        j = random.randint(0, 48)
        k=random.randint(0,100)
        
        item_A = color.rgb2lab(io.imread(self.files_A[index % len(self.files_A)]))
        item_A=item_A[:,:,0]
        item_A=resize(item_A,(448,448))
        item_A=item_A[i:i+400,j:j+400]
        if k>50:
            item_A=np.fliplr(item_A)
        item_A=np.asarray(item_A)/50.0-1.0
        item_A=torch.from_numpy(item_A).float()
        item_A=item_A.view(400,400,1)
        item_A=item_A.transpose(0, 1).transpose(0, 2).contiguous()
        if self.unaligned:
            item_B = color.rgb2lab(io.imread(self.files_B[random.randint(0, len(self.files_B) - 1)]))
            item_B=item_B[:,:,0]
            item_B=resize(item_B,(448,448))
            item_B=item_B[i:i+400,j:j+400]
            if k>50:
                item_B=np.fliplr(item_B)
            item_B=np.asarray(item_B)/50.0-1.0
            item_B=torch.from_numpy(item_B).float()
            item_B=item_B.view(400,400,1)
            item_B=item_B.transpose(0, 1).transpose(0, 2).contiguous()
        else:
            item_B = color.rgb2lab(io.imread(self.files_B[index % len(self.files_B)]))
            item_B=item_B[:,:,0]
            item_B=resize(item_B,(448,448))
            item_B=item_B[i:i+400,j:j+400]
            if k>50:
                item_B=np.fliplr(item_B)
            item_B=np.asarray(item_B)/50.0-1.0
            item_B=torch.from_numpy(item_B).float()
            item_B=item_B.view(400,400,1)
            item_B=item_B.transpose(0, 1).transpose(0, 2).contiguous()

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

