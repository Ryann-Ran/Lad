from torch.utils.data import Dataset
import os
import numpy as np
import torch

class ABDOMENCT1KDataset(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = root_dir
        self.file_paths = self.get_data_files()

    def get_data_files(self):
        nifti_file_names = os.listdir(self.root_dir)
        folder_names = [os.path.join(self.root_dir, nifti_file_name) for nifti_file_name in nifti_file_names if nifti_file_name.endswith('.npy')]
        return folder_names

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img = np.load(self.file_paths[idx])

        imageout = torch.from_numpy(img.copy()).float()
        imageout = imageout.unsqueeze(0)

        data_path = self.file_paths[idx]
        
        mask_path = data_path.replace("/image", "/mask")
        mask_array = np.load(mask_path)
        mask_array = torch.from_numpy(mask_array).float()  # torch.Size([32,256,256])
        mask_array = mask_array.unsqueeze(0)
        
        topo_path = mask_path.replace('mask', 'topo').replace("nii.gz", "npy")
        topo = torch.from_numpy(np.load(topo_path)).long()

        return {'data': imageout, 'mask': mask_array, 'topo': topo, 'data_path': data_path, 'mask_path': mask_path}