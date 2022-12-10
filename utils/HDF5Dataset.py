from torch.utils.data import Dataset
import h5py
import torch

class HDF5Dataset(Dataset):
    def __init__(self, file_path, name, transform=None, img_transform=True):
        super().__init__
        self.file_path = file_path
        self.data_cache = {}
        self.transform = transform # rotate or flip
        self.img_transform = img_transform # normalize
        self.name = str(name)
        self.name_label = str(name) + "_label"
        self.size = None
        with h5py.File(file_path, 'r') as hf:
            self.data_cache[self.name] = hf[self.name][:]
            self.data_cache[self.name_label] = hf[self.name_label][:]
            self.size = len(hf[self.name_label])

    def __getitem__(self, index):
        imgs = torch.from_numpy(self.data_cache[self.name][index]).float()
        imgs[0] = imgs[0] / 255
        if self.img_transform:
            imgs[0] = self.img_transform(imgs[0].unsqueeze(0))

        if self.transform:
            imgs = self.transform(imgs)  

        img = imgs[0].unsqueeze(0)
        mask = imgs[1].unsqueeze(0)
        label = self.data_cache[self.name_label][index]
        # image, mask, label
        return img, mask ,label

    def __len__(self):
        return self.size