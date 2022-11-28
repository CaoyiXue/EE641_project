from torch.utils.data import Dataset
import h5py
import torch

class HDF5Dataset(Dataset):
    def __init__(self, file_path, name, transform=None):
        super().__init__
        self.file_path = file_path
        self.data_cache = {}
        self.transform = transform
        self.name = str(name)
        self.name_label = str(name) + "_label"
        self.size = None
        with h5py.File(file_path, 'r') as hf:
            self.data_cache[self.name] = hf[self.name][:]
            self.data_cache[self.name_label] = hf[self.name_label][:]
            self.size = len(hf[self.name_label])

    def __getitem__(self, index):
        imgs = self.data_cache[self.name][index]
        if self.transform:
            img = self.transform(imgs[0])
        else:
            img = torch.from_numpy(img)

        mask = torch.from_numpy(imgs[1]).float()
        label = self.data_cache[self.name_label][index]
        # image, mask, label
        return img, mask ,label

    def __len__(self):
        return self.size