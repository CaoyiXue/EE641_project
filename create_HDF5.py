import h5py    
import numpy as np
import os
from PIL import Image
import glob
import requests
import zipfile

def create_hdf5data(root, hdf5_root, names, label2num):
    folder_names = ['images', 'infection masks', 'lung masks']
    if 'lung' in hdf5_root:
        folder_names = ['images', 'lung masks']
    with h5py.File(os.path.join(hdf5_root), 'w') as hf:
        for name in names:
            image_data, label_data = [], []
            for label_name, num in label2num.items():
                path_list = glob.glob(os.path.join(root, name, label_name,
                                                folder_names[0], '*.png'))
                for path in path_list:
                    images = [np.asarray(Image.open(path))]
                    for folder_name in folder_names[1:]:
                        mask_path = path.split('/')
                        mask_path[-2] = folder_name
                        mask_path = os.path.join(*mask_path)
                        # change mask 255 to 1
                        mask = np.asarray(Image.open(mask_path))//255
                        images.append(mask)
                    image_data.append(images)
                    label_data.append(num)

            image_data = np.array(image_data)
            label_data = np.array(label_data)
            print(image_data.shape, label_data.shape)
            hf.create_dataset(name.lower(), data=image_data)
            hf.create_dataset(name.lower()+'_label', data=label_data)


if __name__ == '__main__':
    print("Whether download data? [yes/no]")
    str = input()
    if str.lower() == 'yes':
        print('download start')
        URL = 'https://github.com/CaoyiXue/EE641_project_data/archive/refs/heads/main.zip'
        req = requests.get(URL)
        filename = "data.zip"

        with open(filename,'wb') as output_file:
            output_file.write(req.content)


        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall("./")

        os.remove(filename)

        print('download complete')
    else:
        pass

    if not os.path.exists('HDF5Data'):
        os.makedirs('HDF5Data')

    roots = ["EE641_project_data-main/Infection Segmentation Data/Infection Segmentation Data/",
            "EE641_project_data-main/Lung Segmentation Data/Lung Segmentation Data"]

    hdf5_roots = ["HDF5Data/infect_data.hdf5", "HDF5Data/lung_data.hdf5"]
    names = ['Train', 'Val', 'Test']
    label2num = {'COVID-19': 0, 'Non-COVID': 1, 'Normal': 2}
    for root, hdf5_root in zip(roots, hdf5_roots):
        create_hdf5data(root, hdf5_root, names, label2num)