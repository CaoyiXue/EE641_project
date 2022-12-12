# EE641_project Efficiently Detecting COVID-19 in Patient X-ray Images
[GitHubRepo](https://github.com/CaoyiXue/EE641_project.git)

## Data Preparation
We convert the original data set to HDF5 format, run 
```shell
python create_HDF5.py
```
to get corresponding data. It will ask you whether download data. If you type in yes, it will automatically download data, otherwise you need to go to [Kaggle](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu) to download COVID-QU-Ex Dataset

## Training
We use the Colab platform to train our model.\ 
[infect_all_models](infect_all_models.ipynb) trains all infection segmentation models.\ 
[lung_all_models](lung_all_models.ipynb) trains all lung segmentation models.\
[seg_models](seg_models/) implements Unet with ResNet, MobileNet V2 and MicroNet.\
[utils](utils/) implements HDF5Dataset, training loop, test loop, IoU, DSC, COVID detection, and plot one image python functions for use

## Test
[test_all_models] compares Flops, infection segmentation performance, lung segmentation performance, and plots the comparision on one input image. If you want to run this code, you need to download the models from [githubrepo](https://github.com/CaoyiXue/EE641_trained_models.git) and put them under the folder [trained_models](trained_models/)
