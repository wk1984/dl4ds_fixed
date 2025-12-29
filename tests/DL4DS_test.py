
import warnings, os

warnings.filterwarnings("ignore")

os.environ['KEC_KERAS_TRACETBACK_FILTERING'] = '0'

import numpy as np
import xarray as xr
import ecubevis as ecv
import dl4ds as dds
import scipy as sp
import netCDF4 as nc
import climetlab as cml

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import models

print('try to download datasets from MAELSTROM project ...')
cmlds_train = cml.load_dataset("maelstrom-downscaling-tier1", dataset="training")
cmlds_val = cml.load_dataset("maelstrom-downscaling-tier1", dataset="validation")
cmlds_test = cml.load_dataset("maelstrom-downscaling-tier1", dataset="testing")

# t2m_hr_train = cmlds_train.to_xarray().sel(lat=slice(52,50), lon=slice(15,17)).t2m_tar
# t2m_hr_test = cmlds_test.to_xarray().sel(lat=slice(52,50), lon=slice(15,17)).t2m_tar
# t2m_hr_val = cmlds_val.to_xarray().sel(lat=slice(52,50), lon=slice(15,17)).t2m_tar
# 
# z_hr_train = cmlds_train.to_xarray().sel(lat=slice(52,50), lon=slice(15,17)).z_tar
# z_hr_test = cmlds_test.to_xarray().sel(lat=slice(52,50), lon=slice(15,17)).z_tar
# z_hr_val = cmlds_val.to_xarray().sel(lat=slice(52,50), lon=slice(15,17)).z_tar
# 
# t2m_scaler_train = dds.StandardScaler(axis=None)
# t2m_scaler_train.fit(t2m_hr_train)  
# y_train = t2m_scaler_train.transform(t2m_hr_train)
# y_test = t2m_scaler_train.transform(t2m_hr_test)
# y_val = t2m_scaler_train.transform(t2m_hr_val)
# 
# z_scaler_train = dds.StandardScaler(axis=None)
# z_scaler_train.fit(z_hr_train)  
# y_z_train = z_scaler_train.transform(z_hr_train)
# y_z_test = z_scaler_train.transform(z_hr_test)
# y_z_val = z_scaler_train.transform(z_hr_val)
# 
# y_train = y_train.expand_dims(dim='channel', axis=-1)
# y_test = y_test.expand_dims(dim='channel', axis=-1)
# y_val = y_val.expand_dims(dim='channel', axis=-1)
# 
# y_z_train = y_z_train.expand_dims(dim='channel', axis=-1)
# y_z_test = y_z_test.expand_dims(dim='channel', axis=-1)
# y_z_val = y_z_val.expand_dims(dim='channel', axis=-1)
# 
# print(y_train.shape, y_test.shape, y_val.shape)
# print(y_z_train.shape, y_z_test.shape, y_z_val.shape)
# 
# _ = dds.create_pair_hr_lr(y_train.values[0], None, 'spc', 8, None, None, y_z_train.values[0], season=None, debug=False, interpolation='inter_area')
# 
# ARCH_PARAMS = dict(n_filters=8,
#                    n_blocks=8,
#                    normalization=None,
#                    dropout_rate=0.0,
#                    dropout_variant='spatial',
#                    attention=False,
#                    activation='relu',
#                    localcon_layer=True)
# 
# trainer = dds.SupervisedTrainer(
#     backbone='resnet',
#     upsampling='spc', 
#     data_train=y_train, 
#     data_val=y_val,
#     data_test=y_test,
#     data_train_lr=None, # here you can pass the LR dataset for training with explicit paired samples
#     data_val_lr=None, # here you can pass the LR dataset for training with explicit paired samples
#     data_test_lr=None, # here you can pass the LR dataset for training with explicit paired samples
#     scale=1,
#     time_window=None, 
#     static_vars=None,
#     predictors_train=[y_z_train],
#     predictors_val=[y_z_val],
#     predictors_test=[y_z_test],
#     interpolation='inter_area',
#     patch_size=None, 
#     batch_size=10, 
#     loss='mae',
#     epochs=10, 
#     steps_per_epoch=None, 
#     validation_steps=None, 
#     test_steps=None, 
#     learning_rate=(1e-3, 1e-4), lr_decay_after=1e4,
#     early_stopping=False, patience=6, min_delta=0, 
#     save=False, 
#     save_path=None,
#     show_plot=None, verbose=None, 
#     device='CPU', 
#     **ARCH_PARAMS)
# 
# trainer.run()

print('All done')