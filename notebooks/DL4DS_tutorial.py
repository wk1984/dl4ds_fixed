
import warnings, os

warnings.filterwarnings("ignore")

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

try:
    print('use local files ...'); use_local = True;
    cmlds_train = xr.open_dataset('cmlds_train.nc')
    cmlds_test  = xr.open_dataset('cmlds_test.nc')
    cmlds_val   = xr.open_dataset('cmlds_val.nc')
except:
    print('try to download datasets from MAELSTROM project ...')
    cmlds_train = cml.load_dataset("maelstrom-downscaling-tier1", dataset="training")
    cmlds_val = cml.load_dataset("maelstrom-downscaling-tier1", dataset="validation")
    cmlds_test = cml.load_dataset("maelstrom-downscaling-tier1", dataset="testing")

if use_local:

    t2m_hr_train = cmlds_train.t2m_tar
    t2m_hr_test = cmlds_test.t2m_tar
    t2m_hr_val = cmlds_val.t2m_tar

    t2m_lr_train = cmlds_train.t2m_in
    t2m_lr_test = cmlds_test.t2m_in
    t2m_lr_val = cmlds_val.t2m_in

    z_hr_train = cmlds_train.z_tar
    z_hr_test = cmlds_test.z_tar
    z_hr_val = cmlds_val.z_tar

    z_lr_train = cmlds_train.z_in
    z_lr_test = cmlds_test.z_in
    z_lr_val = cmlds_val.z_in

else:

    t2m_hr_train = cmlds_train.to_xarray().t2m_tar
    t2m_hr_test = cmlds_test.to_xarray().t2m_tar
    t2m_hr_val = cmlds_val.to_xarray().t2m_tar

    t2m_lr_train = cmlds_train.to_xarray().t2m_in
    t2m_lr_test = cmlds_test.to_xarray().t2m_in
    t2m_lr_val = cmlds_val.to_xarray().t2m_in

    z_hr_train = cmlds_train.to_xarray().z_tar
    z_hr_test = cmlds_test.to_xarray().z_tar
    z_hr_val = cmlds_val.to_xarray().z_tar

    z_lr_train = cmlds_train.to_xarray().z_in
    z_lr_test = cmlds_test.to_xarray().z_in
    z_lr_val = cmlds_val.to_xarray().z_in

t2m_scaler_train = dds.StandardScaler(axis=None)
t2m_scaler_train.fit(t2m_hr_train)  
y_train = t2m_scaler_train.transform(t2m_hr_train)
y_test = t2m_scaler_train.transform(t2m_hr_test)
y_val = t2m_scaler_train.transform(t2m_hr_val)

x_train = t2m_scaler_train.transform(t2m_lr_train)
x_test = t2m_scaler_train.transform(t2m_lr_test)
x_val = t2m_scaler_train.transform(t2m_lr_val)

z_scaler_train = dds.StandardScaler(axis=None)
z_scaler_train.fit(z_hr_train)  
y_z_train = z_scaler_train.transform(z_hr_train)
y_z_test = z_scaler_train.transform(z_hr_test)
y_z_val = z_scaler_train.transform(z_hr_val)

x_z_train = z_scaler_train.transform(z_lr_train)
x_z_test = z_scaler_train.transform(z_lr_test)
x_z_val = z_scaler_train.transform(z_lr_val)

y_train = y_train.expand_dims(dim='channel', axis=-1)
y_test = y_test.expand_dims(dim='channel', axis=-1)
y_val = y_val.expand_dims(dim='channel', axis=-1)

x_train = x_train.expand_dims(dim='channel', axis=-1)
x_test = x_test.expand_dims(dim='channel', axis=-1)
x_val = x_val.expand_dims(dim='channel', axis=-1)

y_z_train = y_z_train.expand_dims(dim='channel', axis=-1)
y_z_test = y_z_test.expand_dims(dim='channel', axis=-1)
y_z_val = y_z_val.expand_dims(dim='channel', axis=-1)

x_z_train = x_z_train.expand_dims(dim='channel', axis=-1)
x_z_test = x_z_test.expand_dims(dim='channel', axis=-1)
x_z_val = x_z_val.expand_dims(dim='channel', axis=-1)

print(y_train.shape, y_test.shape, y_val.shape)
print(x_train.shape, x_test.shape, x_val.shape)

print(x_z_train.shape, x_z_test.shape, x_z_val.shape)
print(y_z_train.shape, y_z_test.shape, y_z_val.shape)

_ = dds.create_pair_hr_lr(y_train.values[0], None, 'spc', 8, None, None, y_z_train.values[0], season=None, debug=False, interpolation='inter_area')

ARCH_PARAMS = dict(n_filters=8,
                   n_blocks=8,
                   normalization=None,
                   dropout_rate=0.0,
                   dropout_variant='spatial',
                   attention=False,
                   activation='relu',
                   localcon_layer=True)

trainer = dds.SupervisedTrainer(
    backbone='resnet',
    upsampling='spc', 
    data_train=y_train, 
    data_val=y_val,
    data_test=y_test,
    data_train_lr=None, # here you can pass the LR dataset for training with explicit paired samples
    data_val_lr=None, # here you can pass the LR dataset for training with explicit paired samples
    data_test_lr=None, # here you can pass the LR dataset for training with explicit paired samples
    scale=8,
    time_window=None, 
    static_vars=None,
    predictors_train=[y_z_train],
    predictors_val=[y_z_val],
    predictors_test=[y_z_test],
    interpolation='inter_area',
    patch_size=None, 
    batch_size=60, 
    loss='mae',
    epochs=100, 
    steps_per_epoch=None, 
    validation_steps=None, 
    test_steps=None, 
    learning_rate=(1e-3, 1e-4), lr_decay_after=1e4,
    early_stopping=False, patience=6, min_delta=0, 
    save=False, 
    save_path=None,
    show_plot=None, verbose=None, 
    device='GPU', 
    **ARCH_PARAMS)

trainer.run()

pred = dds.Predictor(
    trainer, 
    y_test, 
    scale=8, 
    array_in_hr=True,
    static_vars=None, 
    predictors=[y_z_test], 
    time_window=None,
    interpolation='inter_area', 
    batch_size=8,
    scaler=t2m_scaler_train,
    save_path=None,
    save_fname=None,
    return_lr=True,
    device='CPU')

unscaled_y_pred, coarsened_array = pred.run()

unscaled_y_test = t2m_scaler_train.inverse_transform(y_test)

unscaled_y_test.to_netcdf('final_out.nc')

print('All done')