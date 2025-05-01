import xarray as xr
import numpy as np
import os

def norm_params(ds, var_name):
    """Calculating the gridwise mean and standard deviation for a given variable assuming a normal distribtuion (to be used for Tmeperature"""
    var= ds[var_name]
    mean= var.mean(dim="time", skipna=True)
    std= var.std(dim="time", skipna=True)

    return mean, std

def min_max_scaler(ds, var_name):
    "Calcualting the gridwise min and max for a given variable(to be used for non normally distributed variales such as precipitation)"""
    var= ds[var_name]
    min= var.min(dim="time", skipna=True)
    max= var.max(dim="time", skipna=True)

    return min,max

#Depending on the distribution of the var (precip or temperature) , now standardisation will be performed

def standardise(ds, var_name, mean, std):
