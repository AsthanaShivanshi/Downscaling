
import xarray as xr
import numpy as np

def norm_params(ds, var_name):
    """Calculating the gridwise mean and standard deviation for a given variable assuming a normal distribtuion (to be used for Tmeperature"""
    var= ds[var_name]
    mean= var.mean(dim="time", skipna=True)
    std= var.std(dim="time", skipna=True)

    return mean, std

def min_max_calculator(ds, var_name):
    "Calcualting the gridwise min and max for a given variable(to be used for non normally distributed variales such as precipitation)"""
    var= ds[var_name]
    min= var.min(dim="time", skipna=True)
    max= var.max(dim="time", skipna=True)

    return min,max

#Depending on the distribution of the var (precip or temperature) , now standardisation will be performed

def normalise(var, mean, std):
    return (var-mean)/std

def min_max_scaler(var, min, max):
    return (var-min)/(max-min)

#Using all the previously written functions to standardise any dataset depending on the variables it has 

def standardise(input_path, output_path, var):
    ds= xr.open_dataset(input_path,chunks={"time":100})

    if var in ["pr", "RhiresD"]: #More variables can be added depending on the dataset specifications
        min, max= min_max_calculator(ds, var)
        scaled_var= min_max_scaler(ds[var],min, max)

    elif var in ["tas", "TabsD"]:
        mean, std= norm_params(ds, var)
        scaled_var= normalise(ds[var], mean,std)
    
    else:
        print("Unknown variable type or name. Please check and make edits in the fucntion as required")
    
    scaled_ds= scaled_var.to_dataset(name=var)
    scaled_ds.to_netcdf(output_path)
    print(f"Normalised/min max scaled variable saved to {output_path}")
