import os
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from datetime import datetime
from alive_progress import alive_bar

directory = '../weatherregimes/assets/all'
file_list = [os.path.join(directory, f) for f in os.listdir(directory)]

def merge_two_datasets(ds1, ds2):
    return xr.merge([ds1, ds2])

# Recursive function for merging datasets in log(n) fashion
def merge_datasets(files):
    if len(files) == 1:
        return xr.open_dataset(files[0])
    elif len(files) == 2:
        ds1 = xr.open_dataset(files[0])
        ds2 = xr.open_dataset(files[1])
        return merge_two_datasets(ds1, ds2)
    else:
        mid = len(files) // 2
        left_merged = merge_datasets(files[:mid])
        right_merged = merge_datasets(files[mid:])
        return merge_two_datasets(left_merged, right_merged)

# Merge all datasets
combined_ds = merge_datasets(file_list)

combined_ds.to_netcdf('combined_output.nc')