import os
import xarray as xr
from alive_progress import alive_bar
import dask
import shutil

# Directory containing the NetCDF files
directory = '../weatherregimes/assets/all'

# List all NetCDF files in the directory
file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nc')]

# Set Dask temporary directory
dask_temp_dir = 'dask_temp'
dask.config.set({'temporary-directory': dask_temp_dir})

# Initialize the progress bar
with alive_bar(len(file_list), title='Merging datasets') as bar:
    def preprocess(ds):
        # Update the progress bar
        bar()
        return ds

    # Open and merge multiple datasets using xarray with Dask for parallel processing
    combined_ds = xr.open_mfdataset(
        file_list,
        combine='by_coords',
        preprocess=preprocess,
        chunks='auto',
        parallel=True
    )

# Save the combined dataset to a NetCDF file
combined_ds.to_netcdf('combined_output.nc')

# Close the dataset to release resources
combined_ds.close()
del combined_ds

# Remove Dask temporary directory
shutil.rmtree(dask_temp_dir, ignore_errors=True)
