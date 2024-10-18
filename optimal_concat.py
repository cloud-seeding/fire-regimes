import os
import shutil
import logging
import dask
import xarray as xr
from alive_progress import alive_bar

# Set up logging
logging.basicConfig(
    filename='merge_datasets.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Directory containing the NetCDF files
directory = './assets/all'

# List all NetCDF files in the directory
file_list = [
    os.path.join(directory, f)
    for f in os.listdir(directory)
    if f.endswith('.nc')
]

# Set Dask temporary directory
dask_temp_dir = 'dask_temp'
dask.config.set({'temporary-directory': dask_temp_dir})

valid_file_list = []

# Function to check if a file is a valid NetCDF file


def is_valid_netcdf(file_path):
    engines = ['netcdf4', 'scipy']
    for engine in engines:
        try:
            xr.open_dataset(file_path, engine=engine)
            logging.info(f"Valid NetCDF file: {file_path} (engine: {engine})")
            return True
        except Exception as e:
            logging.warning(f"Failed to open {
                            file_path} with engine {engine}: {e}")
            continue
    logging.error(f"Invalid NetCDF file: {file_path}")
    return False


# Validate files before processing
logging.info("Starting validation of NetCDF files.")
with alive_bar(len(file_list), title='Validating datasets') as bar:
    for file_path in file_list:
        if is_valid_netcdf(file_path):
            valid_file_list.append(file_path)
        else:
            print(f"Skipping invalid file: {file_path}")
        bar()

logging.info(f"Validation complete. {
             len(valid_file_list)} valid files found out of {len(file_list)}.")

if not valid_file_list:
    logging.error("No valid NetCDF files found. Exiting.")
    print("No valid NetCDF files found.")
    exit(1)

# Initialize the progress bar for merging
logging.info("Starting merging of NetCDF files.")
with alive_bar(len(valid_file_list), title='Merging datasets') as bar:
    def preprocess(ds):
        # Update the progress bar
        bar()
        return ds

    # Open and merge multiple datasets using xarray with Dask for parallel processing
    try:
        combined_ds = xr.open_mfdataset(
            valid_file_list,
            combine='by_coords',
            preprocess=preprocess,
            chunks='auto',
            parallel=True,
            engine='netcdf4'
        )
        logging.info("Datasets successfully merged.")
    except Exception as e:
        logging.error(f"Error during merging datasets: {e}")
        print(f"Error during merging datasets: {e}")
        exit(1)

# Save the combined dataset to a NetCDF file
output_file = 'combined_output.nc'
try:
    combined_ds.to_netcdf(output_file)
    logging.info(f"Combined dataset saved to {output_file}.")
except Exception as e:
    logging.error(f"Error saving combined dataset: {e}")
    print(f"Error saving combined dataset: {e}")
    exit(1)
finally:
    # Close the dataset to release resources
    combined_ds.close()
    del combined_ds

# Remove Dask temporary directory
shutil.rmtree(dask_temp_dir, ignore_errors=True)
logging.info("Temporary files cleaned up. Processing complete.")
