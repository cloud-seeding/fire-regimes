import os
import xarray as xr
import dask
import shutil
import logging
from dask.diagnostics import ProgressBar

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
            ds = xr.open_dataset(file_path, engine=engine)
            ds.close()
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
for idx, file_path in enumerate(file_list):
    if is_valid_netcdf(file_path):
        valid_file_list.append(file_path)
    else:
        print(f"on {idx}: Skipping invalid file: {file_path}")

logging.info(f"Validation complete. {
             len(valid_file_list)} valid files found out of {len(file_list)}.")

if not valid_file_list:
    logging.error("No valid NetCDF files found. Exiting.")
    print("No valid NetCDF files found.")
    exit(1)

# Optionally process files in batches to limit open file handles
batch_size = 500  # Adjust this based on your system's limits
batches = [valid_file_list[i:i + batch_size]
           for i in range(0, len(valid_file_list), batch_size)]

merged_datasets = []

for batch_idx, batch_files in enumerate(batches):
    logging.info(f"Processing batch {batch_idx + 1} out of {len(batches)}")
    print(f"Processing batch {batch_idx + 1} out of {len(batches)}")

    try:
        with ProgressBar():
            ds = xr.open_mfdataset(
                batch_files,
                combine='by_coords',
                chunks='auto',
                parallel=True,
                engine='netcdf4'
            )
            # Trigger computation to catch any errors
            ds.load()
        merged_datasets.append(ds)
        logging.info(f"Batch {batch_idx + 1} merged successfully.")
    except Exception as e:
        logging.error(f"Error during merging batch {batch_idx + 1}: {e}")
        print(f"Error during merging batch {batch_idx + 1}: {e}")
        # Optionally, identify problematic files within the batch
        for file in batch_files:
            try:
                ds = xr.open_dataset(file, engine='netcdf4')
                ds.close()
            except Exception as file_e:
                logging.error(f"Problem with file {file}: {file_e}")
                print(f"Problem with file {file}: {file_e}")
        exit(1)

# Combine all batches into a single dataset
logging.info("Combining all batches into a single dataset.")
print("Combining all batches into a single dataset.")

try:
    with ProgressBar():
        combined_ds = xr.combine_by_coords(
            merged_datasets, combine_attrs='override')
    logging.info("All batches combined successfully.")
except Exception as e:
    logging.error(f"Error during final combination: {e}")
    print(f"Error during final combination: {e}")
    exit(1)

# Save the combined dataset to a NetCDF file
output_file = 'combined_output.nc'
try:
    combined_ds.to_netcdf(output_file)
    logging.info(f"Combined dataset saved to {output_file}.")
    print(f"Combined dataset saved to {output_file}.")
except Exception as e:
    logging.error(f"Error saving combined dataset: {e}")
    print(f"Error saving combined dataset: {e}")
    exit(1)
finally:
    # Close datasets to release resources
    combined_ds.close()
    del combined_ds
    for ds in merged_datasets:
        ds.close()
    del merged_datasets

# Remove Dask temporary directory
shutil.rmtree(dask_temp_dir, ignore_errors=True)
logging.info("Temporary files cleaned up. Processing complete.")
print("Temporary files cleaned up. Processing complete.")
