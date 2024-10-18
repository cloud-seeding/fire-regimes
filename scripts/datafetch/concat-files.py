import os
import xarray as xr
from alive_progress import alive_bar

directory = './assets/all'
file_list = [os.path.join(directory, f) for f in os.listdir(directory)]


def open_dataset_with_engines(file):
    engines = ['netcdf4', 'h5netcdf', 'scipy']
    for engine in engines:
        try:
            return xr.open_dataset(file, engine=engine)
        except Exception as e:
            print(f"Failed to open {file} with {engine} engine: {str(e)}")
    return None


def merge_two_datasets(ds1, ds2):
    return xr.merge([ds1, ds2])


def merge_datasets(files):
    if len(files) == 1:
        return open_dataset_with_engines(files[0])
    elif len(files) == 2:
        ds1 = merge_datasets([files[0]])
        ds2 = merge_datasets([files[1]])
        if ds1 is None or ds2 is None:
            return ds1 if ds1 is not None else ds2
        return merge_two_datasets(ds1, ds2)
    else:
        mid = len(files) // 2
        left_merged = merge_datasets(files[:mid])
        right_merged = merge_datasets(files[mid:])
        if left_merged is None or right_merged is None:
            return left_merged if left_merged is not None else right_merged
        return merge_two_datasets(left_merged, right_merged)


# Merge all datasets
print("Starting to merge datasets...")
with alive_bar(len(file_list)) as bar:
    combined_ds = merge_datasets(file_list)
    bar()

if combined_ds is not None:
    print("Merging complete. Saving combined dataset...")
    combined_ds.to_netcdf('combined_output.nc')
    print("Combined dataset saved as 'combined_output.nc'")
else:
    print("Failed to merge datasets due to errors.")

print("Listing files that couldn't be opened:")
for file in file_list:
    if open_dataset_with_engines(file) is None:
        print(f"- {file}")
