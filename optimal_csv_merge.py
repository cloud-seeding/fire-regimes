import os
import csv
from netCDF4 import Dataset
import numpy as np
from alive_progress import alive_bar
import multiprocessing as mp
from functools import partial


def process_nc_file(file_path, output_directory):
    filename = os.path.basename(file_path)
    output_file = os.path.join(output_directory, f"{filename[:-3]}.csv")

    with Dataset(file_path, 'r') as nc, open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Get dimensions
        time = nc.variables['time'][:]
        level = nc.variables['level'][:]
        y = nc.variables['y'][:]
        x = nc.variables['x'][:]

        # Get variables
        variables = ['air', 'shum', 'omega', 'hgt', 'uwnd', 'vwnd']

        # Write header
        header = ['time', 'level', 'y', 'x'] + variables
        csv_writer.writerow(header)

        # Iterate through all combinations of dimensions
        for t in range(len(time)):
            for l in range(len(level)):
                for yi in range(len(y)):
                    for xi in range(len(x)):
                        row = [time[t], level[l], y[yi], x[xi]]
                        for var in variables:
                            row.append(nc.variables[var][t, l, yi, xi])
                        csv_writer.writerow(row)

    return output_file


def main():
    input_directory = './assets/all'
    output_directory = './output_csv'
    final_output = 'merged_data.csv'

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    nc_files = [os.path.join(input_directory, f)
                for f in os.listdir(input_directory) if f.endswith('.nc')]
    total_files = len(nc_files)

    # Determine the number of CPU cores to use (leave one core free)
    num_cores = max(1, mp.cpu_count() - 1)

    print(f"Processing {total_files} files using {num_cores} CPU cores...")

    # Create a partial function with the output directory
    process_file = partial(process_nc_file, output_directory=output_directory)

    # Process files in parallel with a progress bar
    with alive_bar(total_files, title="Processing files", unit="file") as bar:
        with mp.Pool(num_cores) as pool:
            for _ in pool.imap_unordered(process_file, nc_files):
                bar()

    print("All files processed. Merging CSV files...")

    # Merge all CSV files
    with open(final_output, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        header_written = False

        for filename in os.listdir(output_directory):
            if filename.endswith(".csv"):
                with open(os.path.join(output_directory, filename), 'r', newline='') as infile:
                    reader = csv.reader(infile)
                    if not header_written:
                        writer.writerow(next(reader))  # write header
                        header_written = True
                    else:
                        next(reader)  # skip header
                    for row in reader:
                        writer.writerow(row)

    print(f"All done! Final output: {final_output}")


if __name__ == "__main__":
    main()
