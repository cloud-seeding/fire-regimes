import os
import csv
from netCDF4 import Dataset
import numpy as np
from alive_progress import alive_bar


def process_nc_file(file_path, csv_writer, first_file=False):
    with Dataset(file_path, 'r') as nc:
        # Get dimensions
        time = nc.variables['time'][:]
        level = nc.variables['level'][:]
        y = nc.variables['y'][:]
        x = nc.variables['x'][:]

        # Get variables
        variables = ['air', 'shum', 'omega', 'hgt', 'uwnd', 'vwnd']

        # Write header if it's the first file
        if first_file:
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


def main():
    input_directory = './assets/all'
    output_file = 'merged_data.csv'

    nc_files = [f for f in os.listdir(input_directory) if f.endswith('.nc')]
    total_files = len(nc_files)

    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        with alive_bar(total_files, title="Processing files", unit="file") as bar:
            for i, file in enumerate(nc_files, 1):
                file_path = os.path.join(input_directory, file)
                process_nc_file(file_path, csv_writer, first_file=(i == 1))
                bar()  # Update the progress bar


if __name__ == "__main__":
    main()
