import pandas as pd
import glob
import os
from pathlib import Path
import re
from tqdm import tqdm


def extract_ids_from_filename(filename):
    """Extract ID1 and ID2 from filename using regex."""
    pattern = r'.*?_(\d+)_(\d+)\.csv$'
    match = re.match(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def merge_csvs(directory_path, output_file, chunksize=10000):
    """
    Merge all CSVs in the directory while preserving source file information.
    Uses chunking for memory efficiency.
    """
    # Get list of all CSV files
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

    # Create output file and write header
    first_chunk = True

    print(f"Found {len(csv_files)} CSV files to process")

    # Process each CSV file
    for file_path in tqdm(csv_files, desc="Processing files"):
        filename = Path(file_path).name
        id1, id2 = extract_ids_from_filename(filename)

        # Read the CSV in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # Add source file information
            chunk['source_file'] = filename
            chunk['id1'] = id1
            chunk['id2'] = id2

            # Write to output file
            mode = 'w' if first_chunk else 'a'
            chunk.to_csv(output_file,
                         mode=mode,
                         index=False,
                         header=first_chunk)

            first_chunk = False

    print(f"Merged data saved to: {output_file}")


if __name__ == "__main__":
    # Example usage
    input_directory = "./output_csv"
    output_file = "merged_output.csv"

    merge_csvs(input_directory, output_file)
