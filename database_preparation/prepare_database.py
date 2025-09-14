import argparse
import pandas as pd
import re
import os
import numpy as np
from tqdm import tqdm

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../datasets/Dataset1_gold_0.25.xlsx", help="Path to dataset .xlsx file")
    parser.add_argument("--input_dir", type=str, default="database_80", help="Path where rotated grids are stored")
    parser.add_argument("--output_dir", type=str, default=os.getcwd(), help="Path to save the database")
    parser.add_argument("--output_csv", type=str, default="database_train.csv", help="Name of output .csv file")
    parser.add_argument("--include_descriptors", action="store_true", help="Include morphological and topological descriptors in output")
    return vars(parser.parse_args())
    
def sort_paths(path):
    filename = os.path.basename(path)
    match = re.match(r'(\d+)_rot-([xyz])\.npy', filename)
    if match:
        number = int(match.group(1))
        direction = match.group(2)
        dir_order = {'x': 0, 'y': 1, 'z': 2}
        return (dir_order[direction], number)
    else:
        return (999, 999999)

args = parse_command_line_args()
npy_dir = args['input_dir']
excel_path = args['dataset']
output_path = os.path.join(args['output_dir'], args['output_csv'])

df_all = pd.read_excel(excel_path)
include_descriptors = args["include_descriptors"]
if include_descriptors:
    desc_columns = df_all.columns[32:-1]
df_all['Direction'] = df_all['Direction'].str.lower()

lookup_table = {(row['Number'], row['Direction']): row for _, row in df_all.iterrows()}
npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
print(f'{len(npy_files)} .npy grids in "{npy_dir}"')

dataset = []
matched_keys = set()
for filename in tqdm(npy_files):
    try:
        base = os.path.splitext(filename)[0]  
        number_str, rot = base.split("_rot-")
        number = int(number_str)
        direction = rot.lower()

        key = (number, direction)

        if key in lookup_table:
            row = lookup_table[key]
            cii = row['Cii']
            npy_path = os.path.join(npy_dir, filename)

            datapoint = {'npy_path': os.path.abspath(npy_path),
                         'cii': cii}
            if include_descriptors:           
                for col in desc_columns:
                    datapoint[col] = row[col]

            dataset.append(datapoint)
            matched_keys.add(key)
        else:
            print(f'{filename} grids do not match any row in Excel')

    except Exception as e:
        print(f'Error processing "{filename}": {e}')

missing_files = []
for key in lookup_table:
    if key not in matched_keys:
        number, direction = key
        expected_name = f'{number}_rot-{direction}.npy'
        expected_path = os.path.join(npy_dir, expected_name)
        if not os.path.exists(expected_path):
            missing_files.append(expected_name)

if missing_files:
    print(f'{len(missing_files)} grids are missing')

database = pd.DataFrame(dataset)
database = database.sort_values(by="npy_path", key=lambda col: col.map(sort_paths)).reset_index(drop=True)
database.to_csv(output_path, index=False)
print(f'{len(database)} datapoints saved to {output_path}')

