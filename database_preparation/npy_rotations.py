import argparse
import os
import numpy as np
from tqdm import tqdm

cur_dir = os.getcwd()

def rotate_array(cube: np.array, direction: str):
    if direction == "x":
        return cube
    if direction == "y":
        return np.transpose(cube, (1, 2, 0))
    if direction == "z":
        return np.transpose(cube, (2, 0, 1))
    raise ValueError(f"Incorrect direction: {direction}")

def process_npy(inputfilepath: str, outputfilepath: str, rotation: str) -> None:
    X = np.load(inputfilepath)
    X = X.astype("bool")
    X = rotate_array(X, direction=rotation)
    np.save(outputfilepath, X)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../datasets/results/grids_80", help="Path to directory where npy files are stored")
    parser.add_argument("--output_dir", type=str, default="database_80", required=True, help="Output directory")
    parser.add_argument("--rotation", choices=["x", "y", "z"], default="x", help="How to rotate structure - use x for no rotation")
    return vars(parser.parse_args())


def main():
    args = parse_command_line_args()
    rotation = args['rotation']

    npyfilenames = [x for x in os.listdir(args['input_dir']) if x.endswith('.npy')]
    if len(npyfilenames) == 0:
        print('List of .npy files is empty!')
        return

    os.makedirs(args['output_dir'], exist_ok=True)
    for npyfilename in tqdm(npyfilenames):
        inputpath = os.path.join(args['input_dir'], npyfilename)
        basefilename = npyfilename.replace('grid_', '').replace('.npy', '')
        outputpath = os.path.join(args['output_dir'], f'{basefilename}_rot-{rotation}.npy')
        process_npy(inputfilepath=inputpath, outputfilepath=outputpath, rotation=rotation)


if __name__ == "__main__":
    main()
