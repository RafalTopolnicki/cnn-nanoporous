import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json

from generate_model import generate_model, ALLOWED_MODELS
from utils import DownSample
from data import MDDataset
from train import check_npy_cubic_and_equalsize


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputfile", type=str, default="inference_results.csv", help="Path to output parquet file")
    parser.add_argument("--model_name", choices=ALLOWED_MODELS, default=ALLOWED_MODELS[0], help="Model type")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model file")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--size", type=int, default=80)
    parser.add_argument("--json_path", type=str, default=None, help="Path to json file produced during training")
    parser.add_argument(
        "--txt_path", type=str, default=None, help="Path to txt file containing absolute paths to .npy files"
    )
    return vars(parser.parse_args())


def flatten(xss):
    return [x for xs in xss for x in xs]


def paths_from_json(jsonpath):
    data = json.load(open(jsonpath, "r"))
    paths = flatten(data["paths"])
    return paths


def paths_from_txt(txtpath):
    data = pd.read_csv(txtpath, header=None)
    paths = data.iloc[:, 0].tolist()
    return paths


def main():
    args = parse_command_line_args()
    batch_size = args["batch_size"]
    outputfile = args["outputfile"]
    model_name = args["model_name"]
    model_path = args["model_path"]
    target_size = args["size"]
    json_path = args["json_path"]
    txt_path = args["txt_path"]
    if json_path is None and txt_path is None:
        raise ValueError(f"json_path and txt_path cannot by empty")
    if json_path is not None and txt_path is not None:
        raise ValueError(f"Use only one: json_path or txt_path")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Will run on device={device}")
    if json_path:
        paths = paths_from_json(json_path)
    if txt_path:
        paths = paths_from_txt(txt_path)
    database = pd.DataFrame(paths, columns=["npy_path"])
    database["cii"] = np.nan

    npy_size = check_npy_cubic_and_equalsize(df=database)
    print(f"Structures are cubic: {npy_size}x{npy_size}x{npy_size}")
    print(f"Structures will be rescaled to: {target_size}x{target_size}x{target_size}")

    downsampler = DownSample(scale_factor=target_size / npy_size)
    # the same transformation as for test
    transform = transforms.Compose([transforms.ToTensor(), downsampler])

    inferenceset = MDDataset(df=database, transform=transform, use_descriptors=False)
    dataloader = DataLoader(inferenceset, batch_size=batch_size, shuffle=False)

    # FIXME: this will work only if there are no descriptors
    model = generate_model(model_name=model_name, use_descriptors=False, fc_first_size=0, fc_second_size=0, p_dropout=0)
    model.load_state_dict(torch.load(model_path))
    fully_connected = model.fc
    model.fc = torch.nn.Identity()
    model.to(device)
    fully_connected.to(device)

    allpaths = []
    predictions = []
    embeddings = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Inference"):
            inputs = data[0].to(device)
            embedding = model(inputs)
            y_pred = torch.squeeze(fully_connected(embedding))
            allpaths.append(data[2])
            predictions.append(y_pred.detach().cpu().numpy())
            embeddings.append(embedding.detach().cpu().numpy())
    allpaths = flatten(allpaths)
    predictions = flatten(predictions)
    embeddings = flatten(embeddings)
    df = pd.DataFrame({"path": allpaths, "pred": predictions, "embedding": embeddings})
    df.to_parquet(outputfile, engine="pyarrow")
    print(f"Written to {outputfile}")


if __name__ == "__main__":
    main()
