import argparse
import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import FolderDataset
from utils import Xception


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--your-team-name", required=True)
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--model-weights", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hflip", action="store_true")
    return parser.parse_args()


def get_dataset(data_folder):
    import torchvision.transforms as Transforms

    transform = Transforms.Compose(
        [
            Transforms.Resize((299, 299)),
            Transforms.ToTensor(),
            Transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return FolderDataset(data_folder, transform)


def get_model(weights_path):
    model = Xception()
    model.fc = torch.nn.Linear(2048, 1)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval().cuda()
    return model


@torch.no_grad()
def run(model, dataset, batch_size, num_workers, hflip):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    names = [name.strip() for name in dataset.get_img_name()]
    predictions = {}
    offset = 0
    start = time.time()
    print("Detection model inferring with TTA ...")
    for images in tqdm(loader):
        images = images.cuda(non_blocking=True)
        pred = model(images)
        if hflip:
            pred = (pred + model(torch.flip(images, dims=[3]))) / 2.0
        pred = pred.detach().cpu().view(-1).numpy().tolist()
        for name, score in zip(names[offset : offset + len(pred)], pred):
            predictions[name] = score
        offset += len(pred)
    return {"predictions": predictions, "time": time.time() - start}


def main():
    opts = get_opts()
    dataset = get_dataset(opts.data_folder)
    model = get_model(opts.model_weights)
    results = run(model, dataset, opts.batch_size, opts.num_workers, opts.hflip)

    os.makedirs(opts.result_path, exist_ok=True)
    writer = pd.ExcelWriter(os.path.join(opts.result_path, opts.your_team_name + ".xlsx"))
    prediction_frame = pd.DataFrame(
        data={
            "img_names": results["predictions"].keys(),
            "predictions": results["predictions"].values(),
        }
    )
    time_frame = pd.DataFrame(
        data={
            "Data Volume": [len(results["predictions"].keys())],
            "Time": [results["time"]],
        }
    )
    prediction_frame.to_excel(writer, sheet_name="predictions", index=False)
    time_frame.to_excel(writer, sheet_name="time", index=False)
    writer.close()


if __name__ == "__main__":
    main()
