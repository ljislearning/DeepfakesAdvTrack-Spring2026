import argparse
import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hardworking111_data import CourseImageFolder, build_eval_transform
from hardworking111_model import DEFAULT_WEIGHTS, Hardworking111TriGuard


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--your-team-name", default="Hardworking111")
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--model-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--result-path", default="results")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-tta", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.result_path, exist_ok=True)
    dataset = CourseImageFolder(args.data_folder, build_eval_transform())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    names = dataset.get_img_name()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Hardworking111TriGuard(args.model_weights).to(device).eval()

    predictions = {}
    offset = 0
    start = time.time()
    print("Hardworking111 TriGuard inferring ...")
    with torch.no_grad():
        for images in tqdm(loader):
            images = images.to(device)
            scores = model.predict_fake(images, tta=not args.no_tta)
            scores = scores.detach().cpu().numpy().tolist()
            for name, score in zip(names[offset : offset + len(scores)], scores):
                predictions[name] = score
            offset += len(scores)
    elapsed = time.time() - start

    output_path = os.path.join(args.result_path, args.your_team_name + ".xlsx")
    writer = pd.ExcelWriter(output_path)
    pd.DataFrame({"img_names": predictions.keys(), "predictions": predictions.values()}).to_excel(
        writer, sheet_name="predictions", index=False
    )
    pd.DataFrame({"Data Volume": [len(predictions)], "Time": [elapsed]}).to_excel(writer, sheet_name="time", index=False)
    writer.close()
    print("saved:", output_path)


if __name__ == "__main__":
    main()
