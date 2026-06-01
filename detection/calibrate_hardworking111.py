import argparse
import os

import numpy as np
import pandas as pd


def logit(x):
    eps = 1e-7
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.06568622589111328)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    predictions = pd.read_excel(args.input, sheet_name="predictions")
    time_sheet = pd.read_excel(args.input, sheet_name="time")
    raw_scores = predictions["predictions"].to_numpy(dtype=np.float64)
    new_scores = sigmoid((logit(raw_scores) - logit(args.threshold)) / args.temperature)

    predictions = predictions.copy()
    predictions["img_names"] = predictions["img_names"].astype(str).str.strip()
    predictions["predictions"] = np.clip(new_scores, 0.0, 1.0)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = pd.ExcelWriter(args.output)
    predictions.to_excel(writer, sheet_name="predictions", index=False)
    time_sheet.to_excel(writer, sheet_name="time", index=False)
    writer.close()
    print("saved:", args.output)
    print(predictions["predictions"].describe().to_string())


if __name__ == "__main__":
    main()
