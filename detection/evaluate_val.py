import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def normalize_name(name):
    return os.path.splitext(str(name).strip())[0]


def load_ground_truth(path):
    data = pd.read_excel(path)
    required = {"img_names", "labels"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Ground-truth file is missing columns: {}".format(sorted(missing)))

    data = data.copy()
    data["key"] = data["img_names"].map(normalize_name)
    return data[["key", "labels"]]


def load_predictions(path):
    data = pd.read_excel(path, sheet_name="predictions")
    required = {"img_names", "predictions"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("{} is missing columns: {}".format(path, sorted(missing)))

    data = data.copy()
    data["key"] = data["img_names"].map(normalize_name)
    return data[["key", "predictions"]]


def evaluate_one(gt, pred_path):
    pred = load_predictions(pred_path)
    merged = gt.merge(pred, on="key", how="left")
    missing_count = int(merged["predictions"].isna().sum())
    if missing_count:
        missing = merged.loc[merged["predictions"].isna(), "key"].head(10).tolist()
        raise ValueError(
            "{} predictions are missing in {}. First missing keys: {}".format(
                missing_count,
                pred_path,
                missing,
            )
        )

    labels = merged["labels"].to_numpy()
    scores = merged["predictions"].to_numpy()
    binary_preds = (scores >= 0.5).astype(np.int64)
    return {
        "submit": os.path.splitext(os.path.basename(pred_path))[0],
        "count": len(merged),
        "auc": roc_auc_score(labels, scores),
        "ap": average_precision_score(labels, scores),
        "acc@0.5": accuracy_score(labels, binary_preds),
        "mean_score": float(np.mean(scores)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--submit-files", nargs="+", required=True)
    parser.add_argument("--output-path", default=None)
    opts = parser.parse_args()

    gt = load_ground_truth(opts.gt_path)
    rows = [evaluate_one(gt, submit_file) for submit_file in opts.submit_files]
    result = pd.DataFrame(rows).sort_values("auc", ascending=False)

    print(result.to_string(index=False))
    if opts.output_path:
        os.makedirs(os.path.dirname(opts.output_path), exist_ok=True)
        result.to_excel(opts.output_path, index=False)


if __name__ == "__main__":
    main()
