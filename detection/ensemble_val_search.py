import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def normalize_name(name):
    return os.path.splitext(str(name).strip())[0]


def load_gt(path):
    gt = pd.read_excel(path)
    gt["key"] = gt["img_names"].map(normalize_name)
    return gt[["key", "labels"]]


def load_pred(path):
    pred = pd.read_excel(path, sheet_name="predictions")
    pred["key"] = pred["img_names"].map(normalize_name)
    name = os.path.splitext(os.path.basename(path))[0]
    return name, pred[["key", "predictions"]].rename(columns={"predictions": name})


def iter_weights(n, step):
    units = int(round(1.0 / step))
    for parts in itertools.product(range(units + 1), repeat=n):
        if sum(parts) == units:
            yield np.array(parts, dtype=np.float64) / float(units)


def evaluate(labels, scores):
    pred = (scores >= 0.5).astype(np.int64)
    return {
        "auc": roc_auc_score(labels, scores),
        "ap": average_precision_score(labels, scores),
        "acc@0.5": accuracy_score(labels, pred),
        "mean_score": float(np.mean(scores)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--submit-files", nargs="+", required=True)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-path", default=None)
    opts = parser.parse_args()

    data = load_gt(opts.gt_path)
    model_names = []
    for path in opts.submit_files:
        name, pred = load_pred(path)
        model_names.append(name)
        data = data.merge(pred, on="key", how="left")
        if data[name].isna().any():
            raise RuntimeError("{} has missing predictions".format(path))

    labels = data["labels"].to_numpy().astype(np.int64)
    pred_matrix = data[model_names].to_numpy(dtype=np.float64)

    rows = []
    for weights in iter_weights(len(model_names), opts.step):
        scores = pred_matrix.dot(weights)
        row = evaluate(labels, scores)
        row.update({"weights": " ".join("{:.2f}".format(x) for x in weights)})
        for model_name, weight in zip(model_names, weights):
            row["w_" + model_name] = weight
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(["auc", "ap"], ascending=False)
    columns = ["auc", "ap", "acc@0.5", "mean_score", "weights"] + [
        "w_" + name for name in model_names
    ]
    print(result[columns].head(opts.top_k).to_string(index=False))

    if opts.output_path:
        os.makedirs(os.path.dirname(opts.output_path), exist_ok=True)
        result.to_excel(opts.output_path, index=False)


if __name__ == "__main__":
    main()
