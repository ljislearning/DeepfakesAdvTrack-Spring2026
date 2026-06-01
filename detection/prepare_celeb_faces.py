import argparse
import csv
import os
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm


REAL_DIRS = {"YouTube-real", "Celeb-real"}
FAKE_DIRS = {"Celeb-synthesis"}
DETECTOR = None
WORKER_OPTS = None


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/duyijie/DeepfakesAdvTrack-Spring2026/detection/dataset/Celeb")
    parser.add_argument("--output-root", default="/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/data/Celeb_faces")
    parser.add_argument("--frames-per-video", type=int, default=12)
    parser.add_argument("--official-test-list", default=None)
    parser.add_argument("--face-scale", type=float, default=1.3)
    parser.add_argument("--min-face-size", type=int, default=60)
    parser.add_argument("--image-ext", default="png", choices=["png", "jpg"])
    parser.add_argument("--num-workers", type=int, default=1)
    return parser.parse_args()


def collect_videos(data_root):
    samples = []
    for dirname in sorted(os.listdir(data_root)):
        folder = os.path.join(data_root, dirname)
        if not os.path.isdir(folder):
            continue
        if dirname in REAL_DIRS:
            label = 0
        elif dirname in FAKE_DIRS:
            label = 1
        else:
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                samples.append((os.path.join(folder, name), label))
    if not samples:
        raise RuntimeError("No videos found under {}".format(data_root))
    return samples


def load_official_val_set(list_path):
    if not list_path:
        return set()
    val_rel_paths = set()
    with open(list_path, "r") as reader:
        for line in reader:
            parts = line.strip().split()
            if len(parts) >= 2:
                val_rel_paths.add(parts[1])
    return val_rel_paths


def get_detector():
    try:
        import dlib
    except ImportError as exc:
        raise RuntimeError(
            "dlib is required for this preprocessing script. "
            "Run it with the DeepfakeBench conda env, where dlib is installed."
        ) from exc
    return dlib.get_frontal_face_detector()


def init_worker(worker_opts):
    global DETECTOR, WORKER_OPTS
    WORKER_OPTS = worker_opts
    DETECTOR = get_detector()


def get_largest_face(detector, frame, min_face_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    candidates = []
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        w, h = x2 - x1, y2 - y1
        if w >= min_face_size and h >= min_face_size:
            candidates.append((x1, y1, x2, y2))
    if not candidates:
        return None
    return max(candidates, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))


def crop_square(frame, box, scale):
    height, width = frame.shape[:2]
    if box is None:
        center_x = width / 2.0
        center_y = height / 2.0
        size = min(width, height)
    else:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        size = int(max(x2 - x1, y2 - y1) * scale)

    x1 = max(int(center_x - size / 2.0), 0)
    y1 = max(int(center_y - size / 2.0), 0)
    size = min(size, width - x1, height - y1)
    x2 = x1 + size
    y2 = y1 + size
    return frame[y1:y2, x1:x2]


def sample_frame_indices(video_path, frames_per_video):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: {}".format(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count <= 0:
        return [0]
    count = min(frames_per_video, frame_count)
    return np.linspace(0, frame_count - 1, count).round().astype(int).tolist()


def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: {}".format(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read frame {} from {}".format(frame_idx, video_path))
    return frame


def save_rows(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as writer:
        csv_writer = csv.writer(writer)
        csv_writer.writerow(["path", "label", "video", "frame_idx"])
        csv_writer.writerows(rows)


def process_video(task):
    video_path, label, rel_video, split = task
    frame_indices = sample_frame_indices(video_path, WORKER_OPTS["frames_per_video"])
    video_stem = os.path.splitext(rel_video)[0]
    save_dir = os.path.join(WORKER_OPTS["output_root"], split, video_stem)
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    failures = 0
    for frame_idx in frame_indices:
        try:
            frame = read_frame(video_path, frame_idx)
            face = get_largest_face(DETECTOR, frame, WORKER_OPTS["min_face_size"])
            crop = crop_square(frame, face, WORKER_OPTS["face_scale"])
            if crop.size == 0:
                failures += 1
                continue
            image_name = "{:06d}.{}".format(frame_idx, WORKER_OPTS["image_ext"])
            image_path = os.path.join(save_dir, image_name)
            cv2.imwrite(image_path, crop)
            rows.append([image_path, label, rel_video, frame_idx])
        except Exception:
            failures += 1
    return split, rows, failures


def main():
    opts = get_opts()
    os.makedirs(opts.output_root, exist_ok=True)
    samples = collect_videos(opts.data_root)
    official_val_set = load_official_val_set(opts.official_test_list)

    tasks = []
    for video_path, label in samples:
        rel_video = os.path.relpath(video_path, opts.data_root)
        split = "val" if rel_video in official_val_set else "train"
        tasks.append((video_path, label, rel_video, split))

    worker_opts = {
        "output_root": opts.output_root,
        "frames_per_video": opts.frames_per_video,
        "face_scale": opts.face_scale,
        "min_face_size": opts.min_face_size,
        "image_ext": opts.image_ext,
    }

    train_rows, val_rows = [], []
    failures = 0
    if opts.num_workers <= 1:
        init_worker(worker_opts)
        iterator = map(process_video, tasks)
        for split, rows, failed in tqdm(iterator, total=len(tasks), desc="crop videos"):
            if split == "val":
                val_rows.extend(rows)
            else:
                train_rows.extend(rows)
            failures += failed
    else:
        with Pool(processes=opts.num_workers, initializer=init_worker, initargs=(worker_opts,)) as pool:
            iterator = pool.imap_unordered(process_video, tasks)
            for split, rows, failed in tqdm(iterator, total=len(tasks), desc="crop videos"):
                if split == "val":
                    val_rows.extend(rows)
                else:
                    train_rows.extend(rows)
                failures += failed

    train_csv = os.path.join(opts.output_root, "train.csv")
    val_csv = os.path.join(opts.output_root, "val.csv")
    save_rows(train_rows, train_csv)
    save_rows(val_rows, val_csv)
    print("saved train csv: {} rows={}".format(train_csv, len(train_rows)))
    print("saved val csv: {} rows={}".format(val_csv, len(val_rows)))
    print("frame failures: {}".format(failures))


if __name__ == "__main__":
    main()
