#!/usr/bin/env python3
"""Generate RP-net-compatible training/evaluation inputs from one radar.

The original data_generator.py combines radar_0 and radar_1.  This script
keeps the identical input layout required by RP-net (70 points x 8 channels)
but uses one selected radar only:

    [x, y, z, radar_feature_1, radar_feature_2, potential, is_radar_0, is_radar_1]

There is no cross-radar observation in this setting. DBSCAN-clustered points
receive the configurable ``potential`` value (1.0 by default, matching
find_llpc's missing-radar fallback); DBSCAN noise points are retained with a
potential of 0.0.

Example:
    python3 single_radar_data_generator.py --sensor 0

To make the existing dataset.py load these files without modification, pass
``--output-dir input_files``.  This overwrites files in that directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pointillism import pointillism


N_POINTS = 70
MIN_POINTS = 20
N_CHANNELS = 8
DEFAULT_VALIDATION_SCENES = {16, 19, 20, 34, 36, 38, 41, 43, 50, 55, 58}


def parse_scene_numbers(value):
    """Parse a comma-separated list such as '13,14,16-20'."""
    scenes = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            scenes.update(range(start, end + 1))
        else:
            scenes.add(int(item))
    if not scenes:
        raise argparse.ArgumentTypeError("At least one scene number is required")
    return sorted(scenes)


def get_bbox_from_json(label_path):
    """Match data_generator.py's label representation: [w, h, l, x, y, z, yaw]."""
    with label_path.open(encoding="utf-8") as handle:
        label_data = json.load(handle)
    if not label_data.get("labels"):
        raise ValueError("No labels in %s" % label_path)

    box = label_data["labels"][0]
    center = [box["center"]["x"], box["center"]["y"], box["center"]["z"]]
    dimensions = [box["size"]["y"], box["size"]["z"], box["size"]["x"]]
    yaw = -box["orientation"]["z"]
    label = np.asarray(dimensions + center + [yaw], dtype=np.float32).reshape(1, 7)

    # RP-net's training script assumes two labels per input.  Duplicate the
    # available label exactly as the original generator does.
    return np.pad(label, ((1, 0), (0, 0)), mode="edge")


def filter_points(points, radius=0.5):
    distance = np.sqrt(np.sum(points[:, :3] ** 2, axis=1))
    return points[distance > radius]


def get_dbscan_noise_mask(points, eps, min_samples):
    """Return the DBSCAN noise mask produced by pointillism.py.

    LLP projects points onto the ground plane before calling
    ``get_centroids_dbscan``.  The same projection is used here so clustering
    is based on the horizontal point arrangement, not height.
    """
    projected_points = points.copy()
    projected_points[:, 2] = 0
    _, _, _, _, labels = pointillism().get_centroids_dbscan(
        projected_points[:, :3], eps, min_samples
    )
    return labels == -1


def read_single_radar_frame(csv_path, sensor, potential, dbscan_eps, dbscan_min_samples):
    """Reproduce the selected-radar preprocessing from data_generator.py."""
    radar = pd.read_csv(csv_path, delimiter=",").values
    if radar.ndim != 2 or radar.shape[1] < 10:
        raise ValueError("%s must contain at least 10 columns" % csv_path)

    radar = radar[:, [0, 1, 2, 4, 3, 5, 6, 7, 8, 9]].astype(np.float32)
    radar[:, 3] += -0.51 if sensor == 0 else 0.51
    points = radar[:, [2, 3, 4, 6, 9]]
    points[:, 2] = -radar[:, 4]
    points = filter_points(points)
    points = points[(points[:, 2] > -1) & (points[:, 2] < 2)]
    if len(points):
        noise_mask = get_dbscan_noise_mask(points, dbscan_eps, dbscan_min_samples)
    else:
        noise_mask = np.empty(0, dtype=bool)

    if sensor == 0:
        source_flags = np.tile([1.0, 0.0], (len(points), 1))
    else:
        source_flags = np.tile([0.0, 1.0], (len(points), 1))
    # Retain all points. DBSCAN noise is communicated to RP-net through the
    # potential channel rather than being discarded; clustered points receive
    # the configured single-radar potential and isolated points receive 0.0.
    point_potential = np.full((len(points), 1), potential, dtype=np.float32)
    point_potential[noise_mask] = 0.0
    points = np.hstack((points, point_potential, source_flags))

    # The original pipeline applies this coordinate convention only after
    # combining the two radars; it must remain present for one-radar inputs.
    points[:, 0] = -points[:, 0]
    return points.astype(np.float32)


def fixed_size_points(points, rng):
    if len(points) < MIN_POINTS:
        return None
    if len(points) >= N_POINTS:
        return points[rng.choice(len(points), N_POINTS, replace=False)]
    repeated_indices = rng.choice(len(points), N_POINTS - len(points), replace=True)
    return np.concatenate((points, points[repeated_indices]), axis=0)


def frame_number(path):
    try:
        return int(path.stem)
    except ValueError as error:
        raise ValueError("Radar filename must be numeric, got %s" % path.name) from error


def parse_args():
    parser = argparse.ArgumentParser(description="Generate RP-net input_files from a single radar.")
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                        help="Directory containing scene*/ folders (default: data)")
    parser.add_argument("--sensor", type=int, choices=(0, 1), required=True,
                        help="Use radar_0 or radar_1")
    parser.add_argument("--scenes", type=parse_scene_numbers, default=parse_scene_numbers("13-60"),
                        help="Scene numbers, e.g. 13-60 or 13,14,20 (default: 13-60)")
    parser.add_argument("--output-dir", type=Path, default=Path("single_radar_input_files"),
                        help="Output folder (default: single_radar_input_files)")
    parser.add_argument("--potential", type=float, default=1.0,
                        help="Constant LLP/potential feature for one-radar data (default: 1.0)")
    parser.add_argument("--dbscan-eps", type=float, default=0.5,
                        help="DBSCAN neighbourhood radius used by pointillism.py (default: 0.5)")
    parser.add_argument("--dbscan-min-samples", type=int, default=2,
                        help="Minimum points per DBSCAN cluster; DBSCAN noise receives potential 0.0 (default: 2)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for point sampling and train shuffle")
    parser.add_argument("--validation-scenes", type=parse_scene_numbers,
                        default=sorted(DEFAULT_VALIDATION_SCENES),
                        help="Comma-separated validation scene numbers")
    return parser.parse_args()


def main():
    args = parse_args()
    # RandomState supports the older NumPy versions commonly used alongside
    # Python 3.7 and the repository's original PyTorch 1.4 environment.
    rng = np.random.RandomState(args.seed)
    validation_scenes = set(args.validation_scenes)
    inputs, labels, scene_numbers, frame_numbers, validation_indices = [], [], [], [], []
    skipped = {"missing_label": 0, "bad_or_empty": 0, "too_few_points": 0}

    for scene in args.scenes:
        radar_dir = args.data_root / ("scene%d" % scene) / ("radar_%d" % args.sensor)
        if not radar_dir.is_dir():
            print("Skipping missing directory: %s" % radar_dir)
            continue
        for csv_path in sorted(radar_dir.glob("*.csv"), key=frame_number):
            number = frame_number(csv_path)
            label_path = args.data_root / ("scene%d" % scene) / "label" / ("%06d.json" % number)
            if not label_path.is_file():
                skipped["missing_label"] += 1
                continue
            try:
                points = fixed_size_points(
                    read_single_radar_frame(
                        csv_path, args.sensor, args.potential,
                        args.dbscan_eps, args.dbscan_min_samples
                    ), rng
                )
                if points is None:
                    skipped["too_few_points"] += 1
                    continue
                label = get_bbox_from_json(label_path)
            except (ValueError, KeyError, json.JSONDecodeError, pd.errors.ParserError) as error:
                print("Skipping %s: %s" % (csv_path, error))
                skipped["bad_or_empty"] += 1
                continue

            if points.shape != (N_POINTS, N_CHANNELS):
                raise RuntimeError("Unexpected point shape %s for %s" % (points.shape, csv_path))
            index = len(inputs)
            inputs.append(points)
            labels.append(label)
            scene_numbers.append(scene)
            frame_numbers.append(number)
            if scene in validation_scenes:
                validation_indices.append(index)

    if not inputs:
        raise RuntimeError("No usable frames were generated. Check --data-root, --sensor, and labels.")

    inputs = np.asarray(inputs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    all_indices = np.arange(len(inputs), dtype=np.int64)
    train_indices = np.setdiff1d(all_indices, validation_indices, assume_unique=True)
    rng.shuffle(train_indices)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "inputs.npy", inputs)
    np.save(args.output_dir / "labels.npy", labels)
    np.save(args.output_dir / "test_nos.npy", np.asarray(scene_numbers, dtype=np.int64))
    np.save(args.output_dir / "frame_nos.npy", np.asarray(frame_numbers, dtype=np.int64))
    np.save(args.output_dir / "val_indices.npy", validation_indices)
    np.save(args.output_dir / "train_indices.npy", train_indices)

    print("Generated %d frames from radar_%d in %s" % (len(inputs), args.sensor, args.output_dir))
    print("Train frames: %d | validation frames: %d" % (len(train_indices), len(validation_indices)))
    print("Skipped: %s" % skipped)


if __name__ == "__main__":
    main()
