#!/usr/bin/env python3
"""Run RP-net inference for one synchronized pair of radar CSV files.

Example:
    python3 inference.py --checkpoint models/epoch_199.pth \
        --radar0 data/scene13/radar_0/000000.csv \
        --radar1 data/scene13/radar_1/000000.csv \
        --output predictions/scene13_000000.json

The checkpoint and CUDA extensions must be compatible with the environment used
for training.  RP-net's custom operators are CUDA-only, so CPU inference is not
supported by this project.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pointillism import pointillism
from net.refinement import Refinement
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import lib.utils.kitti_utils as kitti_utils


N_POINTS = 70
N_CHANNELS = 8
MIN_POINTS = 20


def filter_points(points, radius):
    """Match the origin-noise filter from data_generator.py."""
    distance = np.sqrt(np.sum(points[:, :3] ** 2, axis=1))
    return points[distance > radius]


def read_radar_csv(path, sensor_offset):
    """Read one raw radar CSV and reproduce the training feature ordering."""
    radar = pd.read_csv(path, delimiter=",").values
    if radar.ndim != 2 or radar.shape[1] < 10:
        raise ValueError("%s must contain at least 10 columns" % path)

    # Keep precisely the columns selected by data_generator.py, including its
    # coordinate convention and the lateral sensor-offset compensation.
    radar = radar[:, [0, 1, 2, 4, 3, 5, 6, 7, 8, 9]].astype(np.float32)
    radar[:, 3] += sensor_offset
    points = radar[:, [2, 3, 4, 6, 9]]
    points[:, 2] = -radar[:, 4]
    points = filter_points(points, radius=0.5)
    return points[(points[:, 2] > -1) & (points[:, 2] < 2)]


def prepare_input(radar0_path, radar1_path, seed):
    """Create the one (70, 8) RP-net input array from two radar frames."""
    radar0 = read_radar_csv(radar0_path, sensor_offset=-0.51)
    radar1 = read_radar_csv(radar1_path, sensor_offset=0.51)

    generator = pointillism()
    potential0, potential1 = generator.find_llpc(radar0, radar1, 0.5, 1, True)
    radar0 = np.hstack((radar0, potential0.reshape(-1, 1),
                        np.tile([1, 0], (radar0.shape[0], 1))))
    radar1 = np.hstack((radar1, potential1.reshape(-1, 1),
                        np.tile([0, 1], (radar1.shape[0], 1))))
    points = np.concatenate((radar0, radar1), axis=0)
    points[:, 0] = -points[:, 0]

    if len(points) < MIN_POINTS:
        raise ValueError(
            "Only %d valid points remain; RP-net requires at least %d"
            % (len(points), MIN_POINTS)
        )

    # RandomState is available in older NumPy releases commonly paired with
    # Python 3.7 / PyTorch 1.4 environments.
    rng = np.random.RandomState(seed)
    if len(points) >= N_POINTS:
        indices = rng.choice(len(points), N_POINTS, replace=False)
        points = points[indices]
    else:
        extra = rng.choice(len(points), N_POINTS - len(points), replace=True)
        points = np.concatenate((points, points[extra]), axis=0)

    if points.shape != (N_POINTS, N_CHANNELS):
        raise RuntimeError("Unexpected preprocessed input shape: %s" % (points.shape,))
    return points.astype(np.float32)


def load_model(checkpoint_path, device):
    """Load either this repository's serialized model or a state-dict checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
        model = Refinement(k=7)
        model.load_state_dict(state_dict)
    else:
        raise TypeError("Unsupported checkpoint type: %s" % type(checkpoint).__name__)

    return model.to(device).eval()


def predict(model, points, device, score_threshold, nms_threshold):
    """Return NMS-filtered RP-net boxes as [x, y, z, h, w, l, yaw]."""
    model_input = torch.from_numpy(points.T).unsqueeze(0).to(device)

    # The legacy non-training RPN reads two yaw values despite not using ground
    # truth to select anchors.  Supply a correctly shaped zero placeholder.
    placeholder_labels = torch.zeros((1, 2, 7), dtype=torch.float32, device=device)
    with torch.no_grad():
        residuals, confidences, anchors, _, _, _, _ = model(
            model_input, placeholder_labels, False, 0
        )

        boxes = anchors[0] + residuals[0]
        scores = confidences[:, 1]
        keep = iou3d_utils.nms_gpu(
            kitti_utils.boxes3d_to_bev_torch_orig(boxes), scores, nms_threshold
        )
        if score_threshold is not None:
            keep = keep[scores[keep] >= score_threshold]

    boxes = boxes[keep].detach().cpu().numpy()
    scores = scores[keep].detach().cpu().numpy()
    return [
        {"score": float(score), "box": [float(value) for value in box]}
        for box, score in zip(boxes, scores)
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Run RP-net on one radar_0/radar_1 frame pair.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pth checkpoint")
    parser.add_argument("--radar0", required=True, help="Path to the radar_0 CSV frame")
    parser.add_argument("--radar1", required=True, help="Path to the synchronized radar_1 CSV frame")
    parser.add_argument("--output", required=True, help="JSON file to create")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0)")
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="Discard boxes below this foreground probability")
    parser.add_argument("--nms-threshold", type=float, default=0.1,
                        help="BEV NMS IoU threshold (matches validation)")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = [Path(args.checkpoint), Path(args.radar0), Path(args.radar1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))
    if not torch.cuda.is_available():
        raise RuntimeError("RP-net requires a CUDA-capable PyTorch installation and GPU")

    device = torch.device("cuda")
    points = prepare_input(args.radar0, args.radar1, args.seed)
    model = load_model(args.checkpoint, device)
    predictions = predict(model, points, device, args.score_threshold, args.nms_threshold)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "radar0": str(Path(args.radar0)),
        "radar1": str(Path(args.radar1)),
        "box_format": "[x, y, z, h, w, l, yaw]",
        "score_type": "foreground softmax probability",
        "predictions": predictions,
    }
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print("Wrote %d prediction(s) to %s" % (len(predictions), output))


if __name__ == "__main__":
    main()
