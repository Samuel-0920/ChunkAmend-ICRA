#!/usr/bin/env python3
"""Verify the published LeRobot v2.1 package without loading any policy weights."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


PACKAGE = Path(__file__).resolve().parents[1]
DATASET = PACKAGE / "datasets/local/uf850_task3_lemon_goblet_lerobot_v001"
REQUIRED = ("samples.parquet", "front.mp4", "wrist.mp4")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    info = json.loads((DATASET / "meta/info.json").read_text())
    mapping = json.loads((DATASET / "provenance/source_map.json").read_text())
    assert info["codebase_version"] == "v2.1"
    assert info["total_episodes"] == len(mapping)
    assert info["total_frames"] == sum(row["samples"] for row in mapping)
    assert all(row["task_success"] is None for row in mapping)
    assert all(row["training_eligible"] is False for row in mapping)

    offset = 0
    for episode_index, source in enumerate(mapping):
        data = pd.read_parquet(DATASET / f"data/chunk-000/episode_{episode_index:06d}.parquet")
        count = len(data)
        assert count == source["samples"]
        assert np.array_equal(data["index"].to_numpy(), np.arange(offset, offset + count))
        assert np.asarray(data["observation.state"].tolist()).shape == (count, 7)
        assert np.asarray(data["action"].tolist()).shape == (count, 7)
        raw = Path(source["source_path"])
        for name in REQUIRED:
            assert digest(raw / name) == source["source_files_sha256"][name]
        for role in ("front", "wrist"):
            copied = DATASET / f"videos/chunk-000/observation.images.{role}/episode_{episode_index:06d}.mp4"
            assert digest(copied) == source["source_files_sha256"][f"{role}.mp4"]
        offset += count

    dataset = LeRobotDataset(
        "local/uf850_task3_lemon_goblet_lerobot_v001",
        root=DATASET,
        delta_timestamps={"action": [index / 30 for index in range(15)]},
        video_backend="pyav",
    )
    assert len(dataset) == info["total_frames"]
    assert dataset.num_episodes == info["total_episodes"]
    positions = sorted({0, len(dataset) - 1, *(sum(row["samples"] for row in mapping[:i]) for i in range(1, len(mapping)))})
    for position in positions:
        sample = dataset[position]
        assert sample["action"].shape == (15, 7)
        assert sample["observation.images.front"].shape == (3, 480, 640)
        assert sample["observation.images.wrist"].shape == (3, 480, 640)
    report = {
        "episodes": info["total_episodes"],
        "frames": len(dataset),
        "all_source_and_copied_hashes_match": True,
        "actual_pinned_lerobot_loader": True,
        "loader_positions_checked": len(positions),
        "model_weights_loaded": False,
        "training_started": False,
    }
    (PACKAGE / "evidence").mkdir(exist_ok=True)
    (PACKAGE / "evidence/lerobot_loader_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
