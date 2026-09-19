#!/usr/bin/env python3
"""Offline, lossless conversion of Task 3 lemon-to-goblet episodes to LeRobot v2.1.

The converter never changes raw captures. It writes a staged derived dataset and
publishes it only after every selected episode has passed the checks below.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
DATASET_NAME = "uf850_task3_lemon_goblet_lerobot_v001"
RAW_ROOT = Path(os.environ.get("UF850_RAW_ROOT", str(PACKAGE / "raw_captures")))
TASK_SPEC = RAW_ROOT.parent / "TASK_SPEC.json"
FPS = 30
VECTOR_COLUMNS = [0, 1, 2, 3, 4, 5, 7]
REQUIRED = ("samples.parquet", "front.mp4", "wrist.mp4")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        key: np.atleast_1d(value).tolist()
        for key, value in {
            "min": values.min(0),
            "max": values.max(0),
            "mean": values.mean(0),
            "std": values.std(0),
            "count": np.array([len(values)]),
        }.items()
    }


def map_vector(values):
    values = np.asarray(list(values), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all():
        raise ValueError("Expected a finite raw 8-slot vector")
    if not np.all(values[:, 6] == 0):
        raise ValueError("Raw SDK slot 6 is nonzero; refusing an unsafe 8-to-7 mapping")
    return values[:, VECTOR_COLUMNS].astype(np.float32)


def source_episodes(task):
    sources = []
    for manifest_path in sorted(RAW_ROOT.glob("session_*/episodes/episode_*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        episode = manifest_path.parent
        if manifest.get("task") != task:
            continue
        if manifest.get("termination_reason") != "operator_b_end":
            raise ValueError(f"Episode did not end with B: {episode}")
        if manifest.get("samples", 0) <= 0:
            raise ValueError(f"Empty episode: {episode}")
        expected = manifest.get("files", {})
        source_hashes = {}
        for name in REQUIRED:
            file = episode / name
            if not file.is_file() or file.stat().st_size == 0:
                raise ValueError(f"Missing raw file: {file}")
            source_hashes[name] = digest(file)
            if expected.get(name) != source_hashes[name]:
                raise ValueError(f"Raw hash mismatch: {file}")
        sources.append(
            {
                "episode": episode,
                "manifest_path": manifest_path,
                "manifest": manifest,
                "manifest_sha256": digest(manifest_path),
                "files_sha256": source_hashes,
            }
        )
    if not sources:
        raise ValueError(f"No completed raw episodes for task {task!r}")
    return sources


def image_statistics(path, expected_frames):
    capture = cv2.VideoCapture(str(path))
    count = 0
    pixels = []
    try:
        if abs(capture.get(cv2.CAP_PROP_FPS) - FPS) > 1e-4:
            raise ValueError(f"Unexpected video FPS: {path}")
        while True:
            valid, image = capture.read()
            if not valid:
                break
            if image is None or image.shape != (480, 640, 3):
                raise ValueError(f"Unexpected video frame: {path}")
            pixels.append(image[::8, ::8, ::-1].reshape(-1, 3))
            count += 1
    finally:
        capture.release()
    if count != expected_frames:
        raise ValueError(f"{path} decoded {count} frames, expected {expected_frames}")
    result = statistics(np.concatenate(pixels).astype(np.float64) / 255.0)
    for key in ("min", "max", "mean", "std"):
        result[key] = np.array(result[key]).reshape(3, 1, 1).tolist()
    result["count"] = [expected_frames]
    return result


def validate_timing(data):
    times = data["sample_monotonic_s"].to_numpy(dtype=np.float64)
    intervals = np.diff(times)
    if not np.isfinite(times).all() or np.any(intervals <= 0) or intervals.max() > 0.05:
        raise ValueError("Sample clock is invalid or exceeds the 50 ms review threshold")

    timing = [json.loads(value) for value in data["timing_json"]]
    ages = {role: [] for role in ("front", "wrist")}
    state_ages = []
    repeats = {role: 0 for role in ages}
    for index, row in enumerate(timing):
        for name in ("command", "gripper_command"):
            command = row[name]
            if command["sdk_code"] != 0 or command["return_ns"] > row["sample_ns"]:
                raise ValueError(f"Invalid command receipt at sample {index}")
        state_ages.append((row["sample_ns"] - row["state_read_end_ns"]) / 1e6)
        for role in ages:
            camera = row["cameras"][role]
            ages[role].append((row["sample_ns"] - camera["read_end_ns"]) / 1e6)
            if index and camera["sequence"] == timing[index - 1]["cameras"][role]["sequence"]:
                repeats[role] += 1
    if min(state_ages) < 0 or max(state_ages) > 250:
        raise ValueError("State observation is future or too old")
    if any(min(values) < 0 or max(values) > 250 for values in ages.values()):
        raise ValueError("Camera observation is future or too old")
    return {
        "interval_ms_quantiles": np.quantile(intervals * 1000, [0, 0.5, 0.95, 1]).tolist(),
        "elapsed_drift_vs_nominal_ms": float((times[-1] - times[0] - (len(times) - 1) / FPS) * 1000),
        "camera_read_age_ms_p95_max": {
            role: np.quantile(values, [0.95, 1]).tolist() for role, values in ages.items()
        },
        "state_age_ms_p95_max": np.quantile(state_ages, [0.95, 1]).tolist(),
        "repeated_camera_sequences": repeats,
    }


def feature_spec():
    features = {
        key: {
            "dtype": "float32",
            "shape": [7],
            "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
        }
        for key in ("observation.state", "action")
    }
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[key] = {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
    for role in ("front", "wrist"):
        features[f"observation.images.{role}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "mpeg4",
                "video.pix_fmt": "yuv420p",
                "video.fps": 30.0,
                "video.channels": 3,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return features


def main():
    task = json.loads(TASK_SPEC.read_text())["task_prompt_en"]
    sources = source_episodes(task)
    dataset = PACKAGE / "datasets" / "local" / DATASET_NAME
    stage = dataset.with_name(dataset.name + ".partial")
    if dataset.exists() or stage.exists():
        raise FileExistsError(f"Refusing to overwrite {dataset}")
    stage.mkdir(parents=True)

    mapping, episodes, episode_stats, timing_audit = [], [], [], []
    offset = 0
    try:
        for episode_index, source in enumerate(sources):
            raw = source["episode"]
            manifest = source["manifest"]
            data = pd.read_parquet(raw / "samples.parquet")
            count = len(data)
            if count != manifest["samples"] or count < 15:
                raise ValueError(f"Bad sample count: {raw}")
            if not np.array_equal(data["frame_index"].to_numpy(), np.arange(count)):
                raise ValueError(f"Frame index is not contiguous: {raw}")
            state = map_vector(data["observation.state"])
            action = map_vector(data["action"])
            if np.any(action[:, -1] < 0) or np.any(action[:, -1] > 1):
                raise ValueError(f"Commanded gripper is outside [0, 1]: {raw}")
            if np.any(state[:, -1] < -0.01) or np.any(state[:, -1] > 1.01):
                raise ValueError(f"Measured gripper is outside expected range: {raw}")
            audit = validate_timing(data)

            arrays = {
                "observation.state": state,
                "action": action,
                "timestamp": (np.arange(count) / FPS).astype(np.float32),
                "frame_index": np.arange(count, dtype=np.int64),
                "episode_index": np.full(count, episode_index, dtype=np.int64),
                "index": np.arange(offset, offset + count, dtype=np.int64),
                "task_index": np.zeros(count, dtype=np.int64),
            }
            out = stage / f"data/chunk-000/episode_{episode_index:06d}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({key: list(value) if value.ndim == 2 else value for key, value in arrays.items()}).to_parquet(out, index=False)

            current_stats = {key: statistics(value) for key, value in arrays.items()}
            for role in ("front", "wrist"):
                video_key = f"observation.images.{role}"
                current_stats[video_key] = image_statistics(raw / f"{role}.mp4", count)
                copied = stage / f"videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4"
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(raw / f"{role}.mp4", copied)
                if digest(copied) != source["files_sha256"][f"{role}.mp4"]:
                    raise ValueError(f"Video copy hash mismatch: {copied}")

            timing_path = stage / f"provenance/timing/episode_{episode_index:06d}.parquet"
            timing_path.parent.mkdir(parents=True, exist_ok=True)
            data[["frame_index", "sample_monotonic_s", "timing_json"]].to_parquet(timing_path, index=False)
            mapping.append(
                {
                    "episode_index": episode_index,
                    "source_session": raw.parents[1].name,
                    "source_episode": raw.name,
                    "source_path": str(raw),
                    "source_manifest_sha256": source["manifest_sha256"],
                    "source_files_sha256": source["files_sha256"],
                    "samples": count,
                    "termination_reason": manifest["termination_reason"],
                    "task_success": manifest.get("task_success"),
                    "success_source": "unreviewed_operator_end; not an automated success label",
                    "training_eligible": False,
                }
            )
            episodes.append({"episode_index": episode_index, "tasks": [task], "length": count})
            episode_stats.append({"episode_index": episode_index, "stats": current_stats})
            timing_audit.append({"episode_index": episode_index, **audit})
            offset += count
            print(f"{episode_index + 1}/{len(sources)} converted ({offset} frames)", flush=True)

        info = {
            "codebase_version": "v2.1",
            "robot_type": "UF850",
            "total_episodes": len(sources),
            "total_frames": offset,
            "total_tasks": 1,
            "total_videos": 2 * len(sources),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": FPS,
            "splits": {"train": f"0:{len(sources)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": feature_spec(),
        }
        write_json(stage / "meta/info.json", info)
        write_jsonl(stage / "meta/episodes.jsonl", episodes)
        write_jsonl(stage / "meta/episodes_stats.jsonl", episode_stats)
        write_jsonl(stage / "meta/tasks.jsonl", [{"task_index": 0, "task": task}])
        write_json(stage / "provenance/source_map.json", mapping)
        write_json(stage / "provenance/timing_audit.json", timing_audit)
        write_json(
            stage / "provenance/semantics.json",
            {
                "raw_to_effective_indices": VECTOR_COLUMNS,
                "joint_unit": "radian",
                "action": "last SDK-accepted absolute joint target plus commanded gripper pulse/850",
                "state": "SDK joint feedback plus measured gripper pulse/850",
                "gripper": "0 closed, 1 nominal open; measured overshoot is retained",
                "images": "front and wrist videos copied byte-for-byte",
                "timestamp": "frame_index/30 nominal video time; source host timestamps preserved separately",
                "hardware_synchronized": False,
                "success_source": "none: B-end recording is not a task-success verdict",
                "training_eligible": False,
            },
        )
        write_json(
            stage / "provenance/conversion_result.json",
            {
                "episodes": len(sources),
                "frames": offset,
                "videos": 2 * len(sources),
                "raw_hashes_verified": True,
                "all_videos_fully_decoded": True,
                "output_published": False,
            },
        )
        os.replace(stage, dataset)
    except Exception:
        raise

    result = json.loads((dataset / "provenance/conversion_result.json").read_text())
    result["output_published"] = True
    write_json(dataset / "provenance/conversion_result.json", result)
    print(f"Published {dataset}", flush=True)


if __name__ == "__main__":
    main()
