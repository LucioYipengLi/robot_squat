"""Publish per-case Q timelines and a portable archive for the project viewer."""

import csv
import json
import shutil

from .jobs import result_path, write_json


def write_q_outputs(output_dir, report_dir, results):
    rows, replays = [], []
    for case in results:
        labels = {
            "checkpoint": case["checkpoint"]["label"],
            "condition": case["condition_id"],
            "policy_output": case["policy_output"],
            "seed": case["seed"],
        }
        metadata = case["metadata"]
        replay = metadata.get("q_replay")
        for episode in case["episodes"]:
            rows.extend({**labels, "case_id": episode["case_id"], **window} for window in episode.get("timeline", ()))
            if replay and episode["case_id"] == replay["case_id"]:
                replays.append((case, {**labels, **episode}, replay))

    paths = {}
    if rows:
        path = report_dir / "timeline.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paths["timeline_csv"] = str(path)
    if replays:
        directory = report_dir / "replay"
        clips = directory / "clips"
        clips.mkdir(parents=True)
        geometry_path = output_dir / "replay_geometry.json"
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        shutil.copyfile(geometry_path, directory / "geometry.json")
        first = replays[0][0]["metadata"]
        write_json(
            directory / "run.json",
            {
                "schema_version": 1,
                "archive_kind": "upper_q_eval",
                "control_dt": first["control_dt"],
                "level": first["level"],
                "body_names": geometry["body_names"],
                "joint_names": replays[0][2]["joint_names"],
            },
        )
        with (directory / "clips.jsonl").open("w", encoding="utf-8") as stream:
            for index, (case, sample, replay) in enumerate(replays):
                if (
                    replay["body_names"] != geometry["body_names"]
                    or replay["frames"] != sample["steps"] + 1
                    or case["metadata"]["control_dt"] != first["control_dt"]
                    or replay["joint_names"] != replays[0][2]["joint_names"]
                ):
                    raise ValueError(f"Incompatible or incomplete Q replay: {sample['case_id']}")
                sample_id = f"{index:06d}"
                source = result_path(output_dir, case).with_name("q_replay.npz")
                shutil.copyfile(source, clips / f"{sample_id}.npz")
                stream.write(json.dumps({"sample_id": sample_id, **sample}, allow_nan=False) + "\n")
        write_json(directory / "summary.json", {"completed_episodes": sum(len(case["episodes"]) for case in results)})
        paths["replay_dir"] = str(directory)
    return paths
