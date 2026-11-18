"""Portable regular-switch suites and paired outcome summaries."""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

COORDINATE_FRAME = "world axes minus env origin"
SWITCH_S = 4.0
AFTER_S = 1.0
GROUPS = ("in_up", "out_up", "in_flat")
HARD_CONTACT = "contact_over_hard_threshold"
REACHED = "new_goal_reached_within_1s"


def load_switch_pairs(path):
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(suite, dict)
        or suite.get("schema_version") != 1
        or suite.get("coordinate_frame") != COORDINATE_FRAME
        or type(suite.get("level")) is not int
        or suite["level"] < 0
        or not isinstance(suite.get("pairs"), list)
        or not suite["pairs"]
    ):
        raise ValueError("Expected a version 1 switch-pair suite with a level and nonempty pairs")
    seen = set()
    for pair in suite["pairs"]:
        if (
            not isinstance(pair, dict)
            or not isinstance(pair.get("id"), str)
            or not pair["id"]
            or pair["id"] in seen
            or not isinstance(pair.get("group"), str)
            or not pair["group"]
        ):
            raise ValueError("Each switch pair requires a unique string id and a group")
        seen.add(pair["id"])
        for key in ("right_old_pose", "right_new_pose"):
            pose = np.asarray(pair.get(key), dtype=float)
            if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-6:
                raise ValueError(f"{pair['id']}: {key} must be a finite xyz + nonzero xyzw pose")
    return suite


def generate_switch_pairs(run, output, *, in_up, controls, regressions, seed):
    """Select target pairs by geometry, with known failures in a separate group."""
    manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if manifest.get("coordinate_frame") != COORDINATE_FRAME:
        raise ValueError("Switch archive must use world axes minus the environment origin")
    candidates = {group: [] for group in GROUPS}
    with (run / "switches.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (
                row["arm_mode"] != "right_only"
                or row["level"] != manifest["level"]
                or not row["pre_switch_ready"]
                or row["pre_switch_contact"]
                or row["position_jump_bin"] != "8_to_12cm"
            ):
                continue
            _, dy, dz = row["active_wrist_delta_initial_pelvis_m"][0]
            if dy >= 0.03 and dz >= 0.03:
                candidates["in_up"].append(row)
            elif dy <= -0.03 and dz >= 0.03:
                candidates["out_up"].append(row)
            elif dy >= 0.03 and abs(dz) < 0.03:
                candidates["in_flat"].append(row)

    rng = random.Random(seed)
    selected = []
    for group, count in zip(GROUPS, (in_up, controls, controls), strict=True):
        pool = candidates[group]
        if len(pool) < count:
            raise ValueError(f"{group}: requested {count} pairs but found only {len(pool)}; reduce the requested count")
        selected.extend((group, row) for row in rng.sample(pool, count))
    # Mark known failures separately. Some are already part of the outcome-blind main sample.
    seen = {(row["episode_id"], row["switch_index"]) for _, row in selected}
    known = [
        row for row in candidates["in_up"] if row[HARD_CONTACT] and (row["episode_id"], row["switch_index"]) not in seen
    ]
    selected.extend(("known_failure", row) for row in rng.sample(known, min(regressions, len(known))))
    pairs = [
        {
            "id": f"{group}-{row['episode_id']}-{row['switch_index']}",
            "group": group,
            "right_old_pose": row["old_goal_pose"][1],
            "right_new_pose": row["new_goal_pose"][1],
            "source_episode_id": row["episode_id"],
            "source_switch_index": row["switch_index"],
        }
        for group, row in selected
    ]
    rng.shuffle(pairs)
    suite = {
        "schema_version": 1,
        "coordinate_frame": COORDINATE_FRAME,
        "level": manifest["level"],
        "source_run": str(run.resolve()),
        "selection_seed": seed,
        "pairs": pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(suite, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {len(pairs)} goal pairs to {output} ({2 * len(pairs)} evaluations)")
    return 0


class PairedSwitchSummary:
    def __init__(self, pairs):
        self.groups = {pair["id"]: pair["group"] for pair in pairs}
        self.trials = defaultdict(dict)

    def add(self, episode, switches):
        pair_id, side = episode["pair_id"], episode["pair_side"]
        if pair_id not in self.groups or side in self.trials[pair_id] or len(switches) > 1:
            raise ValueError(f"Unexpected or duplicate switch-pair trial: {pair_id}/{side}")
        row = switches[0] if switches else {}
        self.trials[pair_id][side] = {
            "episode_id": episode["episode_id"],
            "failed": episode["failed"],
            "switched": bool(switches),
            "pre_switch_ready": row.get("pre_switch_ready"),
            "pre_switch_contact": row.get("pre_switch_contact"),
            HARD_CONTACT: row.get(HARD_CONTACT),
            REACHED: row.get(REACHED),
        }

    @staticmethod
    def _outcome(trials):
        if len(trials) < 2:
            return "incomplete"
        if any(
            not trial["pre_switch_ready"] or trial["pre_switch_contact"] or trial[HARD_CONTACT] is None
            for trial in trials.values()
        ):
            return "not_comparable"
        left, right = (trials[side][HARD_CONTACT] for side in ("left", "right"))
        if left and right:
            return "both_contact"
        if left or right:
            return "left_only_contact" if left else "right_only_contact"
        return "neither_contact"

    def report(self):
        rows = []
        for pair_id, group in self.groups.items():
            trials = self.trials.get(pair_id, {})
            rows.append({"pair_id": pair_id, "group": group, "outcome": self._outcome(trials), "trials": trials})
        return {
            "overall": self._summarize(rows),
            "by_group": {
                group: self._summarize([row for row in rows if row["group"] == group])
                for group in sorted(set(self.groups.values()))
            },
            "by_pair": rows,
        }

    @staticmethod
    def _summarize(rows):
        outcomes = (
            "right_only_contact",
            "left_only_contact",
            "both_contact",
            "neither_contact",
            "not_comparable",
            "incomplete",
        )
        result = {"pairs": len(rows), **{name: sum(row["outcome"] == name for row in rows) for name in outcomes}}
        result["comparable_pairs"] = len(rows) - result["not_comparable"] - result["incomplete"]
        for side in ("left", "right"):
            trials = [row["trials"][side] for row in rows if side in row["trials"]]
            result[side] = {
                "completed": len(trials),
                "failed_before_switch": sum(t["failed"] and not t["switched"] for t in trials),
                "not_ready_at_switch": sum(t["pre_switch_ready"] is False for t in trials),
                "already_touching_at_switch": sum(t["pre_switch_contact"] is True for t in trials),
                "early_hard_contacts": sum(t[HARD_CONTACT] is True for t in trials),
                "contact_assessed": sum(t[HARD_CONTACT] is not None for t in trials),
                "reached_within_1s": sum(t[REACHED] is True for t in trials),
                "reach_assessed": sum(t[REACHED] is not None for t in trials),
            }
        return result
