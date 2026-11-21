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
RETRACT_BINS = ("5_to_8cm", "8_to_12cm")
DUAL_RETRACT_BINS = ("8_to_12cm", "12cm_and_over")
HARD_CONTACT = "contact_over_hard_threshold"
REACHED = "new_goal_reached_within_1s"


def load_switch_pairs(path):
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(suite, dict)
        or suite.get("schema_version") not in (1, 2)
        or suite.get("coordinate_frame") != COORDINATE_FRAME
        or type(suite.get("level")) is not int
        or suite["level"] < 0
        or not isinstance(suite.get("pairs"), list)
        or not suite["pairs"]
    ):
        raise ValueError("Expected a version 1 or 2 switch-pair suite with a level and nonempty pairs")
    dual = suite["schema_version"] == 2
    pose_keys = ("old_poses", "new_poses") if dual else ("right_old_pose", "right_new_pose")
    pose_shape = (2, 7) if dual else (7,)
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
        for key in pose_keys:
            pose = np.asarray(pair.get(key), dtype=float)
            if (
                pose.shape != pose_shape
                or not np.isfinite(pose).all()
                or (np.linalg.norm(pose[..., 3:], axis=-1) < 1e-6).any()
            ):
                raise ValueError(
                    f"{pair['id']}: {key} must have shape {pose_shape} and finite xyz + nonzero xyzw poses"
                )
    return suite


def generate_switch_pairs(run, output, *, in_up, controls, regressions, seed, suite="inward", per_bin=50):
    """Select target pairs by geometry, with known failures in a separate group."""
    manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if manifest.get("coordinate_frame") != COORDINATE_FRAME:
        raise ValueError("Switch archive must use world axes minus the environment origin")
    retract = suite in ("retract", "dual_retract")
    dual = suite == "dual_retract"
    bins = DUAL_RETRACT_BINS if dual else RETRACT_BINS
    arm_mode = "both" if dual else "right_only"
    group_counts = (
        {f"{suite}_{bin_name}": per_bin for bin_name in bins}
        if retract
        else dict(zip(GROUPS, (in_up, controls, controls), strict=True))
    )
    candidates = {group: [] for group in group_counts}
    with (run / "switches.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (
                row["arm_mode"] != arm_mode
                or row["level"] != manifest["level"]
                or not row["pre_switch_ready"]
                or row["pre_switch_contact"]
            ):
                continue
            jump_bin = row["position_jump_bin"]
            if retract:
                if row["direction"] == "retract" and jump_bin in bins:
                    if dual:
                        delta = row["active_wrist_delta_initial_pelvis_m"]
                        if len(delta) != 2 or any(wrist[0] > -0.03 for wrist in delta):
                            continue
                    candidates[f"{suite}_{jump_bin}"].append(row)
                continue
            if jump_bin != "8_to_12cm":
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
    for group, count in group_counts.items():
        pool = candidates[group]
        if len(pool) < count:
            raise ValueError(f"{group}: requested {count} pairs but found only {len(pool)}; reduce the requested count")
        selected.extend((group, row) for row in rng.sample(pool, count))
    # Mark known failures separately. Some are already part of the outcome-blind main sample.
    seen = {(row["episode_id"], row["switch_index"]) for _, row in selected}
    regression_groups = group_counts if retract else ("in_up",)
    known = [
        row
        for group in regression_groups
        for row in candidates[group]
        if row[HARD_CONTACT] and (row["episode_id"], row["switch_index"]) not in seen
    ]
    selected.extend(("known_failure", row) for row in rng.sample(known, min(regressions, len(known))))
    pairs = [
        {
            "id": f"{group}-{row['episode_id']}-{row['switch_index']}",
            "group": group,
            **(
                {"old_poses": row["old_goal_pose"], "new_poses": row["new_goal_pose"]}
                if dual
                else {"right_old_pose": row["old_goal_pose"][1], "right_new_pose": row["new_goal_pose"][1]}
            ),
            "source_episode_id": row["episode_id"],
            "source_switch_index": row["switch_index"],
        }
        for group, row in selected
    ]
    rng.shuffle(pairs)
    document = {
        "schema_version": 2 if dual else 1,
        "coordinate_frame": COORDINATE_FRAME,
        "level": manifest["level"],
        "source_run": str(run.resolve()),
        "selection_seed": seed,
        "selection_suite": suite,
        "pairs": pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {len(pairs)} goal pairs to {output} ({(1 if dual else 2) * len(pairs)} evaluations)")
    return 0


class PairedSwitchSummary:
    def __init__(self, pairs, *, dual=False):
        self.groups = {pair["id"]: pair["group"] for pair in pairs}
        self.trials = defaultdict(dict)
        self.dual = dual

    def add(self, episode, switches):
        pair_id, side = episode["pair_id"], episode["pair_side"]
        expected_sides = ("both",) if self.dual else ("left", "right")
        if (
            pair_id not in self.groups
            or side not in expected_sides
            or side in self.trials[pair_id]
            or len(switches) > 1
        ):
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
        if "switch_diagnostic" in episode:
            self.trials[pair_id][side]["switch_diagnostic"] = episode["switch_diagnostic"]

    @staticmethod
    def _comparable(trial):
        diagnostic = trial.get("switch_diagnostic")
        if diagnostic is None:
            started_ready, started_touching = trial["pre_switch_ready"], trial["pre_switch_contact"]
        else:
            started_ready = diagnostic["ready_before_intervention"]
            started_touching = diagnostic["contact_before_intervention"]
        return started_ready and not started_touching and trial[HARD_CONTACT] is not None

    def _outcome(self, trials):
        if len(trials) < (1 if self.dual else 2):
            return "incomplete"
        if not all(self._comparable(trial) for trial in trials.values()):
            return "not_comparable"
        if self.dual:
            return "contact" if trials["both"][HARD_CONTACT] else "clear"
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

    def _summarize(self, rows):
        if self.dual:
            counts = {
                name: sum(row["outcome"] == name for row in rows)
                for name in ("contact", "clear", "not_comparable", "incomplete")
            }
            trials = [row["trials"]["both"] for row in rows if "both" in row["trials"]]
            return {
                "cases": len(rows),
                **counts,
                "comparable_cases": counts["contact"] + counts["clear"],
                **self._trial_counts(trials),
            }
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
            result[side] = self._trial_counts(trials)
        return result

    @staticmethod
    def _trial_counts(trials):
        result = {
            "completed": len(trials),
            "failed_before_switch": sum(t["failed"] and not t["switched"] for t in trials),
            "not_ready_at_switch": sum(t["pre_switch_ready"] is False for t in trials),
            "already_touching_at_switch": sum(t["pre_switch_contact"] is True for t in trials),
            "early_hard_contacts": sum(t[HARD_CONTACT] is True for t in trials),
            "contact_assessed": sum(t[HARD_CONTACT] is not None for t in trials),
            "reached_within_1s": sum(t[REACHED] is True for t in trials),
            "reach_assessed": sum(t[REACHED] is not None for t in trials),
        }
        diagnostics = [t["switch_diagnostic"] for t in trials if "switch_diagnostic" in t]
        if diagnostics:
            retreats = [value for d in diagnostics if (value := d["pelvis_retreat_at_switch_m"]) is not None]
            result.update(
                ready_before_intervention=sum(d["ready_before_intervention"] is True for d in diagnostics),
                stable_last_0p2s=sum(d["stable_last_0p2s"] for d in diagnostics),
                pelvis_retreat_measured=len(retreats),
                pelvis_retreat_at_switch_mean_m=sum(retreats) / len(retreats) if retreats else None,
            )
        return result
