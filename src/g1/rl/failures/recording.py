"""Opt-in pre-reset recording; no training step or termination semantics are replaced."""

import math

import torch

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from g1.tasks.upper_limb_agent.mdp.contacts import torso_force_samples

from .storage import TensorHistory, contact_statistics

CATEGORIES = ("regular", "height_shifted", "planar")


class FailureRecorderTerm(RecorderTerm):
    def record_post_step(self):
        miner = getattr(self._env, "failure_miner", None)
        if miner is not None:
            miner.capture()
        return None, None

    def reset(self, env_ids=None):
        miner = getattr(self._env, "failure_miner", None)
        if miner is not None:
            miner.reset(slice(None) if env_ids is None else env_ids)
        return {}


@configclass
class FailureRecorderCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    failure_frames = RecorderTermCfg(class_type=FailureRecorderTerm)


def prepare_recording(env_cfg, num_envs):
    env_cfg.scene.num_envs = num_envs
    env_cfg.commands.wrist_pose.debug_vis = False
    env_cfg.recorders = FailureRecorderCfg()
    env_cfg.scene.torso_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        update_period=env_cfg.sim.dt,
        history_length=max(3, env_cfg.decimation),
        track_contact_points=True,
        max_contact_data_count_per_prim=32,
    )


def force_category(term, category):
    """Preserve native regular-goal refresh and all reset-time timing calculations."""
    if category == "mixed":
        return
    from types import MethodType

    original = term._sample_task_kind

    def sample(self, ids):
        original(ids)
        self.planar_task_mask[ids] = category == "planar"
        self.height_task_mask[ids] = False
        self.height_shift[ids] = 0
        if category == "height_shifted":
            self._sample_height_task(ids, force=True)

    term._sample_task_kind = MethodType(sample, term)


class FailureMiner:
    def __init__(
        self, env, archive, partners, *, history_s, episodes, max_failures, switch_tracker=None, fixed_pairs=None
    ):
        self.env, self.archive, self.partners = env, archive, partners
        self.term = env.command_manager.get_term("wrist_pose")
        self.robot = env.scene["robot"]
        self.sensor = env.scene["torso_contacts"]
        self.termination_sensor = env.scene["contact_forces"]
        torso_ids, _ = self.termination_sensor.find_bodies("torso_link")
        if len(torso_ids) != 1:
            raise ValueError("Expected one torso body in the original termination sensor")
        self.torso_id = torso_ids[0]
        self.physics_dt = env.cfg.sim.dt
        self.substeps = env.cfg.decimation
        self.episodes, self.max_failures = episodes, max_failures
        self.history = TensorHistory(math.ceil(history_s / env.step_dt), env.num_envs)
        self.switch_tracker = switch_tracker
        self.fixed_pairs = fixed_pairs
        self.elapsed_robot_s = 0.0
        self.unrecorded_completions = 0

    @property
    def done(self):
        return self.archive.completed >= self.episodes or self.archive.saved >= self.max_failures

    def reset(self, ids):
        self.history.reset(ids)
        if self.switch_tracker is not None:
            self.switch_tracker.reset(ids)

    def capture(self):
        if self.done:
            return
        env, term, robot = self.env, self.term, self.robot
        data = self.sensor.data
        pair = data.normal_force_matrix_w_history.torch[:, : self.substeps, 0].flip(1)
        net = data.net_normal_forces_w_history.torch[:, : self.substeps, 0].flip(1)
        if pair.shape[-2] != len(self.partners) or pair.shape[1] != self.substeps:
            raise RuntimeError("Contact filter/history layout differs from the archived channel labels")
        origin = env.scene.env_origins
        poses = robot.data.body_link_pose_w.torch.clone()
        poses[..., :3] -= origin[:, None]
        goals = term.goal_pose_w.clone()
        goals[..., :3] -= origin[:, None]
        frame = {
            "time": term.episode_age_steps().float() * env.step_dt,
            "body_pose": poses,
            "joint_pos": robot.data.joint_pos.torch,
            "joint_vel": robot.data.joint_vel.torch,
            "goal_pose": goals,
            "action": env.action_manager.action,
            "lower_command": env.action_manager.get_term("joint_pos").lower_command,
            "pair_force": pair,
            "net_force": net,
            "termination_force_history": self.termination_sensor.data.net_normal_forces_w_history.torch[
                :, :, self.torso_id
            ],
            "contact_pos": data.contact_pos_w.torch[:, 0] - origin[:, None],
        }
        self.history.push(frame)
        if self.switch_tracker is not None:
            arm_force, _ = torso_force_samples(env, None)
            self.switch_tracker.observe(goals, arm_force[:, : self.substeps].flip(1), env.reset_terminated)
        self.elapsed_robot_s += env.num_envs * env.step_dt
        ended = env.reset_terminated | env.reset_time_outs
        if self.fixed_pairs is not None:
            ended &= self.fixed_pairs.trial_ids >= 0
        ids = ended.nonzero().flatten().tolist()
        accepted = 0
        for index in ids:
            if self.done:
                break
            failed = bool(env.reset_terminated[index])
            reasons = [
                name
                for name in env.termination_manager.active_terms
                if bool(env.termination_manager.get_term(name)[index])
            ]
            course = term.range_curriculum
            episode = {
                "env_id": index,
                "category": CATEGORIES[int(term.task_ids[index])],
                "arm_mode": int(term.mode_id[index]),
                "failed": failed,
                "reasons": reasons,
                "elapsed_s": float(frame["time"][index]),
                "level": int(course.episode_levels[index]),
                "evaluation_start_s": float(term.evaluation_start_steps[index]) * env.step_dt,
                "deadline_s": float(term.episode_deadline_steps[index]) * env.step_dt,
                "goal_pose": goals[index].detach().cpu().tolist(),
                "planar_distance_m": float(term.planar_distance_m[index]),
                "planar_heading_rad": float(term.planar_heading_rad[index]),
                "phase": "evaluation"
                if int(term.episode_age_steps()[index]) >= int(term.evaluation_start_steps[index])
                else "execution",
            }
            if self.fixed_pairs is not None:
                episode.update(self.fixed_pairs.metadata(index))
            trace = self.history.extract(index) if failed else None
            if trace is not None:
                episode["contacts"] = contact_statistics(trace, self.partners, self.physics_dt)
                episode["contacts"]["termination_window_peak_n"] = float(
                    torch.as_tensor(trace["termination_force_history"][-1]).norm(dim=-1).max()
                )
                episode["clip_start_s"] = float(trace["time"][0])
            switches = (
                self.switch_tracker.episode(index, episode["arm_mode"])
                if self.switch_tracker is not None and episode["category"] == "regular"
                else ()
            )
            self.archive.add_episode(episode, trace, switches=switches)
            accepted += 1
        self.unrecorded_completions += len(ids) - accepted
