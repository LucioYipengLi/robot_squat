"""COM 指令的 CPU 隔离回归测试；运行 unittest 即可，不启动仿真。

直接执行生产类/函数的 AST，替换仿真管理器、资产和四元数运算依赖。
覆盖张量逻辑、门控、参考系和原切片兼容性，不代替 PhysX 接触及策略训练验收。
"""

from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "source/g1/g1/tasks/manager_based/g1/mdp"


def load_definitions(path, names, namespace):
    """仅执行指定定义，避免导入时启动 Isaac Lab 或污染 sys.modules。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def quat_from_euler_xyz(roll, pitch, yaw):
    """独立解析四元数替身，遵循当前框架的 xyzw 约定。"""
    cr, sr = (roll / 2).cos(), (roll / 2).sin()
    cp, sp = (pitch / 2).cos(), (pitch / 2).sin()
    cy, sy = (yaw / 2).cos(), (yaw / 2).sin()
    return torch.stack((sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                        cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy), dim=-1)


def quat_apply(quat, vector):
    xyz = quat[..., :3]
    cross = 2 * torch.linalg.cross(xyz, vector)
    return vector + quat[..., 3:] * cross + torch.linalg.cross(xyz, cross)


def quat_apply_inverse(quat, vector):
    inverse = quat.clone()
    inverse[..., :3] *= -1
    return quat_apply(inverse, vector)


def yaw_quat(quat):
    x, y, z, w = quat.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zeros = torch.zeros_like(yaw)
    return quat_from_euler_xyz(zeros, zeros, yaw)


MATH = SimpleNamespace(quat_apply=quat_apply, quat_apply_inverse=quat_apply_inverse,
                       quat_from_euler_xyz=quat_from_euler_xyz, yaw_quat=yaw_quat)


class CommandTermStub:
    """保留 reset/compute 的关键调用顺序，物理状态由测试显式更新。"""

    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env
        self.num_envs, self.device = env.num_envs, env.device
        self.metrics = {}
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.long)
        self.time_left = torch.zeros(self.num_envs)

    def _resample(self, ids):
        self._resample_command(ids)
        self.command_counter[ids] += 1

    def reset(self, ids):
        result = {name: values[ids].mean().item() for name, values in self.metrics.items()}
        for values in self.metrics.values():
            values[ids] = 0
        self.command_counter[ids] = 0
        self._resample(ids)
        return result


NAMESPACE = load_definitions(MDP / "commands/commands.py", {"SquatWalkCommand"}, {
    "torch": torch, "math": math, "math_utils": MATH, "CommandTerm": CommandTermStub,
})
Command = NAMESPACE["SquatWalkCommand"]
REWARDS = load_definitions(MDP / "rewards.py", {"track_com_xy_exp", "track_velocity_exp"}, {
    "torch": torch, "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
})
OBS = load_definitions(MDP / "observations.py", {
    "task_command_height", "task_command_velocity", "task_command_com",
}, {"torch": torch})


class Scene(dict):
    pass


def make_term(n=4, **overrides):
    ranges = SimpleNamespace(height_offset=(-0.15, 0.02), lin_vel_x=(-0.3, 0.6),
                             lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.8, 0.8),
                             com_offset_x=(-0.01, 0.01), com_offset_y=(-0.01, 0.01))
    cfg = SimpleNamespace(ranges=ranges, asset_name="robot", rel_mode_envs=(0.4, 0.4, 0.2, 0.0),
                          com_target_speed=0.02, com_zero_probability=0.2, com_success_threshold=0.005,
                          com_foot_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                          envelope_depth_frac=0.8, envelope_vx_floor=0.2, envelope_wz_floor=0.3,
                          height_success_threshold=0.03, vel_xy_success_threshold=0.5,
                          vel_yaw_success_threshold=0.4, cmd_kind=None, element_names=None)
    for name, value in overrides.items():
        if hasattr(ranges, name):
            setattr(ranges, name, value)
        else:
            setattr(cfg, name, value)
    positions = torch.tensor([[0., .12, .03], [0., -.12, .03], [0., 0., .7],
                              [0., .2, .9], [0., -.2, .9]]).repeat(n, 1, 1)
    quats = torch.tensor([0., 0., 0., 1.]).repeat(n, 5, 1)
    wrap = lambda value: SimpleNamespace(torch=value)
    root = torch.zeros(n, 7)
    root[:, 2], root[:, 6] = .75, 1.
    data = SimpleNamespace(body_mass=wrap(torch.tensor([1., 1., 8., 1., 1.]).repeat(n, 1)),
                           body_com_pose_w=wrap(torch.cat((positions.clone(), quats.clone()), dim=-1)),
                           body_pos_w=wrap(positions), body_quat_w=wrap(quats),
                           root_quat_w=wrap(root[:, 3:]), default_root_pose=wrap(root),
                           root_pos_w=wrap(root[:, :3]), root_lin_vel_b=wrap(torch.zeros(n, 3)),
                           root_ang_vel_b=wrap(torch.zeros(n, 3)))
    robot = SimpleNamespace(data=data, find_bodies=lambda names, preserve_order: ([0, 1], names))
    scene = Scene(robot=robot)
    scene.env_origins = torch.zeros(n, 3)
    env = SimpleNamespace(scene=scene, num_envs=n, device="cpu", step_dt=1 / 60,
                          episode_length_buf=torch.ones(n, dtype=torch.long), extras={})
    term = Command(cfg, env)
    env.command_manager = SimpleNamespace(get_term=lambda _: term, get_command=lambda _: term.command)
    term.mode[:] = torch.arange(n) % 4
    return term, env


class ComCommandTests(unittest.TestCase):
    def test_six_dimensional_views_and_observations(self):
        term, env = make_term()
        term.task_command.copy_(torch.arange(24).reshape(4, 6))
        torch.testing.assert_close(OBS["task_command_height"](env), term.task_command[:, :1])
        torch.testing.assert_close(OBS["task_command_velocity"](env), term.task_command[:, 1:4])
        torch.testing.assert_close(OBS["task_command_com"](env), term.task_command[:, 4:6])
        term.com_offset[:] = 0.004
        self.assertTrue(torch.all(term.command[:, 4:6] == 0.004))
        self.assertEqual(len(term.cfg.element_names), 6)

    def test_com_uses_body_centers_not_link_origins(self):
        term, _ = make_term()
        term.robot.data.body_com_pose_w.torch[:, 2, 0] = .12
        self.assertTrue(torch.allclose(term.whole_body_com_w()[:, 0], torch.full((4,), .08)))
        self.assertTrue(torch.all(term.robot.data.body_pos_w.torch[..., 0] == 0))

    def test_mass_scaling_and_nonuniform_randomization(self):
        term, _ = make_term()
        term.robot.data.body_com_pose_w.torch[:, 3, 0] = .24
        before = term.whole_body_com_w().clone()
        term.robot.data.body_mass.torch *= 2
        torch.testing.assert_close(before, term.whole_body_com_w())
        term.robot.data.body_mass.torch[:, 3] *= 3
        self.assertTrue(torch.all(term.whole_body_com_w()[:, 0] > before[:, 0]))

    def test_translation_and_yaw_equivariance(self):
        term, _ = make_term()
        term.robot.data.body_com_pose_w.torch[:, 2, 0] = .03
        term.com_offset[:] = torch.tensor([.004, -.003])
        before = term.com_error_xy().clone()
        yaw = torch.tensor([0., math.pi / 2, math.pi, -2.1])
        q = quat_from_euler_xyz(torch.zeros(4), torch.zeros(4), yaw)
        shift = torch.tensor([[3., 2., 0.], [-4., 1., 0.], [2., -5., 0.], [1., 8., 0.]])
        data = term.robot.data
        data.body_pos_w.torch[:] = quat_apply(q[:, None].expand(-1, 5, -1), data.body_pos_w.torch) + shift[:, None]
        data.body_com_pose_w.torch[..., :3] = quat_apply(
            q[:, None].expand(-1, 5, -1), data.body_com_pose_w.torch[..., :3]) + shift[:, None]
        data.body_quat_w.torch[:] = q[:, None]
        data.root_quat_w.torch[:] = q
        torch.testing.assert_close(term.com_error_xy(), before, atol=1e-6, rtol=1e-5)
        origin, frame = term.com_support_frame_w()
        local_goal = quat_apply_inverse(frame, term.com_target_pos_w() - origin)
        torch.testing.assert_close(local_goal[:, :2], term.com_offset, atol=1e-6, rtol=1e-5)

    def test_circular_mean_across_pi_and_opposite_fallback(self):
        term, _ = make_term()
        angles = torch.tensor([[math.pi - .1, -math.pi + .1], [0., math.pi], [0., 0.], [0., 0.]])
        term.robot.data.body_quat_w.torch[:, :2] = quat_from_euler_xyz(
            torch.zeros_like(angles), torch.zeros_like(angles), angles)
        _, q = term.com_support_frame_w()
        forward = quat_apply(q, torch.tensor([1., 0., 0.]).repeat(4, 1))
        torch.testing.assert_close(forward[0], torch.tensor([-1., 0., 0.]), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(forward[1], torch.tensor([1., 0., 0.]), atol=1e-6, rtol=1e-5)

    def test_root_pitch_does_not_mix_height_into_xy(self):
        term, _ = make_term()
        term.robot.data.root_quat_w.torch[:] = quat_from_euler_xyz(torch.zeros(4), torch.full((4,), .3), torch.zeros(4))
        before = term.com_error_xy().clone()
        term.robot.data.body_com_pose_w.torch[:, 2:, 2] += .3
        torch.testing.assert_close(term.com_error_xy(), before)

    def test_sampling_and_mode_gating(self):
        term, _ = make_term(1024)
        term._resample(torch.arange(1024))
        self.assertTrue(torch.all(term.com_target_offset.abs() <= .01))
        self.assertTrue(torch.all(term.com_target_offset[~term.com_tracking_mask] == 0))
        self.assertTrue(torch.any(term.com_target_offset[term.com_tracking_mask] != 0))
        self.assertTrue(torch.any((term.com_target_offset[term.com_tracking_mask] == 0).all(dim=1)))

    def test_mid_episode_resampling_preserves_velocity_and_squat_walk_height(self):
        term, _ = make_term()
        ids = torch.arange(4)
        term._resample(ids)
        velocities, heights = term.vel_command_b.clone(), term.height_command_offset.clone()
        term._resample(ids)
        torch.testing.assert_close(term.vel_command_b, velocities)
        torch.testing.assert_close(term.height_command_offset[3], heights[3])

    def test_subset_sampling_does_not_touch_other_environments(self):
        term, _ = make_term()
        term.com_target_offset[:] = .007
        term._resample(torch.tensor([0]))
        torch.testing.assert_close(term.com_target_offset[1:], torch.full((3, 2), .007))

    def test_slew_is_vector_bounded_and_converges_without_overshoot(self):
        term, env = make_term()
        term.com_target_offset[:] = torch.tensor([.01, -.01])
        term._update_command()
        self.assertTrue(torch.all(torch.linalg.vector_norm(term.com_offset, dim=-1) <= .02 * env.step_dt + 1e-8))
        for _ in range(100):
            term._update_command()
        torch.testing.assert_close(term.com_offset[:2], torch.tensor([[.01, -.01], [.01, -.01]]))
        self.assertTrue(torch.all(term.com_offset[2:] == 0))

    def test_reward_mask_and_fresh_state(self):
        term, env = make_term()
        reward = REWARDS["track_com_xy_exp"]
        torch.testing.assert_close(reward(env), torch.tensor([1., 1., 0., 0.]))
        term.robot.data.body_com_pose_w.torch[:, 2, 0] += .03
        changed = reward(env)
        self.assertTrue(torch.all(changed[:2] < 1))
        self.assertTrue(torch.all(changed[2:] == 0))
        self.assertTrue(torch.isfinite(changed).all())

    def test_old_velocity_reward_ignores_appended_dimensions(self):
        term, env = make_term()
        before = REWARDS["track_velocity_exp"](env)
        term.com_offset[:] = 100
        torch.testing.assert_close(REWARDS["track_velocity_exp"](env), before)

    def test_manual_commands_clamp_and_preserve_update_hook(self):
        term, _ = make_term()
        raw = torch.tensor([[0., 0., 0., 0., .1, -.1], [-.1, 0., 0., 0., .005, .005],
                            [0., .2, 0., 0., .01, .01], [-.1, .2, 0., 0., .01, .01]])
        hook = term._update_command.__func__
        term.set_manual_command(raw)
        self.assertIs(term._update_command.__func__, hook)
        torch.testing.assert_close(term.mode, torch.arange(4))
        torch.testing.assert_close(term.com_target_offset[0], torch.tensor([.01, -.01]))
        self.assertTrue(torch.all(term.com_target_offset[2:] == 0))
        self.assertTrue(torch.all(term.com_offset == 0))
        term._update_command()
        before = term.com_offset.clone()
        term.set_manual_command(raw)
        torch.testing.assert_close(term.com_offset, before)
        term.reset(torch.tensor([0]))
        self.assertTrue(torch.all(term.com_offset[0] == 0))
        torch.testing.assert_close(term.com_offset[1:], before[1:])
        torch.testing.assert_close(term.com_target_offset[0], torch.tensor([.01, -.01]))

    def test_dynamic_to_static_has_no_old_effective_offset(self):
        term, _ = make_term()
        raw = torch.zeros(4, 6)
        raw[:, 4:] = .01
        term.set_manual_command(raw)
        for _ in range(100):
            term._update_command()
        raw[:, 1] = .2
        term.set_manual_command(raw)
        self.assertTrue(torch.all(term.com_offset == 0))
        raw[:] = 0
        term.set_manual_command(raw)
        term._update_command()
        self.assertTrue(torch.all(term.com_offset == 0))

    def test_metrics_exclude_walking_and_reset_frame(self):
        term, env = make_term()
        env.episode_length_buf[:] = torch.tensor([1, 0, 1, 1])
        term._update_metrics()
        torch.testing.assert_close(term._com_step_count, torch.tensor([1., 0., 0., 0.]))
        extras = term.reset(torch.arange(4))
        self.assertEqual(extras["error_com_xy"], 0.)
        self.assertEqual(extras["success_rate_com"], 1.)
        self.assertTrue(torch.all(term._com_step_count == 0))
        self.assertNotIn("error_com_xy", term.reset(torch.arange(4)))

    def test_invalid_parameters_fail_early(self):
        for options in ({"com_target_speed": 0}, {"com_target_speed": float("nan")},
                        {"com_offset_x": (.01, .02)}, {"com_offset_y": (-float("inf"), .01)},
                        {"com_zero_probability": 2}, {"com_success_threshold": -1},
                        {"element_names": ["h", "vx", "vy", "wz"]}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_term(**options)
        term, env = make_term()
        for std in (0., -1., float("inf"), float("nan")):
            with self.subTest(std=std), self.assertRaises(ValueError):
                REWARDS["track_com_xy_exp"](env, std=std)
        for raw in (torch.zeros(4, 4), torch.full((4, 6), float("nan"))):
            with self.assertRaises(ValueError):
                term.set_manual_command(raw)

    def test_com_debug_markers_start_hidden(self):
        class Marker:
            def __init__(self, cfg):
                self.visible = True

            def set_visibility(self, visible):
                self.visible = visible

        term, _ = make_term()
        for name in ("goal_height", "current_height", "goal_vel", "current_vel", "goal_com", "current_com"):
            setattr(term.cfg, f"{name}_visualizer_cfg", object())
        with patch.dict(NAMESPACE, {"VisualizationMarkers": Marker}):
            term._set_debug_vis_impl(True)
        self.assertFalse(term.goal_com_visualizer.visible)
        self.assertFalse(term.current_com_visualizer.visible)
        self.assertTrue(term.goal_height_visualizer.visible)
        term._set_debug_vis_impl(False)
        self.assertFalse(term.goal_height_visualizer.visible)

    def test_keyboard_com_steps_limits_and_reset(self):
        teleop = load_definitions(ROOT / "scripts/rsl_rl/play_teleop.py", {"_clamp", "KeyboardCommandTeleop"}, {
            "COM_OFFSET_STEP": 0.002,
        })
        keyboard_type = teleop["KeyboardCommandTeleop"]
        keyboard = keyboard_type.__new__(keyboard_type)
        keyboard.reset()
        keyboard.com_x_limits, keyboard.com_y_limits = (-.006, .006), (-.004, .004)
        keys = "W UP S DOWN A D Q LEFT E RIGHT Z X I K J L V SPACE H C".split()
        kb = SimpleNamespace(**{key: key for key in keys})
        keyboard._handle_key(kb.I, kb)
        self.assertEqual(keyboard.com_dx, .002)
        for _ in range(10):
            keyboard._handle_key(kb.I, kb)
            keyboard._handle_key(kb.J, kb)
        self.assertEqual(keyboard.as_tuple(), (0., 0., 0., 0., .006, .004))
        for _ in range(10):
            keyboard._handle_key(kb.K, kb)
            keyboard._handle_key(kb.L, kb)
        self.assertEqual(keyboard.as_tuple(), (0., 0., 0., 0., -.006, -.004))
        keyboard.h = -.1
        keyboard._handle_key(kb.V, kb)
        self.assertEqual(keyboard.as_tuple(), (-.1, 0., 0., 0., 0., 0.))
        keyboard._handle_key(kb.SPACE, kb)
        self.assertEqual(keyboard.as_tuple(), (0.,) * 6)

    def test_observation_is_appended_and_exports_are_present(self):
        cfg_path = MDP.parent / "g1_env_cfg.py"
        tree = ast.parse(cfg_path.read_text(encoding="utf-8"))
        policy = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "PolicyCfg")
        fields = [node.targets[0].id for node in policy.body if isinstance(node, ast.Assign)]
        self.assertEqual(fields[-2:], ["upper_body_target", "com_offset_cmd"])
        self.assertFalse(any("active" in name or "mode" in name for name in fields))
        stub = (MDP / "__init__.pyi").read_text(encoding="utf-8")
        for name in ("task_command_com", "track_com_xy_exp"):
            self.assertIn(f'"{name}"', stub)
            self.assertGreaterEqual(stub.count(name), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
