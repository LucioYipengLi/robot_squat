[中文](README.md) | English

# g1

An installable downstream Isaac Lab task package with a standard uv `src` layout.

The registered tasks are:

- `Template-G1-v0`: Phase 1.
- `Template-G1-Phase2-v0`: Phase 2, adding static physics randomization.
- `Template-G1-Phase3-v0`: Phase 3, adding external pushes.
- `Template-G1-Upper-v0`: fixed-base waist/arm control with left-only, right-only and dual-arm modes, tracking world-fixed wrist goals observed in the live pelvis frame.
- `Template-G1-Upper-Mobile-v0`: free-base Upper training with a frozen Lower actor executing a fixed zero-velocity standing command. See [frozen Lower standing](docs/FROZEN_LOWER_STANDING.md).


## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment.
`uv sync` does not install Isaac Sim. This task defaults to `isaacsim_physx`; use `--extra isaacsim`
when running it, or explicitly select another installed backend:

```bash
uv sync
```

This project's `pyproject.toml` uses editable relative paths to an Isaac Lab source checkout. Update
`[tool.uv.sources]` if either directory moves.

The project forwards every optional extra declared by the Isaac Lab package used to create it, including physics,
rendering, visualizer, RL, teleoperation, and development features. Inspect `pyproject.toml` for the complete list and
pass each required extra to `uv run`:

```bash
# Standalone OV PhysX
uv run --extra ovphysx isaaclab random_agent --task <TASK_NAME> physics=ovphysx

# Isaac Sim with PhysX and RTX rendering
uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx

# Isaac Sim with the Rerun visualizer
uv run --extra isaacsim --extra rerun isaaclab random_agent --task <TASK_NAME> --viz rerun
```

The `all` extra forwards Isaac Lab's `all` selection. The `ov` extra installs both the `ovphysx` and `ovrtx` runtimes;
you can also select either independently. Commit `pyproject.toml` and `uv.lock` so collaborators use the same
environment.

## Run the tasks

### Shared training entrypoint

Use `python -m g1.rl train` for training/resumption, `play` for playback, `teleop` for manual control,
and `eval` for fixed-condition checkpoint comparisons.
`train/play` infers the task family and its Runner from the registered `--task`. Optional `--task-family`
checks that inferred ownership. The lower-limb Runner stores the model and live curriculum state in one checkpoint, so Phase 2/3 automatically
inherit the preceding phase's task distribution. It currently supports the Torch environment frontend and
single-process RSL-RL `OnPolicyRunner`; no additional RL library option is needed.

`upper_limb_agent` provides fixed-base G1 dual wrist pose tracking and its own Runner with standard training logs under
`logs/rsl_rl/g1_upper/`. It independently owns its robot configuration and MDP modules.
`teleop/eval` remain lower-limb only. See the
[task Runner guide](docs/TASK_RUNNERS.md) (Chinese) for selection and new-task integration.

All commands below use Isaac Sim PhysX. Replace each quoted `<...>` placeholder with an actual checkpoint path.
Iteration counts are examples; adjust them to your training plan.

| Argument | Meaning |
| --- | --- |
| `--task-family` | Optional ownership check: `lower_limb_agent` or `upper_limb_agent`; inferred from `--task` by default. |
| `--task` | Task ID for this invocation; lower-limb tasks also distinguish training phases. |
| `--checkpoint` | Model to load for resumption or playback; lower-limb phase transitions use the immediately preceding phase. No additional `--resume` flag is needed. |
| `--max_iterations` | Number of training iterations to execute in this invocation. After loading, this is the additional iteration count, not a cumulative limit. |
| `--seed` | Draw a random run seed by default. Pass a nonnegative integer, e.g. `--seed 42`, for a fixed seed, or `--seed -1` to explicitly draw one. |
| `--num_envs`, `--device` | Optional, for example `--num_envs 1024 --device cuda:0`. |

Each new project agent configuration draws its default seed from system randomness, including training,
resumption, playback and lower-limb teleoperation. Training logs record the selected integer in both `params/agent.yaml`
and `params/env.yaml`; pass that value to `--seed` to reuse it. An explicit `agent.seed=<integer>` also overrides
the random default, with `--seed` taking precedence. `--deterministic` is a separate determinism option and
does not fix the default random seed. Reusing a seed does not resume the original random-number sequence.

### G1 upper-body pose tracking

Headless training and visual playback:

```bash
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Upper-v0 --viz none --max_iterations 1500 physics=isaacsim_physx

uv run --extra isaacsim python -m g1.rl play --task Template-G1-Upper-v0 --checkpoint "<g1_upper_model.pt>" --num_envs 16 --real-time physics=isaacsim_physx
```

The default is 4096 environments; override it with `--num_envs`. Playback displays both current wrist frames and active targets.
Goals are sampled as offsets from the default wrist poses, with hold times based on goal-to-pelvis distance.
The policy now receives 93 observations, including wrist pose errors, and requires retraining. See the
[task structure and pose command guide](docs/TASK_RUNNERS.md#g1-上肢位姿跟踪结构) (Chinese).

Upper tasks also provide a 0 kg workspace Q prior, queried at pelvis-relative target positions after active wrists meet the position tolerance.
Its reward weight defaults to zero while diagnostics remain active. See the [workspace Q guide](docs/WORKSPACE_QUALITY.md) (Chinese) for configuration, metrics and archived resources.

`Template-G1-Upper-Mobile-v0` lets Upper choose base-frame `vx`, `vy`, yaw rate, and a target height offset for frozen Lower.
It uses 21 actions (3 velocities, 1 height offset, then 17 Upper joints) and 107 observations. Velocity bounds are
±0.3 m/s, ±0.1 m/s and ±0.3 rad/s; height offsets range from −0.25 to +0.02 m, with zero action preserving default height.
The shared `lower_command_rate=-0.10` retains H2's penalty on changes in all four normalized velocity/height commands;
holding any constant height has zero height-change cost. Magnitude and overflow costs still cover only velocities,
with the R9C weights retained. Mobile samples regular targets in 70% of episodes and adds a shared world-Z
shift of −0.15 to −0.05 m to active wrist targets in the other 30%, holding the full shifted poses until reset.
See the [height target guide](docs/UPPER_HEIGHT_TARGETS.md) (Chinese) for configuration, metrics and paired continuation runs.
`base_vertical_acc=-0.03` retains the H3B setting and adds a soft penalty above 4 m/s² for pelvis-link world-Z acceleration, covering every physics substep.
Set both `env.actions.lower_command.enabled=false` and `env.actions.height_offset.enabled=false` for a zero-command control.
Older 20-action/106-observation Mobile checkpoints require retraining or explicit migration.
See the [frozen Lower control guide](docs/FROZEN_LOWER_STANDING.md) (Chinese).

### Initial training: Phase 1

Ordinary WALK uses actual-liftoff joint references and support-duration shaping; squat-walk retains the phase clock.
See the [WALK reward guide](docs/WALK_REWARDS.md) (Chinese) for the local settings, metrics and validation.

```bash
uv run --extra isaacsim python -m g1.rl train --task Template-G1-v0 --max_iterations 10000 physics=isaacsim_physx
```

### Resume within a phase

For example, load a Phase 2 checkpoint and continue Phase 2 training for another 1000 iterations:

```bash
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Phase2-v0 --checkpoint "<phase2_model.pt>" --max_iterations 1000 physics=isaacsim_physx
```

For Phase 1/3, replace both the task ID and checkpoint with those of the corresponding phase.
Same-phase resumption restores curriculum progress, survival EMA and completed-episode summaries.

### Continue across phases: Phase 1 → 2 → 3

Select the target phase's task ID and explicitly provide a checkpoint from the immediately preceding phase.
Command ranges, mode weights and disturbance magnitudes are inherited automatically:

```bash
# Phase 1 → Phase 2: add static physics randomization
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Phase2-v0 --checkpoint "<phase1_model.pt>" --max_iterations 5000 physics=isaacsim_physx

# Phase 2 → Phase 3: add external pushes
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Phase3-v0 --checkpoint "<phase2_model.pt>" --max_iterations 3000 physics=isaacsim_physx
```

Phase transitions inherit the live task distribution and cumulative curriculum progress, while rebuilding
gate statistics under the new phase's conditions. Phase 2/3 require a checkpoint; phase skips and reverse
transitions are rejected.

Phase 3 starts with per-axis velocity increments of ±0.10 m/s and progressively expands to ±0.70 m/s
when both survival EMA and task success pass their gates. See the [push curriculum configuration](docs/CURRICULUM_CHECKPOINTS.md#phase-3-推扰课程与调参) (Chinese).

### Evaluation and playback

Use the task ID matching the checkpoint's phase, for example Phase 3:

```bash
uv run --extra isaacsim python -m g1.rl play --task Template-G1-Phase3-v0 --checkpoint "<phase3_model.pt>" physics=isaacsim_physx
```

Evaluation restores command ranges and mode weights. `play_mode()` disables curricula, observation noise,
physics randomization and disturbances. Both resumption and evaluation start fresh episodes rather than
continuing the saved physics trajectory.

Checkpoints are saved under `logs/rsl_rl/<experiment_name>/<run>/model_*.pt` by default.
Standard `uv run isaaclab train/play --task ...` uses the upstream Runner and does not save or restore project
curriculum state; use the project entrypoint throughout this workflow. Legacy checkpoints without curriculum
metadata require an explicit `--curriculum-snapshot "<curriculum.json>"`; see [curriculum checkpoint details](docs/CURRICULUM_CHECKPOINTS.md) (Chinese).

### Fixed-condition batch evaluation

Run compatible checkpoints against a versioned suite with fixed episode quotas and replayable task inputs.
The initial modules cover four-mode tracking and upper-body disturbance sweeps; evaluation always runs headless.

```bash
uv run --extra isaacsim python -m g1.rl eval --suite g1_robustness --checkpoints "early=<early_model.pt>" "late=<late_model.pt>" --policy-output both --num_envs 128
```

The terminal shows overall/episode progress, execution stages and estimated remaining time; use `--progress plain`
or `--progress off` to change the display. Results include per-episode records, summary tables and TensorBoard checkpoint comparisons. The built-in suites
use nominal resets and disable observation noise. See the [evaluation guide](docs/EVALUATION.md) (Chinese)
for configuration, metrics, resume commands and module extension.

### Policy teleoperation and visualization

Load a checkpoint from the same phase and control height, velocity and pelvis offsets with a keyboard or
Xbox-layout gamepad. The defaults are one environment, real-time stepping and a visible Kit window:

```bash
uv run --extra isaacsim python -m g1.rl teleop --task Template-G1-v0 --checkpoint "<phase1_model.pt>" --viz kit physics=isaacsim_physx
```

W/S and A/D control translation, Q/E yaw, Z/X height, and I/K and J/L pelvis offsets. SPACE clears all input,
C toggles pelvis/foot geometry, and H prints help. Checkpoint curriculum ranges are restored; falls reset
the robot automatically and timeout termination is disabled. The original `scripts/play_teleop.py` remains
a thin launcher for the same workflow. See the [teleoperation guide](docs/TELEOP.md) (Chinese) for options,
input mappings and visualization semantics.

### Environment tools

The following upstream tools remain available; replace task placeholders with a registered task:

```bash
# List this project's environments and their available presets
uv run isaaclab list_envs --show_presets

# Exercise an environment without a trained policy
uv run --extra isaacsim isaaclab zero_agent --task "<TASK_NAME>" --num_envs 16 physics=isaacsim_physx
uv run --extra isaacsim isaaclab random_agent --task "<TASK_NAME>" --num_envs 16 physics=isaacsim_physx
```

### Performance benchmarks

The following commands use Phase 1 with 16 environments as a small baseline, Isaac Sim PhysX and no visualizer.
Once these run successfully, increase `--num_envs` to 1024 or 4096 to measure throughput at the intended scale.
Keep environment count, device, physics backend and visualization settings identical when comparing runs.

| Type | Measurements |
| --- | --- |
| `startup` | Python imports, configuration resolution, simulation launch, environment creation and the first step. |
| `runtime` | Environment-step timing and throughput with random actions; no policy training. |
| `training` | Rollout collection, policy updates and overall throughput during actual training. |
| `play` | Policy inference and environment-step performance; no model updates. |

```bash
# 1. Startup: no RL library or step-count arguments needed
uv run --extra isaacsim isaaclab benchmark startup --task Template-G1-v0 --num_envs 16 --output_path outputs/benchmarks/startup physics=isaacsim_physx

# 2. Runtime: warm up for 50 steps, then measure 1000 steps
uv run --extra isaacsim isaaclab benchmark runtime --task Template-G1-v0 --num_envs 16 --num_steps 1000 --warmup_steps 50 --output_path outputs/benchmarks/runtime physics=isaacsim_physx

# 3. Training: limit to 10 iterations and exclude the first environment step's cold start
uv run --extra isaacsim isaaclab benchmark training --rl_library rsl_rl --task Template-G1-v0 --num_envs 16 --max_iterations 10 --warmup_steps 1 --output_path outputs/benchmarks/training physics=isaacsim_physx

# 4. Playback: supply a Phase 1 checkpoint and measure 1000 steps after warmup
uv run --extra isaacsim isaaclab benchmark play --rl_library rsl_rl --task Template-G1-v0 --checkpoint "<phase1_model.pt>" --num_envs 16 --num_steps 1000 --warmup_steps 50 --output_path outputs/benchmarks/play physics=isaacsim_physx
```

The `training/play` benchmarks require an explicit `--rl_library`; `startup/runtime` do not.
`--output_path` selects the performance-results directory, using the `schema` formatter by default.
Training logs and models are still written under `logs/rsl_rl/`.

These benchmarks use upstream Runners and do not save or restore project curriculum state.
The `training` benchmark updates the model; explicitly setting `--max_iterations 10` avoids running the
project default of 10000 iterations. Use the project entrypoint for regular three-phase resumption.
For `play`, match the task's phase to the checkpoint to measure runtime performance; use the earlier
`python -m g1.rl play` command to evaluate the checkpoint's restored task distribution.

Use `physics=<PRESET>` to select one of the presets shown by `isaaclab list_envs`.

## Project structure

Code reviews, resolved findings and curriculum checkpoint documentation are collected in the [docs index](docs/README.md).

Task families live in `src/g1/tasks`. Within a manager-based family, task-wide MDP terms live in `mdp`, while
robot-specific scenes, registrations, and agent configurations live in `config/g1`. Add another robot by
creating a sibling of `config/g1`; add another task by creating a sibling task-family directory.

## Project assets

Keep project-owned USD files and related data under `src/g1/assets/data`. The asset module exposes a
stable path that works from an editable checkout and an installed wheel:

```python
from g1.assets import G1_ASSETS_DIR

robot_usd_path = G1_ASSETS_DIR / "robots" / "my_robot.usd"
```

Keep asset configuration in the `g1.assets` package and pass `str(robot_usd_path)` to configuration fields that
expect a string. USD files are configured for Git LFS by the project's `.gitattributes`; install Git LFS before adding
large binary assets.

## Development

Run the registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and writes a git-ignored `pyrightconfig.json`. The resulting
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` extra, include it while generating the editor configuration so the command can discover the
Isaac Sim installation:

```bash
uv run --extra isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.

Upper-body disturbances use progressive references by default. See [upper-body motion](docs/UPPER_BODY_MOTION.md) for configuration, exposure metrics, and checkpoint compatibility.
