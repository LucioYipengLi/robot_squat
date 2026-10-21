"""Standalone CUDA reference-update cost; this does not measure full training FPS."""

import json
import time
from pathlib import Path

import torch

from g1.tasks.lower_limb_agent.trajectory import QuinticTrajectory


def main():
    device = "cuda:0"
    curve = QuinticTrajectory(torch.zeros(4096, 17, device=device))
    # Keep every slot moving throughout all timed windows. This is a synthetic
    # tensor workload, not a joint target sent to a robot or simulator.
    curve.set_goal(slice(None), torch.full_like(curve.position, 100.0))
    rows = []
    with torch.inference_mode():
        for _ in range(100):
            curve.advance(1 / 120)
        for _ in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(1000):
                curve.advance(1 / 120)
            torch.cuda.synchronize()
            rows.append((time.perf_counter() - start) * 1000 / 1000)
    result = {
        "device": torch.cuda.get_device_name(),
        "num_envs": 4096,
        "num_joints": 17,
        "wall_ms_per_advance": rows,
        "persistent_state_mib": sum(
            value.numel() * value.element_size() for value in vars(curve).values() if isinstance(value, torch.Tensor)
        )
        / 1024**2,
        "scope": "trajectory math only; excludes simulator target writes, physics and PPO",
        "workload": "all 4096 slots remain active throughout timing",
    }
    path = Path("outputs/verification/motion_benchmark.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
