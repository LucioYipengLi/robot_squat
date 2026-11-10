"""从已分析的批量指标文件提取信息：给下游程序阅读、复制和改造的中文示例。

在 analysis/ 目录中运行（默认只读，不启动 Viser，不运行 IK/FK/动力学）：
    uv run python scripts/extract_metrics_example.py
    uv run python scripts/extract_metrics_example.py --scenario maximum --feasible-only
    uv run python scripts/extract_metrics_example.py --grid-index 42 --solution-index 0
    uv run python scripts/extract_metrics_example.py --output results/left_0.4ef_best_q.npz
也可在第一个位置显式指定其他批量指标 NPZ。--help 查看全部参数。

一、这些数值针对什么任务？
    G1 骨盆坐标系下，腰 3 关节＋单侧臂 7 关节，末端为 wrist_yaw_link。
    关节角单位 rad，位置单位 m，力矩单位 N·m；非分析侧等关节固定在参考零位。
    本项目默认仅约束腕部位置、朝向自由。具体任务和容差以源文件元数据为准。
    这是已保存静态构型的分析，不代表路径连续性、动态平衡或速度/加速度可行性。

二、四项基础指标的计算过程与意义（这里读取结果，不重新求解）
    1. 最小奇异值 sigma_min：
       对腕部位置 p(q) 关于 10 个活动关节求导，得到平移雅可比 Jp，形状 (3,10)。
       SVD 得到 sigma_1 >= sigma_2 >= sigma_3 >= 0，sigma_min = sigma_3。
       单位 m/rad；越大表示最弱平移方向的速度传递能力越强。腰部贡献已包含其中。
    2. 条件数 condition_number：
       kappa = sigma_1 / sigma_3，越小越均衡，理想值为 1。
       数值判秩不足 3 时记为 inf；阈值为 max(atol, rtol*sigma_1)，参数存于 signature。
       做均值或相关性分析时要显式处理 inf，不能把它误换成 0。
    3. 关节限位裕度 joint_limit_margin（构型合理性 J）：
       m_i = clip(2*min(q_i-lower_i, upper_i-q_i)/(upper_i-lower_i), 0, 1)。
       单关节位于区间中心时为 1、到边界时为 0；区间中心不一定是零角。
       整体 J = min_i(m_i)，表示最差关节的裕度，不是十个关节的平均值。
       joint_margins 保存逐关节值，joint_limit_bottleneck 指出瓶颈关节。
    4. 力矩裕度 torque_margin（默认作为动力学性能 D）：
       tau(m) = tau_gravity + m*tau_per_kg；先按符号叠加，再取绝对值。
       tau_per_kg = -Jp.T @ gravity_vector，负载是腕部原点的理想点质量，无质心偏置。
       tau_gravity 由固定骨盆的全身质量模型计算：保留全身重力贡献，去掉橡胶手，
       传感器 link 无质量；最终提取腰＋所选单臂关节的力矩。
       tau_allowed_i = effort_scale * URDF_effort_i，是假设的允许力矩，不是厂家确认额定值。
       rho = max_i(abs(tau_i)/tau_allowed_i)，torque_margin = 1-rho。
       越大越好；负值表示至少一个关节超限。0/典型/最大负载三列的实际质量从元数据读取。
       文件名中的 0.4ef 只是命名习惯，必须读取 signature.torque_settings.effort_scale 核实。

三、综合性能 Q
    默认 sigma_ref 为全文件最大 sigma_min；s_sigma_min = clip(sigma_min/sigma_ref, 0, 1)。
    全文件 sigma_min 均为零时 s_sigma_min=0；s_k = 1/kappa（1/inf=0）。
    K = sqrt(s_sigma_min*s_k)，J = joint_limit_margin，D = torque_margin。
    Q = 0.3*K + 0.3*J + 0.4*D；默认保留负 D 扣分，故 Q 可为负。
    动力学权重 0.4 和力矩降额系数 0.4 是两个独立参数！
    实际权重、sigma 参考值、是否截断 D 必须读取 composite_metrics，不要硬编码默认值。
    固定参考值或截断模式下，已存的 sigma_min_score / dynamics_performance 已体现这些设置。
    不同文件直接比较 Q，需统一归一化基准、权重、质量和力矩假设。
    正 Q 不保证力矩未超限；若需要负载可行构型，应先用 torque_overloaded 筛选。
    旧版指标文件没有 Q 时，BatchMetricsResult.load() 只在内存中按默认规则补算，不改文件。

四、最容易混淆的数组与索引
    N = points 的行数：原 NPZ 保留的目标点，其中有些点没有保存有效构型。
    M = joint_configs 的行数：全部已保存有效构型，已去掉未使用的填充槽位。
    sample_id：本指标文件中 [0,M) 的构型行号。一个点可以对应多个 sample_id。
    point_indices[sample_id]：该构型对应 points 中的行号，也就是 --point-index。
    source_grid_indices[point_index]：原始网格索引，也就是 Viser 显示的原网格点索引。
    solution_indices[sample_id]：原 IK 文件中该点的构型槽位，也就是 --solution-index。
    上述索引全部从 0 开始；Viser 中供人切换的连续序号通常从 1 开始，不能直接混用。
    joint_configs[sample_id] 的列顺序严格对应 metadata['joint_names']，不能猜测 URDF 顺序。
    points[point_index] 是目标网格位置；wrist_poses[sample_id] 是实际 FK 位姿，
    格式为 [qw,qx,qy,qz, px,py,pz]。二者位置允许存在原 IK 容差内的误差。

五、下游使用建议
    保留 sample_id、原网格索引、构型槽位、关节名称、参数和源文件指纹，才能追溯结果。
    每点的最佳 Q 必须先逐构型计算，再在同一点内选最大值；不能拼接不同构型的最好指标。
    示例默认所有有效 IK 构型参与统计；--feasible-only 先排除当前负载超限构型，再选最佳 Q。
    若另加 J 下限等约束，也应先在全部构型中筛选，再分组取最优；先取最优后删点可能漏解。
    没有保存满足条件的构型，不等于证明该空间点绝对不可达。
    --output 导出的是每点一个所选负载下的最优样本，不是完整构型集；切换负载重新选最优时，
    应回到原文件。导出中其他负载的指标也对应这同一个被选中的构型。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from g1_analysis.batch_metrics import BatchMetricsResult, aggregate_metric
from g1_analysis.configuration_metrics import file_record, output_paths
from g1_analysis.paths import RESULTS_DIR


DEFAULT_INPUT = RESULTS_DIR / "left_5cm.batch.metrics_0.4ef.npz"
SCENARIOS = {"unloaded": 0, "typical": 1, "maximum": 2}


def point_sample_ids(result, point_index):
    """示例：用前缀和定位某点的全部有效构型，避免对 M 行反复做全表扫描。"""
    if not 0 <= point_index < len(result["points"]):
        raise ValueError("--point-index 超出目标点数组范围")
    start, stop = result.offsets[point_index:point_index + 2]
    return np.arange(start, stop, dtype=np.int64)


def sample_record(result, sample_id, scenario):
    """把一个有效构型整理为小型字典；下游可直接复用此函数。"""
    r, sid = result, int(sample_id)
    if not 0 <= sid < len(r["joint_configs"]):
        raise ValueError("sample_id 超出有效构型范围")
    row = int(r["point_indices"][sid])
    names = r.metadata["joint_names"]
    mass = r.metadata["payload_masses_kg"][scenario]
    # 下游可无需 MuJoCo，直接重建这个已保存构型在所选负载下的逐关节保持力矩。
    torque = r["gravity_torques_nm"][sid] + mass * r["payload_holding_torque_per_kg_nm"][sid]
    condition = float(r["condition_number"][sid])
    return {
        "sample_id": sid, "point_index": row, "source_grid_index": int(r["source_grid_indices"][row]),
        "solution_index": int(r["solution_indices"][sid]), "target_point_m": r["points"][row].tolist(),
        "actual_wrist_pose_wxyz_xyz": r["wrist_poses"][sid].tolist(),
        "position_error_m": float(np.linalg.norm(r["wrist_poses"][sid, 4:] - r["points"][row])),
        "joint_config_rad_by_name": dict(zip(names, r["joint_configs"][sid].astype(float).tolist())),
        "sigma_min_m_per_rad": float(r["sigma_min"][sid]),
        "condition_number": condition if np.isfinite(condition) else None,  # 标准 JSON 用 null 表示 inf。
        "is_singular": bool(r["is_singular"][sid]), "joint_limit_margin": float(r["joint_limit_margin"][sid]),
        "joint_limit_bottleneck": names[int(r["joint_limit_bottleneck"][sid])],
        "payload_mass_kg": float(mass), "torque_margin": float(r["torque_margin"][sid, scenario]),
        "torque_overloaded": bool(r["torque_overloaded"][sid, scenario]),
        "holding_torque_nm_by_name": dict(zip(names, torque.astype(float).tolist())),
        "torque_bottleneck": names[int(r["torque_bottleneck"][sid, scenario])],
        "K": float(r["kinematic_performance"][sid]), "J": float(r["joint_limit_margin"][sid]),
        "D": float(r["dynamics_performance"][sid, scenario]), "Q": float(r["composite_score"][sid, scenario]),
    }


def best_q_per_point(result, scenario, *, feasible_only=False):
    """返回有效点行号、对应最佳真实构型行号和 Q；不会把无数据的 -1 当数组索引。"""
    scores, best_ids, counts = aggregate_metric(result, "composite_score", scenario=scenario,
                                               mode="best", feasible_only=feasible_only)
    rows = np.flatnonzero(counts > 0)
    return rows, best_ids[rows], scores[rows]


def export_best(result, source, output, rows, sample_ids, scenario, feasible_only, selected_record, *, overwrite=False):
    """另存紧凑 NPZ 和小型 JSON 摘要。

    导出数组的第 i 行已相互对齐；point_indices[i] 仍是源文件的点行号，
    不要用它再次索引导出文件自己的 points；sample_ids 同样是源文件的构型行号。
    """
    meta = result.metadata
    protected = [Path(source["path"]).with_suffix(".json")]
    records = [meta.get("source_npz"), meta.get("urdf"), *meta.get("source_ik_signature", {}).get("sources", {}).values()]
    for record in records:
        if record and record.get("path"):
            path = Path(record["path"])
            protected.extend((path, path.with_suffix(".metadata.json")))
    # 复用项目输出保护：符号链接/硬链接也不能绕过源文件保护；已有导出需显式 --overwrite。
    output, summary = output_paths(Path(source["path"]), Path(output), protected=protected, overwrite=overwrite)
    mass = meta["payload_masses_kg"][scenario]
    payload = {"point_indices": rows, "sample_ids": sample_ids,
        "source_grid_indices": result["source_grid_indices"][rows], "points": result["points"][rows],
        "joint_names": np.asarray(meta["joint_names"]),
        "holding_torques_nm": result["gravity_torques_nm"][sample_ids] + mass * result["payload_holding_torque_per_kg_nm"][sample_ids]}
    for name in ("solution_indices", "joint_configs", "wrist_poses", "sigma_min", "condition_number", "is_singular",
                 "joint_margins", "joint_limit_margin", "joint_limit_bottleneck", "torque_margin", "torque_overloaded",
                 "torque_utilization", "torque_bottleneck", "sigma_min_score", "condition_score", "kinematic_performance",
                 "dynamics_performance", "composite_score"):
        payload[name] = result[name][sample_ids]
    document = {"schema_version": 1, "kind": "g1_example_best_q_extraction",
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_metrics": source,
        "source_metrics_schema_version": meta["schema_version"], "target_link": meta["target_link"],
        "coordinate_frame": meta["coordinate_frame"], "joint_names": meta["joint_names"],
        "units": {"joint_configs": "rad", "points": "m", "holding_torques_nm": "N*m", "sigma_min": "m/rad"},
        "wrist_pose_order": "qw,qx,qy,qz,px,py,pz", "payload_masses_kg": meta["payload_masses_kg"],
        "torque_settings": meta["signature"]["torque_settings"], "composite_metrics": meta["composite_metrics"],
        "selection": {"metric": "composite_score", "mode": "best", "scenario_index": scenario,
                      "payload_mass_kg": mass, "feasible_only": feasible_only, "exported_points": len(rows)},
        "index_meaning": {"sample_ids": "row in SOURCE metrics, not row in this export",
                          "point_indices": "row in SOURCE points", "source_grid_indices": "original grid index",
                          "solution_indices": "slot in original IK point; all indices zero-based"},
        "array_shapes": {key: list(value.shape) for key, value in payload.items()},
        "selected_sample_example": selected_record,
        "note": "one saved best-Q configuration per point for the selected load; other-load columns refer to the SAME chosen configuration"}
    encoded = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    payload["metadata_json"] = np.array(encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, binary in ((output, True), (summary, False)):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                if binary:
                    np.savez_compressed(stream, **payload)
                else:
                    stream.write(encoded.encode("utf-8"))
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return output, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("result", type=Path, nargs="?", default=DEFAULT_INPUT, help="默认 results/left_5cm.batch.metrics_0.4ef.npz")
    parser.add_argument("--scenario", choices=SCENARIOS, default="typical", help="默认典型负载；实际质量从文件读取")
    parser.add_argument("--feasible-only", action="store_true", help="先排除所选负载下超限构型，再选每点最佳 Q")
    point = parser.add_mutually_exclusive_group()
    point.add_argument("--point-index", type=int, help="查看原 points 行号，从 0 开始")
    point.add_argument("--grid-index", type=int, help="查看 Viser 的原网格点索引，从 0 开始")
    parser.add_argument("--solution-index", type=int, help="查看该点原构型槽位；默认选该点最佳 Q")
    parser.add_argument("--limit", type=int, default=5, help="最多打印几个点内构型及全局高分样本，默认 5")
    parser.add_argument("--output", type=Path, help="可选：另存每点最佳 Q 的 .npz 和摘要 .json")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖独立导出，始终保护源文件")
    args = parser.parse_args(argv)
    try:
        if args.limit < 1:
            raise ValueError("--limit 必须为正数")
        source = file_record(args.result)
        result = BatchMetricsResult.load(args.result)  # 验证类型、数组、映射；v1 仅在内存补齐 Q。
        r, meta, scenario = result, result.metadata, SCENARIOS[args.scenario]
        rows, best_ids, best_q = best_q_per_point(r, scenario, feasible_only=args.feasible_only)
        print(f"源文件: {source['path']}\n格式 v{meta['schema_version']} | 目标 {meta['target_link']} | 坐标系 {meta['coordinate_frame']}")
        print(f"N={len(r['points']):,} 点；有解点={np.count_nonzero(r['solution_counts']):,}；M={len(r['joint_configs']):,} 有效构型")
        print(f"力矩缩放={meta['signature']['torque_settings']['effort_scale']:g} × URDF effort；负载 kg={meta['payload_masses_kg']}")
        quality = meta["composite_metrics"]
        print(f"Q 权重 K/J/D={quality['weights']}；sigma_ref={quality['sigma_reference_m_per_rad']:.8g} m/rad；D 策略={quality['torque_policy']}")
        print(f"当前场景={args.scenario} ({meta['payload_masses_kg'][scenario]:g} kg)；feasible_only={args.feasible_only}；可选点={len(rows):,}")

        # 全部有效构型的四指标列，可直接交给相关性、分组统计或其他后处理程序。
        all_four_metrics = np.column_stack((r["sigma_min"], r["condition_number"], r["joint_limit_margin"], r["torque_margin"][:, scenario]))
        print(f"全部四指标矩阵 shape={all_four_metrics.shape}；含非有限值的行数={np.count_nonzero(~np.isfinite(all_four_metrics).all(axis=1))}")
        # 在下游增加其他约束时，先组合此类布尔掩码，再在每个点内部挑选构型。
        example_mask = (~r["torque_overloaded"][:, scenario]) & (r["joint_limit_margin"] >= 0.1)
        print(f"自定义筛选示意：力矩未超限且 J>=0.1 的构型={np.count_nonzero(example_mask):,}（仅示意，不改变本次选择）")

        if args.grid_index is not None:
            found = np.flatnonzero(r["source_grid_indices"] == args.grid_index)
            if not len(found):
                raise ValueError("原网格点未保存在文件中，可能被粗网格筛选排除")
            row = int(found[0])
        elif args.point_index is not None:
            row = args.point_index
        else:
            row = int(rows[0]) if len(rows) else int(np.flatnonzero(r["solution_counts"])[0])
        ids = point_sample_ids(r, row)
        chosen = None
        if args.solution_index is not None:
            found = ids[r["solution_indices"][ids] == args.solution_index]
            if not len(found):
                raise ValueError("该点没有所选有效构型槽位")
            chosen = int(found[0])
            if args.feasible_only and r["torque_overloaded"][chosen, scenario]:
                raise ValueError("所选构型在当前负载下超限，不满足 --feasible-only")
        else:
            eligible = ids[~r["torque_overloaded"][ids, scenario]] if args.feasible_only else ids
            if len(eligible):
                chosen = int(eligible[np.argmax(r["composite_score"][eligible, scenario])])

        print(f"\n查看点：point_index={row}，原网格索引={int(r['source_grid_indices'][row])}，目标 xyz(m)={r['points'][row].tolist()}")
        print("sample_id | 原构型槽位 | sigma_min | kappa | J | torque_margin | Q | 超限")
        preview = ids[:args.limit]
        if chosen is not None and chosen not in preview:
            preview = np.r_[preview, chosen]
        for sid in preview:
            print(f"{'*' if sid == chosen else ' '} {sid:7d} | {r['solution_indices'][sid]:4d} | {r['sigma_min'][sid]:.6g} | "
                  f"{r['condition_number'][sid]:.6g} | {r['joint_limit_margin'][sid]:.5f} | "
                  f"{r['torque_margin'][sid, scenario]:+.5f} | {r['composite_score'][sid, scenario]:+.5f} | {bool(r['torque_overloaded'][sid, scenario])}")
        record = sample_record(r, chosen, scenario) if chosen is not None else None
        if record is None:
            print("该点没有符合当前条件的已保存构型；这不构成不可达证明。")
        else:
            print(f"选中 sample_id={chosen}；K/J/D={record['K']:.5f}/{record['J']:.5f}/{record['D']:+.5f}；Q={record['Q']:+.5f}")
            print(f"限位瓶颈={record['joint_limit_bottleneck']}；力矩瓶颈={record['torque_bottleneck']}；位置误差={record['position_error_m']:.3g} m")
            print("关节名称 | 角度 rad | 角度 deg | 保持力矩 N*m")
            for name, angle in record["joint_config_rad_by_name"].items():
                print(f"{name} | {angle:+.6f} | {np.rad2deg(angle):+.3f} | {record['holding_torque_nm_by_name'][name]:+.5f}")

        print("\n每点最佳 Q 的高分样本（每点一个真实构型）：")
        for index in np.argsort(-best_q, kind="stable")[:args.limit]:
            print(f"原网格={int(r['source_grid_indices'][rows[index]])}，sample_id={best_ids[index]}，Q={best_q[index]:+.6f}")
        if args.output is not None:
            output, summary = export_best(r, source, args.output, rows, best_ids, scenario, args.feasible_only, record,
                                          overwrite=args.overwrite)
            print(f"\n已另存 {len(rows):,} 个点的最佳 Q 构型：\n{output}\n{summary}")
        else:
            print("\n只读示例运行结束；指定 --output 才会另存导出文件。")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

# 下游重新读取导出的文件（注意它是提取子集，不再是完整的 BatchMetricsResult 格式）：
# with np.load("results/left_0.4ef_best_q.npz", allow_pickle=False) as data:
#     meta = json.loads(str(data["metadata_json"]))
#     scenario = meta["selection"]["scenario_index"]
#     points = data["points"]                 # (P,3)，导出的 P 个目标点
#     q_rad = data["joint_configs"]           # (P,10)，按 data["joint_names"] 列顺序
#     quality = data["composite_score"][:, scenario]
#     source_sample_ids = data["sample_ids"]  # 回原指标文件追溯所选构型
#     torque_nm = data["holding_torques_nm"]  # (P,10)，本次选定负载的保持力矩
