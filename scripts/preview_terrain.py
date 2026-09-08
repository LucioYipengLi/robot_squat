"""预览程序化生成的地形（金字塔阶梯 + 随机起伏）。

参考 IsaacLab 官方独立脚本示例：``IsaacLab/scripts/demos/procedural_terrain.py``
"""

import argparse
from isaaclab.app import AppLauncher

# 1. 初始化 Isaac Sim App 参数
parser = argparse.ArgumentParser(description="Preview Terrain in Isaac Lab")
AppLauncher.add_app_launcher_args(parser)
# 预览脚本默认应打开 Kit 可视化窗口；否则 AppLauncher 在未显式指定 --viz 时会强制进入 headless 模式
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

# 启动模拟器（开启 GUI）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg


def design_scene() -> TerrainImporter:
    """构建灯光与地形，返回 :class:`TerrainImporter` 实例。"""
    # 没有灯光场景会是全黑的，看不到地形
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # 2. 地形生成器配置（此处以金字塔阶梯地和随机起伏地为例）
    terrain_generator_cfg = TerrainGeneratorCfg(
        seed=42,
        size=(8.0, 8.0),
        border_width=5.0,
        num_rows=5,
        num_cols=5,
        sub_terrains={
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.5,
                # step_height_range 是 (最小台阶高度, 最大台阶高度) 的元组，不是标量 step_height
                step_height_range=(0.05, 0.15),
                step_width=0.3,
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.5, noise_range=(0.02, 0.1), noise_step=0.02
            ),
        },
    )

    # TerrainImporterCfg 与 TerrainGeneratorCfg 的字段是分开的：
    #   - num_rows / num_cols / size 属于 TerrainGeneratorCfg，不能写在 TerrainImporterCfg 上
    #   - 正确的字段名是 terrain_generator，不是 generator
    #   - prim_path 是必填字段（MISSING），必须显式给出
    terrain_importer_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        num_envs=1,
        terrain_type="generator",
        terrain_generator=terrain_generator_cfg,
        max_init_terrain_level=None,
        debug_vis=True,
    )

    # 3. 导入并生成地形
    # 注意：TerrainImporter.__init__ 内部会调用 sim_utils.SimulationContext.instance().device，
    # 因此必须在调用本函数之前先创建 SimulationContext（见 main()），否则会抛出 AttributeError。
    return TerrainImporter(terrain_importer_cfg)


def main():
    """主函数：先创建 SimulationContext，再构建地形，最后进入渲染循环。"""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # 设置一个能观察到整片地形（约 50m x 50m）的相机视角
    sim.set_camera_view(eye=[50.0, 50.0, 50.0], target=[0.0, 0.0, 0.0])

    # 构建场景（灯光 + 地形）
    design_scene()

    # 必须 reset 之后，场景才会真正开始渲染与物理步进
    sim.reset()
    print("[INFO]: 地形生成完毕，按下 Ctrl+C 或关闭窗口退出。")

    # 4. 保持窗口开启以供观察
    # 用 sim.step() 而不是 simulation_app.update()，这样物理与渲染才能正常推进
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
