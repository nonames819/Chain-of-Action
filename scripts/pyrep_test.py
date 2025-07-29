from pyrep import PyRep
import os

SCENE_FILE = '/root/CoppeliaSim/scenes/workspace.ttt'  # 或其他 .ttt 场景路径

pr = PyRep()
pr.launch(SCENE_FILE, headless=True)  # 注意 headless=True
pr.start()

print('成功连接 CoppeliaSim！')

pr.stop()
pr.shutdown()