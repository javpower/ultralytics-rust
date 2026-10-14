# 用官方 ultralytics（py312 conda env）生成 yolo11n 的参考输出
import json, subprocess, sys, os
code = '''
import torch, json
from ultralytics import YOLO
m = YOLO("E:/ultralytics-rust/weights/yolo11n.pt")
m.model.eval()
with torch.no_grad():
    y = m.model(torch.zeros(1, 3, 640, 640))
if isinstance(y, (tuple, list)):
    y = y[0]
if isinstance(y, dict):
    y = y.get("one2many", y)
print(type(y))
if hasattr(y, "shape"):
    vals = y.flatten()[:4096].tolist()
    print(json.dumps({"shape": list(y.shape), "values": vals}))
else:
    raise SystemExit(f"unexpected output type {type(y)}")
'''
env = os.environ.copy()
r = subprocess.run([r"D:\Dev\Tools\Miniconda3\envs\py312\python.exe", "-c", code],
                   capture_output=True, text=True, env=env)
for line in r.stdout.splitlines():
    if line.startswith("{"):
        data = json.loads(line)
        with open("E:/ultralytics-rust/weights/yolo11n_ref.json", "w") as f:
            json.dump(data, f)
        print("saved ref:", data["shape"], len(data["values"]), "values")
        break
else:
    print("STDOUT:", r.stdout[-2000:])
    print("STDERR:", r.stderr[-2000:])
    sys.exit(1)
