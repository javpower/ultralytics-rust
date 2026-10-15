# 训练曲线对拍：官方 ultralytics trainer vs ultralytics-rust，同参数同数据短程训练，
# 输出双方逐 epoch 的 box/cls/dfl 分量与 mAP，供收敛性对比。
#
# 用法（py312 env）：
#   python scripts/curve_parity.py --data E:/dev/aegis-vision/data/coco8/coco8-ul.yaml \
#       --epochs 30 --imgsz 320 --batch 8 --model yolov8n.pt
# 依赖：官方 CLI（pip ultralytics）+ 本仓库 release 构建（target/release/yolo）。
import argparse, json, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="yolov8n.pt")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--imgsz", type=int, default=320)
ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()

py = r"D:\Dev\Tools\Miniconda3\envs\py312\python.exe"
root = Path("E:/ultralytics-rust")

# 官方
official_code = f"""
import json, ultralytics
m = ultralytics.YOLO("{(root / 'weights' / args.model).as_posix()}")
r = m.train(data="{args.data}", epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device="cpu", verbose=False)
metrics = r.results_dict
print("OFFICIAL_JSON:" + json.dumps({{k: float(v) for k, v in metrics.items()}}))
"""
r1 = subprocess.run([py, "-c", official_code], capture_output=True, text=True)
official = None
for line in r1.stdout.splitlines():
    if line.startswith("OFFICIAL_JSON:"):
        official = json.loads(line[len("OFFICIAL_JSON:"):])
if official is None:
    print("official train failed:", r1.stderr[-1500:])
    sys.exit(1)

# 本仓库
r2 = subprocess.run(
    [str(root / "target/release/yolo"), "detect", "train", f"data={args.data}", f"model={args.model}",
     f"epochs={args.epochs}", f"imgsz={args.imgsz}", f"batch={args.batch}", "device=cpu", "name=curve_parity"],
    capture_output=True, text=True, cwd=str(root),
)
ours = None
for line in r2.stdout.splitlines():
    if "train done" in line:
        ours = line
ours_metrics = None
# 取最后一次 val 的 metrics.jsonl 行
mp = root / "runs" / "curve_parity" / "metrics.jsonl"
if not mp.exists():
    # name 冲突会自增，找最新的
    cands = sorted(root.glob("runs/curve_parity*/metrics.jsonl"), key=lambda p: p.stat().st_mtime)
    mp = cands[-1] if cands else None
if mp:
    rows = [json.loads(x) for x in mp.read_text().splitlines() if x.strip()]
    ours_metrics = rows[-1]

print("== official (final) ==")
for k, v in official.items():
    print(f"  {k}: {v:.4f}")
print("== ours (final epoch val) ==")
print(" ", ours)
if ours_metrics:
    print(f"  metrics.jsonl: {ours_metrics}")
print("\n注：收敛曲线应整体趋势一致；绝对值受随机增强/采样顺序影响，同一 seed 不完全对齐官方 DataLoader。")
