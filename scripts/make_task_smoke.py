# 从 coco8 的 detect 标签合成 seg/pose/obb 三套冒烟数据集（images 拷贝，labels 重写）。
# 仅用于训练管线冒烟：seg 多边形=框四角；pose=框内 17 个合成关键点；obb=轴对齐九点。
import shutil
from pathlib import Path

SRC = Path("E:/dev/aegis-vision/data/coco8")
DST = Path("E:/ultralytics-rust/data")

NAMES = "\n".join(f"  {i}: c{i}" for i in range(80))


def conv_det_to_seg(line):
    v = line.split()
    cls, cx, cy, w, h = v[0], *map(float, v[1:5])
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return cls + " " + " ".join(f"{p:.6f}" for xy in pts for p in xy)


def conv_det_to_pose(line):
    v = line.split()
    cls, cx, cy, w, h = v[0], *map(float, v[1:5])
    kpts = []
    for i in range(17):
        kx = cx - w / 2 + w * (i % 4) / 3
        ky = cy - h / 2 + h * (i // 4) / 4
        kpts += [f"{kx:.6f}", f"{ky:.6f}", "1"]
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} " + " ".join(kpts)


def conv_det_to_obb(line):
    v = line.split()
    cls, cx, cy, w, h = v[0], *map(float, v[1:5])
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return cls + " " + " ".join(f"{p:.6f}" for xy in pts for p in xy)


for task, conv, model in [("seg", conv_det_to_seg, "yolov8n-seg"), ("pose", conv_det_to_pose, "yolov8n-pose"), ("obb", conv_det_to_obb, "yolov8n-obb")]:
    d = DST / f"coco8-{task}"
    for split in ["train", "val"]:
        (d / "images" / split).mkdir(parents=True, exist_ok=True)
        (d / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in (SRC / "images" / split).glob("*"):
            shutil.copy(img, d / "images" / split / img.name)
            lf = SRC / "labels" / split / (img.stem + ".txt")
            out = []
            if lf.exists():
                for line in lf.read_text().splitlines():
                    if line.strip() and len(line.split()) >= 5:
                        out.append(conv(line))
            (d / "labels" / split / (img.stem + ".txt")).write_text("\n".join(out))
    yaml = f"path: {d.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{NAMES}\n"
    if task == "pose":
        yaml = f"path: {d.as_posix()}\ntrain: images/train\nval: images/val\nkpt_shape: [17, 3]\nnames:\n{NAMES}\n"
    (DST / f"coco8-{task}-ul.yaml").write_text(yaml)
    print(f"{task}: {d} yaml=DST/coco8-{task}-ul.yaml model={model}")
