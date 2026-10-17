# 用官方 ultralytics（Miniconda env）生成 yolo26-pose / yolo26-obb E2E 损失参照。
# 随机初始化模型整包 torch.save（port 的 pickle 解析器可直接加载），
# 输出 weights/yolo26n_pose_loss_ref.json 与 yolo26n_obb_loss_ref.json。
import json

import torch

SEED = 2026
B, S = 2, 320

code = r'''
import torch, json
from ultralytics import YOLO
from types import SimpleNamespace

def dump_model(net, path):
    torch.save({"model": net, "train_args": {}}, path)

def run_pose():
    torch.manual_seed(SEED)
    m = YOLO("yolo26n-pose.yaml")
    net = m.model
    if isinstance(net.args, dict):
        net.args = SimpleNamespace(**net.args)
    net.args.box, net.args.cls, net.args.dfl = 7.5, 0.5, 1.5
    net.args.pose, net.args.kobj = 12.0, 1.0
    dump_model(net, "E:/ultralytics-rust/weights/y26pose_rand.pt")

    torch.manual_seed(SEED + 1)
    img = torch.rand(B, 3, S, S)
    batch_idx = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
    cls = torch.tensor([[2.0], [7.0], [24.0], [1.0], [55.0]])
    bboxes = torch.tensor([
        [0.50, 0.50, 0.30, 0.40],
        [0.20, 0.70, 0.16, 0.12],
        [0.05, 0.10, 0.06, 0.05],
        [0.60, 0.30, 0.25, 0.25],
        [0.30, 0.60, 0.10, 0.08],
    ])
    kpts = torch.rand(5, 17, 3)
    kpts[:, :, 2] = (kpts[:, :, 2] > 0.5).float()
    batch = {"batch_idx": batch_idx, "cls": cls, "bboxes": bboxes, "keypoints": kpts, "img": img}

    net.train()
    crit = net.init_criterion()
    print("pose criterion:", type(crit).__name__)
    preds = net(batch["img"])
    if isinstance(preds, tuple):
        preds = preds[1]
    print("pose preds keys:", list(preds["one2many"].keys()) if isinstance(preds["one2many"], dict) else type(preds["one2many"]))
    # 官方 v8PoseLoss.loss 需要 preds 里有 "keypoints"（解码后 kpts）与 "kpts"（原始）
    for br in ("one2many", "one2one"):
        if isinstance(preds[br], dict) and "keypoints" not in preds[br]:
            preds[br]["keypoints"] = preds[br].get("kpts")
    lm = crit.one2many.loss(preds["one2many"], batch)
    lo = crit.one2one.loss(preds["one2one"], batch)

    def flat(it):
        out = {}
        for k, v in it.items():
            out[k] = float(v) if torch.is_tensor(v) and v.numel() == 1 else v.detach().reshape(-1).tolist()
        return out

    print("pose o2m:", flat(lm[1]))
    print("pose o2o:", flat(lo[1]))
    total = lm[0] * crit.o2m + lo[0] * crit.o2o
    print("pose total x bs:", float(total.sum()) * B)
    out = {
        "batch_size": B, "imgsz": S,
        "img": img.reshape(-1).tolist(),
        "batch_idx": batch_idx.reshape(-1).tolist(),
        "cls": cls.reshape(-1).tolist(),
        "bboxes": bboxes.reshape(-1).tolist(),
        "kpts": kpts.reshape(-1).tolist(),
        "o2m": flat(lm[1]), "o2o": flat(lo[1]),
        "total_sum": float(total.sum()) * B,
    }
    with open("E:/ultralytics-rust/weights/yolo26n_pose_loss_ref.json", "w") as f:
        json.dump(out, f)

def run_obb():
    torch.manual_seed(SEED + 100)
    m = YOLO("yolo26n-obb.yaml")
    net = m.model
    if isinstance(net.args, dict):
        net.args = SimpleNamespace(**net.args)
    net.args.box, net.args.cls, net.args.dfl = 7.5, 0.5, 1.5
    dump_model(net, "E:/ultralytics-rust/weights/y26obb_rand.pt")

    torch.manual_seed(SEED + 101)
    img = torch.rand(B, 3, S, S)
    batch_idx = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
    cls = torch.tensor([[2.0], [7.0], [24.0], [1.0], [55.0]])
    bboxes = torch.tensor([
        [0.50, 0.50, 0.30, 0.40, 0.10],
        [0.20, 0.70, 0.16, 0.12, -0.30],
        [0.05, 0.10, 0.06, 0.05, 0.55],
        [0.60, 0.30, 0.25, 0.25, -0.10],
        [0.30, 0.60, 0.10, 0.08, 0.70],
    ])  # xywh + r（弧度，原始值）
    batch = {"batch_idx": batch_idx, "cls": cls, "bboxes": bboxes, "img": img}

    net.train()
    crit = net.init_criterion()
    print("obb criterion:", type(crit).__name__)
    preds = net(batch["img"])
    if isinstance(preds, tuple):
        preds = preds[1]
    lm = crit.one2many.loss(preds["one2many"], batch)
    lo = crit.one2one.loss(preds["one2one"], batch)

    print("obb o2m:", flat := {k: float(v) if torch.is_tensor(v) and v.numel() == 1 else "vec" for k, v in lm[1].items()})
    print("obb o2o:", {k: float(v) if torch.is_tensor(v) and v.numel() == 1 else "vec" for k, v in lo[1].items()})
    total = lm[0] * crit.o2m + lo[0] * crit.o2o
    print("obb total x bs:", float(total.sum()) * B)
    out = {
        "batch_size": B, "imgsz": S,
        "img": img.reshape(-1).tolist(),
        "batch_idx": batch_idx.reshape(-1).tolist(),
        "cls": cls.reshape(-1).tolist(),
        "bboxes": bboxes.reshape(-1).tolist(),
        "o2m": {k: (float(v) if torch.is_tensor(v) and v.numel() == 1 else v.detach().reshape(-1).tolist()) for k, v in lm[1].items()},
        "o2o": {k: (float(v) if torch.is_tensor(v) and v.numel() == 1 else v.detach().reshape(-1).tolist()) for k, v in lo[1].items()},
        "total_sum": float(total.sum()) * B,
    }
    with open("E:/ultralytics-rust/weights/yolo26n_obb_loss_ref.json", "w") as f:
        json.dump(out, f)

run_pose()
run_obb()
'''

exec(code)
