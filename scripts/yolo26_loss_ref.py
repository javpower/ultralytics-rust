# 用官方 ultralytics（Miniconda py3.13 env）生成 yolo26 E2E (reg_max=1) 损失参照。
# 固定输入/标签 + yolo26n.pt 完整权重（nc=80，避免双方 nc 改写差异），
# 输出 weights/yolo26n_loss_ref.json，供 ul-engine/tests/loss_parity_y26.rs 对拍。
import json

import torch

code = r'''
import torch, json
from ultralytics import YOLO
from types import SimpleNamespace

torch.manual_seed(2026)
B, S = 2, 320
img = torch.rand(B, 3, S, S)

batch = {
    "batch_idx": torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]),
    "cls": torch.tensor([[2.0], [7.0], [24.0], [1.0], [55.0]]),
    "bboxes": torch.tensor([
        [0.50, 0.50, 0.30, 0.40],
        [0.20, 0.70, 0.16, 0.12],
        [0.05, 0.10, 0.06, 0.05],
        [0.60, 0.30, 0.25, 0.25],
        [0.30, 0.60, 0.10, 0.08],
    ]),
    "img": img,
}

m = YOLO("E:/ultralytics-rust/weights/yolo26n.pt")
net = m.model
if isinstance(net.args, dict):
    net.args = SimpleNamespace(**net.args)
net.args.box, net.args.cls, net.args.dfl = 7.5, 0.5, 1.5
net.train()
loss, items = net.loss(batch)

print("type(loss):", type(loss))
print("items repr:", items if not torch.is_tensor(items) else "tensor")

# 兼容 dict / list 两种 items 形态
def as_float(x):
    if torch.is_tensor(x):
        return float(x)
    return float(x)

if isinstance(items, dict):
    vals = [as_float(v) for v in items.values()]
    names = list(items.keys())
else:
    vals = []
    names = []
    for x in items:
        if torch.is_tensor(x):
            vals.append(as_float(x))
        elif isinstance(x, str):
            names.append(x)
print("names:", names, "vals:", vals)

# ---- 分支级对拍诊断：one2many / one2one 各自的损失与分配器内部量 ----
crit = net.init_criterion()
print("criterion type:", type(crit).__name__)
preds = net(batch["img"])
if isinstance(preds, tuple):
    preds = preds[1]
print("preds keys:", list(preds.keys()) if isinstance(preds, dict) else type(preds))

sub = {}
if hasattr(crit, "one2many"):
    sub["one2many"] = crit.one2many
    sub["one2one"] = crit.one2one
elif hasattr(crit, "one2many_loss"):
    sub["one2many"] = crit.one2many_loss
    sub["one2one"] = crit.one2one_loss

import ultralytics.utils.loss as loss_mod

bl_stats = {}
_orig_bl_fwd = loss_mod.BboxLoss.forward
def bl_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride):
    import torch.nn.functional as F
    with torch.no_grad():
        rows = fg_mask.nonzero(as_tuple=True)
        pd = pred_dist[rows]
        tl = torch.cat((anchor_points - target_bboxes[..., :2], target_bboxes[..., 2:] - anchor_points), -1)
        tgt = tl[rows]
        pdx = pd * stride[rows[1]]
        tgx = tgt * stride[rows[1]]
        bl_stats["n"] = pd.shape[0]
        bl_stats["pred_abs_norm"] = float(pd.abs().mean())
        bl_stats["tgt_abs_norm"] = float(tgt.abs().mean())
        bl_stats["delta_norm"] = float((pdx / imgsz[1] - tgx / imgsz[0]).abs().mean())
        bl_stats["pred_px"] = float(pdx.abs().mean())
        bl_stats["tgt_px"] = float(tgx.abs().mean())
        bl_stats["imgsz"] = imgsz.tolist()
        bl_stats["tss"] = float(target_scores_sum)
        bl_stats["weight_mean"] = float(target_scores[rows].sum(-1).mean())
    out = _orig_bl_fwd(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride)
    bl_stats["l1_out"] = float(out[1])
    bl_stats["box_out"] = float(out[0])
    return out
loss_mod.BboxLoss.forward = bl_forward

for name, sub_crit in sub.items():
    l, it = sub_crit(preds[name], batch)
    v = [float(x) for x in it.values()] if isinstance(it, dict) else [float(x) for x in it if torch.is_tensor(x)]
    print(f"[{name}] items:", v)
    # monkeypatch assigner.assign 捕获真实内部量
    cap = {}
    orig = sub_crit.assigner.forward
    def patched(self, pd_scores, pd_bboxes, anc, gt_l, gt_b, mask_gt, _orig=orig, _cap=cap):
        cap_shapes = (tuple(pd_scores.shape), tuple(pd_bboxes.shape), tuple(anc.shape))
        tl, tb, ts, fg, tgi = _orig(pd_scores, pd_bboxes, anc, gt_l, gt_b, mask_gt)
        _cap["shapes"] = cap_shapes
        _cap["fg"] = int(fg.sum())
        _cap["tss"] = float(ts.sum())
        _cap["ts_max"] = float(ts.max())
        return tl, tb, ts, fg, tgi
    sub_crit.assigner.forward = patched.__get__(sub_crit.assigner)
    l2, it2 = sub_crit(preds[name], batch)
    sub_crit.assigner.forward = orig
    v2 = [float(x) for x in it2.values()] if isinstance(it2, dict) else [float(x) for x in it2 if torch.is_tensor(x)]
    print(f"[{name}] items(recheck):", v2, "assigner:", sub_crit.assigner.topk, sub_crit.assigner.topk2)
    print(f"[{name}] captured: shapes={cap.get('shapes')} fg={cap.get('fg')} tss={cap.get('tss'):.4f} ts_max={cap.get('ts_max'):.4f}")

# loss.sum() 原始值（判 items 是否 ×bs）
print("bl_stats:", bl_stats)
print("loss.sum():", float(loss.sum()))

# loss.sum() 原始值（判 items 是否 ×bs）
print("loss.sum():", float(loss.sum()))

# loss.sum() 原始值（判 items 是否 ×bs）
print("loss.sum():", float(loss.sum()))


out = {
    "batch_size": B,
    "imgsz": S,
    "img": img.reshape(-1).tolist(),
    "batch_idx": batch["batch_idx"].reshape(-1).tolist(),
    "cls": batch["cls"].reshape(-1).tolist(),
    "bboxes": batch["bboxes"].reshape(-1).tolist(),
    "loss": {
        "box_loss": vals[0] if len(vals) > 0 else 0.0,
        "cls_loss": vals[1] if len(vals) > 1 else 0.0,
        "l1_loss": vals[2] if len(vals) > 2 else 0.0,
    },
    "total_sum": float(loss.sum()),
}
with open("E:/ultralytics-rust/weights/yolo26n_loss_ref.json", "w") as f:
    json.dump(out, f)
print("ref written:", {k: out["loss"][k] for k in out["loss"]})
'''

exec(code)
