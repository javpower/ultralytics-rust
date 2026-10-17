# 用官方 ultralytics（Miniconda env）生成 yolo26-seg E2E 损失参照。
# yolo26n-seg.pt 完整权重（nc=80）+ 固定输入/标签/实例掩码；
# 输出 weights/yolo26n_seg_loss_ref.json（o2m/o2o 分支 items + total）。
import json

import torch

code = r'''
import torch
from ultralytics import YOLO
from types import SimpleNamespace

torch.manual_seed(2026)
B, S = 2, 320
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
# 每图 id 掩码 (B, S, S)（overlap=True 官方语义：masks[i] == mask_idx+1）
masks = torch.zeros(B, S, S)
_cnt = {}
for i in range(5):
    bi = int(batch_idx[i])
    cx, cy, w, h = bboxes[i].tolist()
    _cnt[bi] = _cnt.get(bi, 0) + 1
    x1 = int((cx - w / 2) * S); y1 = int((cy - h / 2) * S)
    x2 = max(x1 + 1, int((cx + w / 2) * S)); y2 = max(y1 + 1, int((cy + h / 2) * S))
    masks[bi, y1:y2, x1:x2] = float(_cnt[bi])

sem_masks = torch.randint(0, 80, (B, S, S))
batch = {
    "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes, "masks": masks, "img": img,
    "sem_masks": sem_masks,
}

m = YOLO("E:/xunlian/yolo26n-seg.pt")
net = m.model
if isinstance(net.args, dict):
    net.args = SimpleNamespace(**net.args)
net.args.box, net.args.cls, net.args.dfl = 7.5, 0.5, 1.5
net.args.overlap_mask = True
net.train()
crit = net.init_criterion()
import ultralytics.utils.loss as loss_mod
_csl = loss_mod.v8SegmentationLoss.calculate_segmentation_loss
def csl_patched(self, *a, **k):
    out = _csl(self, *a, **k)
    print("csl return:", float(out) if torch.is_tensor(out) and out.numel() == 1 else out)
    return out
loss_mod.v8SegmentationLoss.calculate_segmentation_loss = csl_patched.__get__(crit.one2many, type(crit.one2many))
import ultralytics.utils.loss as loss_mod
_sm_stats = {}
_orig_sml = loss_mod.v8SegmentationLoss.single_mask_loss
def sml_patched(gt_mask, pred, proto, xyxy, area):
    import torch.nn.functional as F
    with torch.no_grad():
        pm = torch.einsum("in,nhw->ihw", pred, proto)
        _sm_stats["calls"] = _sm_stats.get("calls", 0) + 1
        _sm_stats["n_last"] = int(gt_mask.shape[0])
        _sm_stats["n_total"] = _sm_stats.get("n_total", 0) + int(gt_mask.shape[0])
        s = float((F.binary_cross_entropy_with_logits(pm, gt_mask, reduction="none").mean(dim=(1, 2)) / area).sum())
        _sm_stats["sum_l1w"] = _sm_stats.get("sum_l1w", 0.0) + s
        _sm_stats["per_anchor_last"] = s / max(int(gt_mask.shape[0]), 1)
        mb = float(F.binary_cross_entropy_with_logits(pm, gt_mask, reduction="none").mean(dim=(1, 2)).mean())
        am = float(area.mean())
        print(f"[official-img] n={gt_mask.shape[0]} meanBCE_crop={mb:.5f} area_mean={am:.5f} per_sum={s:.5f}")
    return _orig_sml(gt_mask, pred, proto, xyxy, area)
loss_mod.v8SegmentationLoss.single_mask_loss = staticmethod(sml_patched)
print("seg criterion:", type(crit).__name__, "o2m gain:", crit.o2m, "o2o gain:", crit.o2o)

preds = net(batch["img"])
if isinstance(preds, tuple):
    preds = preds[1]
lm = crit.one2many.loss(preds["one2many"], batch)
lo = crit.one2one.loss(preds["one2one"], batch)

def flat_items(it):
    out = {}
    for k, v in it.items():
        if torch.is_tensor(v):
            if v.numel() == 1:
                out[k] = float(v)
            else:
                out[k] = v.detach().reshape(-1).tolist()
    return out

print("o2m sml accumulated:", _sm_stats)
print("o2m items:", flat_items(lm[1]))
print("o2o items:", flat_items(lo[1]))
total = lm[0] * crit.o2m + lo[0] * crit.o2o
print("total(weighted) shape:", tuple(total.shape), "sum x bs:", float(total.sum()) * B)

out = {
    "batch_size": B, "imgsz": S,
    "img": img.reshape(-1).tolist(),
    "batch_idx": batch_idx.reshape(-1).tolist(),
    "cls": cls.reshape(-1).tolist(),
    "bboxes": bboxes.reshape(-1).tolist(),
    "masks": masks.reshape(-1).tolist(),
    "sem_masks": sem_masks.reshape(-1).tolist(),
    "o2m": flat_items(lm[1]),
    "o2o": flat_items(lo[1]),
    "total_sum": float(total.sum()) * B,
}
with open("E:/ultralytics-rust/weights/yolo26n_seg_loss_ref.json", "w") as f:
    json.dump(out, f)
print("ref written")
'''

exec(code)
