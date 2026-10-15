# 用官方 ultralytics 8.4.157（py312 env）生成 v8DetectionLoss 三分量参照。
# 固定输入/标签写入 weights/yolov8n_loss_ref.json，供 ul-engine/tests/loss_parity.rs 对拍。
import json, os, subprocess, sys

code = r'''
import torch, json
from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

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
}

m = YOLO("E:/ultralytics-rust/weights/yolov8n.pt")
net = m.model
from types import SimpleNamespace
if isinstance(net.args, dict):
    net.args = SimpleNamespace(**net.args)
net.args.box, net.args.cls, net.args.dfl = 7.5, 0.5, 1.5
crit = v8DetectionLoss(net)
net.train()
loss, items = net.loss({**batch, "img": img})

# 分配器内部量（供 Rust 侧对拍）
with torch.no_grad():
    preds = net(img)
    anchor_points, stride_tensor = make_anchors(preds["feats"], crit.stride, 0.5)
    pred_bboxes = crit.bbox_decode(anchor_points, preds["boxes"].permute(0, 2, 1))
    pd_pix = pred_bboxes.detach() * stride_tensor
    # 与 get_assigned_targets_and_loss 相同的分配调用
    targets_all = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    pre = crit.preprocess(targets_all, 2, scale_tensor=torch.tensor([320., 320., 320., 320.]))
    gt_labels, gt_bboxes = pre.split((1, 4), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
    _, tb, ts, fg, tgi = crit.assigner(
        preds["scores"].detach().permute(0, 2, 1).sigmoid(),
        pd_pix.type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels, gt_bboxes, mask_gt,
    )
    print("OFFICIAL fg:", int(fg.sum()), "ts_sum:", float(ts.sum()), "ts_max:", float(ts.max()))
    # 分配器中间量（对拍 Rust align/overlaps）
    mask_in_gts = crit.assigner.select_candidates_in_gts(anchor_points * stride_tensor, gt_bboxes, mask_gt)
    am, ov = crit.assigner.get_box_metrics(
        preds["scores"].detach().permute(0, 2, 1).sigmoid(),
        pd_pix.type(gt_bboxes.dtype),
        gt_labels, gt_bboxes, mask_in_gts * mask_gt,
    )
    print("OFFICIAL in_gts:", int(mask_in_gts.sum()), "align_nz:", int((am > 0).sum()),
          "align_max: %.6f" % float(am.max()), "ov_max: %.6f" % float(ov.max()),
          "ov_nz:", int((ov > 0).sum()))
    # 逐锚解码框对拍（b0 的若干锚）
    picks = [0, 1, 2, 100, 1000, 2000]
    for k in picks:
        r = pd_pix[0, k]
        print("OFFICIAL pd[%d] = [%.3f, %.3f, %.3f, %.3f]" % (k, r[0], r[1], r[2], r[3]))
    print("OFFICIAL anc[0..3]:", anchor_points[:4].tolist(), "stride0:", float(stride_tensor[0]))

out = {
    "img": img.flatten().tolist(),
    "batch_idx": batch["batch_idx"].tolist(),
    "cls": batch["cls"].flatten().tolist(),
    "bboxes": batch["bboxes"].flatten().tolist(),
    "batch_size": B,
    "imgsz": S,
    "loss": {k: float(v) for k, v in items.items()},
    "total": float(loss.sum()),
    "fg_count": int(fg.sum()),
    "ts_sum": float(ts.sum()),
}
with open("E:/ultralytics-rust/weights/yolov8n_loss_ref.json", "w") as f:
    json.dump(out, f)
print("saved:", out["loss"], "total", out["total"])
'''

env = os.environ.copy()
r = subprocess.run([r"D:\Dev\Tools\Miniconda3\envs\py312\python.exe", "-c", code],
                   capture_output=True, text=True, env=env)
print(r.stdout.strip())
if r.returncode != 0:
    print("STDERR:", r.stderr[-3000:])
    sys.exit(1)
