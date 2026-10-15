# 用官方 ultralytics 8.4.157 生成 seg/pose/obb 三任务损失分量参照。
# 每任务一个 JSON：输入张量 + 官方 loss 分量（items 语义，未乘 batch_size），
# 供 ul-engine/tests/loss_parity_task.rs 对拍。
import json, os, subprocess, sys

CODE = r"""
import torch, json
import torch.nn.functional as F
from types import SimpleNamespace
from ultralytics import YOLO
from ultralytics.utils.loss import v8SegmentationLoss, v8PoseLoss, v8OBBLoss

torch.manual_seed(2027)
B, S = 2, 320
img = torch.rand(B, 3, S, S)

def mk_batch(task):
    n0, n1 = 3, 2
    # yolov8n-pose nc=1（person）、yolov8n-obb nc=15（DOTA）、seg nc=80
    clsv = [0.0]*5 if task == "pose" else ([1.0, 2.0, 7.0, 0.0, 3.0] if task == "obb" else [2.0, 7.0, 24.0, 1.0, 55.0])
    batch = {
        "batch_idx": torch.tensor([0.0]*n0 + [1.0]*n1),
        "cls": torch.tensor([[c] for c in clsv]),
        "bboxes": torch.tensor([
            [0.50,0.50,0.30,0.40],[0.20,0.70,0.16,0.12],[0.05,0.10,0.06,0.05],
            [0.60,0.30,0.25,0.25],[0.30,0.60,0.10,0.08],
        ]),
    }
    if task == "seg":
        # overlap 模式：masks (B,S,S)，值=实例序号（1..n）
        masks = torch.zeros(B, S, S)
        for b, (n, off) in enumerate([(n0, 0), (n1, n0)]):
            for i in range(n):
                cx, cy, w, h = batch["bboxes"][off+i]
                x0 = int((cx-w/2)*S); x1 = int((cx+w/2)*S)
                y0 = int((cy-h/2)*S); y1 = int((cy+h/2)*S)
                masks[b, max(0,y0):max(0,y1), max(0,x0):max(0,x1)] = float(off+i+1)
        batch["masks"] = masks
    if task == "pose":
        n = n0 + n1
        kpts = torch.zeros(n, 17, 3)
        for i in range(n):
            cx, cy, w, h = batch["bboxes"][i]
            for j in range(17):
                kpts[i,j,0] = (cx - w/2 + w*(j % 4)/3).clamp(0,1)
                kpts[i,j,1] = (cy - h/2 + h*(j // 4)/4).clamp(0,1)
                kpts[i,j,2] = 1.0
        batch["keypoints"] = kpts
    if task == "obb":
        # bboxes (N,5) xywhr：用 bbox 加 0.3rad
        r = batch["bboxes"].clone()
        batch["bboxes"] = torch.cat([r, torch.full((r.shape[0],1), 0.3)], 1)
    return batch

def dump(task, model_pt, loss_cls, gains, extra_names):
    m = YOLO(f"E:/ultralytics-rust/weights/{model_pt}")
    net = m.model
    if isinstance(net.args, dict):
        net.args = SimpleNamespace(**net.args)
    defaults = {"overlap_mask": True, "pose": 12.0, "kobj": 1.0, "angle": 1.0}
    for k, v in {**defaults, **gains}.items():
        if not hasattr(net.args, k):
            setattr(net.args, k, v)
    for k, v in gains.items():
        setattr(net.args, k, v)
    net.args.box, net.args.cls, net.args.dfl = 7.5, 0.5, 1.5
    crit = loss_cls(net)
    batch = mk_batch(task)
    net.train()
    preds = net(img)
    if isinstance(preds, tuple) and len(preds) == 2 and isinstance(preds[1], dict):
        preds = preds[1]
    loss, items = crit(preds, {**batch, "img": img})
    # 中间量：fg_sum / ts_sum（分配一致性对拍）
    mid = {}
    with torch.no_grad():
        crit2 = loss_cls(net)
        _ = crit2
        try:
            from ultralytics.utils.tal import make_anchors
            ap_, st_ = make_anchors(preds["feats"], crit.stride, 0.5)
            if task == "obb":
                crit.bbox_decode.__self__  # noop
                pd = crit.bbox_decode(ap_, preds["boxes"].permute(0,2,1), preds["angle"].permute(0,2,1))
                tgt = torch.cat((batch["batch_idx"].view(-1,1), batch["cls"].view(-1,1), batch["bboxes"].view(-1,5)), 1)
                rw, rh = tgt[:,4]*320.0, tgt[:,5]*320.0
                tgt = tgt[(rw>=2)&(rh>=2)]
                pre = crit.preprocess(tgt, B, scale_tensor=torch.tensor([320.,320.,320.,320.]))
                gl, gb = pre.split((1,5), 2)
                mg = gb.sum(2, keepdim=True).gt_(0.0)
                assigner = crit.assigner
                pd_in = pd.detach().clone()
                pd_in[..., :4] *= st_
                _, tb, ts, fg, _ = assigner(preds["scores"].detach().permute(0,2,1).sigmoid(), pd_in.type(gb.dtype), ap_*st_, gl, gb, mg)
            else:
                pd = crit.bbox_decode(ap_, preds["boxes"].permute(0,2,1))
                tgt = torch.cat((batch["batch_idx"].view(-1,1), batch["cls"].view(-1,1), batch["bboxes"]), 1)
                pre = crit.preprocess(tgt, B, scale_tensor=torch.tensor([320.,320.,320.,320.]))
                gl, gb = pre.split((1,4), 2)
                mg = gb.sum(2, keepdim=True).gt_(0.0)
                _, tb, ts, fg, _ = crit.assigner(preds["scores"].detach().permute(0,2,1).sigmoid(), (pd.detach()*st_).type(gb.dtype), ap_*st_, gl, gb, mg)
            try:
                if task == "obb":
                    mig = assigner.select_candidates_in_gts(ap_ * st_, gb, mg)
                else:
                    mig = assigner.select_candidates_in_gts(ap_ * st_, gb, mg)
                per_img = mig.sum(-1).flatten().tolist()  # (b, n)
                mid = {"fg_sum": int(fg.sum()), "ts_sum": float(ts.sum()), "in_gts": int(mig.sum()), "in_gts_img": [int(x) for x in per_img]}
            except Exception as e:
                mid = {"fg_sum": int(fg.sum()), "ts_sum": float(ts.sum()), "in_gts_err": str(e)[:80]}
        except Exception as e:
            mid = {"mid_err": str(e)}
    out = {
        "task": task,
        "img": img.flatten().tolist(),
        "batch_idx": batch["batch_idx"].tolist(),
        "cls": batch["cls"].flatten().tolist(),
        "bboxes": batch["bboxes"].flatten().tolist(),
        "batch_size": B,
        "imgsz": S,
        "loss": {k: float(v) for k, v in items.items()},
        "total": float(loss.sum()),
        **mid,
    }
    if task == "seg":
        out["masks"] = batch["masks"].to(torch.uint8).flatten().tolist()
        out["mask_shape"] = [B, S, S]
    if task == "pose":
        out["keypoints"] = batch["keypoints"].flatten().tolist()
        out["kpt_shape"] = [17, 3]
    with open(f"E:/ultralytics-rust/weights/loss_ref_{task}.json", "w") as f:
        json.dump(out, f)
    print(task, "items:", {k: round(float(v), 4) for k, v in items.items()}, "total:", round(float(loss.sum()), 4))

dump("seg", "yolov8n-seg.pt", v8SegmentationLoss, {"box": 7.5, "cls": 0.5, "dfl": 1.5, "overlap": True}, None)
dump("pose", "yolov8n-pose.pt", v8PoseLoss, {"box": 7.5, "cls": 0.5, "dfl": 1.5, "pose": 12.0, "kobj": 1.0}, None)
dump("obb", "yolov8n-obb.pt", v8OBBLoss, {"box": 7.5, "cls": 0.5, "dfl": 1.5, "angle": 1.0}, None)
"""

env = os.environ.copy()
r = subprocess.run([r"D:\Dev\Tools\Miniconda3\envs\py312\python.exe", "-c", CODE],
                   capture_output=True, text=True, env=env)
print(r.stdout.strip())
if r.returncode != 0:
    print("STDERR:", r.stderr[-3000:])
    sys.exit(1)
