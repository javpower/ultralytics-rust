//! 损失：v8DetectionLoss（TAL + BCE + 成对 CIoU + DFL 量化 CE）、E2E 双头、分类 CE。
//! 对齐 ultralytics/utils/loss.py（v8DetectionLoss/BboxLoss/DFLoss）。

use tch::{Kind, Tensor};

use ul_nn::head::HeadOutput;
use ul_nn::model::YoloModel;
use ul_nn::ops::{bbox_iou, dist2bbox, make_anchors};
use ul_nn::tal::TaskAlignedAssigner;

/// 训练批（检测/姿态/旋转框共用）：标签 (m, 6+) = [img_idx, cls, cx, cy, w, h, kpts...] 归一化。
pub struct DetBatch {
    pub img: Tensor,    // (B, 3, S, S) 0..1 RGB
    pub labels: Tensor, // (m, >=6)
    pub batch_size: i64,
    pub imgsz: i64,
}

/// 损失权重。
#[derive(Clone, Copy)]
pub struct LossW {
    pub box_gain: f64,
    pub cls: f64,
    pub dfl: f64,
}

impl Default for LossW {
    fn default() -> Self {
        LossW { box_gain: 7.5, cls: 0.5, dfl: 1.5 }
    }
}

/// 成对 CIoU（tal.py bbox_iou 语义：(N,4)×(N,4) → (N,1)）。
pub fn iou_pairwise(pred: &Tensor, target: &Tensor) -> Tensor {
    bbox_iou_pairwise(pred, target)
}

/// 检测损失入口。返回标量总损失。
pub fn v8_detection_loss(
    model: &YoloModel,
    w: LossW,
    assigner: &TaskAlignedAssigner,
    out: &HeadOutput,
    batch: &DetBatch,
    e2e: bool,
) -> Tensor {
    let device = batch.img.device();
    let bs = batch.batch_size as f64;
    let (o2m, o2o) = match out {
        HeadOutput::TrainDual(a, b) => (a.as_ref(), Some(b.as_ref())),
        HeadOutput::Train(a) => (a.as_ref(), None),
        _ => panic!("loss requires training head output"),
    };
    let feats = &o2m.feats;

    let (points, strides) = make_anchors(feats, &model.stride, 0.5);
    // points (2, A)，strides (1, A)
    let stride_per_anchor = strides.transpose(0, 1); // (A, 1)

    let mut loss = detect_branch_loss(model, assigner, o2m, batch, &points, &stride_per_anchor, w, false);
    if e2e {
        if let Some(o2o) = o2o {
            let l = detect_branch_loss(model, assigner, o2o, batch, &points, &stride_per_anchor, w, false);
            loss = loss + l;
        }
    }
    let _ = device;
    loss * bs
}

/// 单分支检测损失（TAL + cls BCE + CIoU + DFL）。
pub fn detect_branch_loss(
    model: &YoloModel,
    assigner: &TaskAlignedAssigner,
    preds: &ul_nn::head::RawPreds,
    batch: &DetBatch,
    points: &Tensor,           // (2, A)
    stride_per_anchor: &Tensor, // (A, 1)
    w: LossW,
    one2one: bool,
) -> Tensor {
    let device = batch.img.device();
    let feats = &preds.feats;

    // 解码预测
    let scores = preds.scores.sigmoid(); // (B, nc, A)
    let scores_t = scores.transpose(1, 2); // (B, A, nc)
    let raw_boxes = dfl_apply(model, &preds.boxes); // (B, 4, A) 积分后
    let (points2, strides2) = make_anchors(feats, &model.stride, 0.5);
    let _ = strides2;
    let pd_bboxes = dist2bbox(&raw_boxes, &points2, true, 1); // xywh 像素 (B, 4, A)

    // 目标
    let (gt_labels, gt_bboxes, mask_gt) = prepare_targets(batch);

    // 官方语义：锚点以像素坐标参与分配（anchors × strides）
    let anc_pixels = &(&points2 * &strides2); // (2, A)
    let out = assigner.assign(&scores_t, &pd_bboxes.transpose(1, 2), anc_pixels, &gt_labels, &gt_bboxes, &mask_gt);
    let AssignRef { target_bboxes, target_scores, fg_mask } = AssignRef {
        target_bboxes: out.target_bboxes,
        target_scores: out.target_scores,
        fg_mask: out.fg_mask,
    };
    let target_scores_sum = target_scores.sum_dim_intlist(None::<&[i64]>, false, Kind::Float).clamp_min(1.0);

    // cls BCE（BCE-with-logits 需要原始 logits，preds.scores 未过 sigmoid）
    let logits_t = preds.scores.transpose(1, 2).contiguous();
    let cls_loss = bce_sum(&logits_t, &target_scores) / target_scores_sum.shallow_clone() / w.cls;

    let fg_sum = fg_mask.sum_dim_intlist(None::<&[i64]>, false, Kind::Float);
    let has_fg = Vec::<i64>::try_from(&fg_sum).map(|v| v.first().copied().unwrap_or(0)).unwrap_or(0) > 0;
    if !has_fg {
        // 无前景：仅 cls；dfl/box 记零（保持梯度路径有内容：乘 0）
        let zero = scores.sum_dim_intlist(None::<&[i64]>, false, Kind::Float) * 0.0;
        if std::env::var("UL_LOSS_DEBUG").is_ok() {
            eprintln!("[nofg] cls_loss={:.3}", f64::try_from(&cls_loss).unwrap_or(f64::NAN));
        }
        return cls_loss + zero * (w.box_gain + w.dfl) * (if one2one { 1.0 } else { 1.0 });
    }

    // 框：除以每锚 stride → 格单位
    let target_bboxes_g = &target_bboxes / stride_per_anchor.unsqueeze(0); // (B, A, 4)
    let pred_b = pd_bboxes.transpose(1, 2); // (B, A, 4) xywh 像素 → 除 stride
    let pred_b_g = &pred_b / stride_per_anchor.unsqueeze(0);

    let weight = target_scores
        .sum_dim_intlist(Some(&[2i64][..]), false, None)
        .reshape([-1])
        .index_select(0, &fg_mask.reshape([-1]).nonzero().squeeze_dim(1))
        .unsqueeze(1); // (N, 1)

    let pred_fg = select_fg(&pred_b_g, &fg_mask);
    let target_fg = select_fg(&target_bboxes_g, &fg_mask);

    // CIoU（xyxy 格式成对）
    let pxy = xywh_to_xyxy(&pred_fg);
    let txy = xywh_to_xyxy(&target_fg);
    let iou = iou_pairwise(&pxy, &txy); // (N, 1)
    let iou_term: Tensor = (1.0 - iou) * &weight;
    let iou_loss = iou_term.sum_dim_intlist(None::<&[i64]>, false, Kind::Float) / target_scores_sum.shallow_clone() * w.box_gain;

    // DFL
    let dfl_term = match dfl_apply_model(model, &preds.boxes, &fg_mask, &target_bboxes_g, points, &weight) {
        Some(d) => d / target_scores_sum.shallow_clone() * w.dfl,
        None => scores.sum_dim_intlist(None::<&[i64]>, false, Kind::Float) * 0.0, // reg_max=1（yolo26 无 DFL）
    };

    // 数值安全：各分量截到 ≥ 0（退化框时 CIoU 罚项可能越界）
    let cls_loss = cls_loss.clamp_min(0.0);
    let iou_loss = iou_loss.clamp_min(0.0);
    let dfl_term = dfl_term.clamp_min(0.0);

    cls_loss + iou_loss + dfl_term
}

struct AssignRef {
    target_bboxes: Tensor,
    target_scores: Tensor,
    fg_mask: Tensor,
}

fn dfl_apply(model: &YoloModel, boxes: &Tensor) -> Tensor {
    let last = model.layers.last().unwrap();
    if let ul_nn::parse::Module::Head(ul_nn::parse::HeadMod::Detect(dh)) = &last.module {
        if let Some(d) = &dh.dfl {
            return d.forward(boxes);
        }
    }
    boxes.shallow_clone()
}

/// DFL 损失：目标 ltrb 量化 + CE（官方 DFLoss）。
fn dfl_apply_model(
    model: &YoloModel,
    pred_dist: &Tensor,   // (B, 4*rm, A) 未积分分布
    fg_mask: &Tensor,     // (B, A) bool
    target_bboxes: &Tensor, // (B, A, 4) 格单位 xyxy
    points: &Tensor,      // (2, A) anchor 中心（格单位）
    weight: &Tensor,      // (N, 1) 每 fg 锚的分数权重
) -> Option<Tensor> {
    let last = model.layers.last().unwrap();
    let reg_max = if let ul_nn::parse::Module::Head(ul_nn::parse::HeadMod::Detect(dh)) = &last.module {
        dh.reg_max
    } else {
        return None;
    };
    if reg_max <= 1 {
        return None;
    }
    let device = pred_dist.device();
    // 选 fg 分布：(N, 4*rm) → (N, 4, rm)
    let flat = pred_dist.reshape([pred_dist.size()[0] * pred_dist.size()[2], -1]);
    let sel = flat.index_select(0, &fg_mask.reshape([-1]).nonzero().squeeze_dim(1));
    let n = sel.size()[0];
    let pd = sel.reshape([n, 4, reg_max]);

    // 目标 ltrb：anchor 中心到目标框的格距离
    let t = select_fg(target_bboxes, fg_mask); // (N, 4) xyxy 格单位
    let anc = select_fg_points(points, fg_mask); // (N, 2)
    let ax = anc.select(1, 0).unsqueeze(1);
    let ay = anc.select(1, 1).unsqueeze(1);
    let l = &ax - t.select(1, 0).unsqueeze(1);
    let t_ = &ay - t.select(1, 1).unsqueeze(1);
    let r = t.select(1, 2).unsqueeze(1) - ax;
    let b = t.select(1, 3).unsqueeze(1) - ay;
    let ltrb = Tensor::cat(&[l, t_, r, b], 1); // (N, 4)
    let target = ltrb.clamp(0.0, (reg_max - 1) as f64 - 0.01);

    let pos1 = target.to_kind(Kind::Int64); // (N, 4)
    let pos2 = &pos1 + 1;
    let w1 = pos2.to_kind(Kind::Float) - target;
    let w2 = 1.0 - &w1;
    let ce1 = cross_entropy(&pd.reshape([n * 4, reg_max]), &pos1.reshape([-1])).reshape([n, 4]);
    let ce2 = cross_entropy(&pd.reshape([n * 4, reg_max]), &pos2.reshape([-1])).reshape([n, 4]);
    let dfl = ((&ce1 * w1 + &ce2 * w2) as Tensor).mean_dim(Some(&[1i64][..]), true, Kind::Float); // (N, 1)
    Some((dfl * weight).sum_dim_intlist(None::<&[i64]>, false, Kind::Float))
}

fn select_fg_points(points: &Tensor, fg: &Tensor) -> Tensor {
    // points (2, A) → (N, 2)
    let a = points.size()[1];
    let pts = points.transpose(0, 1).reshape([a, 2]);
    pts.index_select(0, &fg.reshape([-1]).nonzero().squeeze_dim(1))
}

fn select_fg(t: &Tensor, fg: &Tensor) -> Tensor {
    // t (B, A, ...) → (N, ...)
    let b = t.size()[0];
    let a = t.size()[1];
    let rest: Vec<i64> = t.size()[2..].to_vec();
    let flat = t.reshape([b * a, -1]);
    let sel = flat.index_select(0, &fg.reshape([-1]).nonzero().squeeze_dim(1));
    let mut shape = vec![sel.size()[0]];
    shape.extend(rest);
    sel.reshape(shape)
}

fn xywh_to_xyxy(x: &Tensor) -> Tensor {
    let parts = x.split_with_sizes(&[2, 2], 1);
    Tensor::cat(&[&parts[0] - &parts[1] / 2.0, &parts[0] + &parts[1] / 2.0], 1)
}

/// 数值稳定 BCE（sum 归约）。
/// BCE with logits：max(z,0) − z·t + log(1+exp(−|z|))，等价 softplus(z) − t·z。
fn bce_sum(pred: &Tensor, target: &Tensor) -> Tensor {
    let p = pred.clamp(-100.0, 100.0);
    let softplus = (&p.neg().exp() + 1.0).log();
    let l = &softplus - target * &p;
    l.sum_dim_intlist(None::<&[i64]>, false, Kind::Float)
}

/// CE（类别索引目标）。
fn cross_entropy(logits: &Tensor, target: &Tensor) -> Tensor {
    let lse = logits.log_softmax(-1, Kind::Float);
    let n = logits.size()[0];
    let picked = lse.gather(1, &target.unsqueeze(1), false).squeeze_dim(1);
    -picked * (1.0 / n as f64)
}

/// batch.labels → (gt_labels (bs, m, 1), gt_bboxes (bs, m, 4) xyxy 像素, mask_gt (bs, m, 1) bool)。
pub fn prepare_targets(batch: &DetBatch) -> (Tensor, Tensor, Tensor) {
    let device = batch.img.device();
    let s = batch.imgsz as f64;
    let lb = batch.labels.shallow_clone();
    if lb.size()[0] == 0 {
        return (
            Tensor::zeros([batch.batch_size, 1, 1], (Kind::Float, device)),
            Tensor::zeros([batch.batch_size, 1, 4], (Kind::Float, device)),
            Tensor::zeros([batch.batch_size, 1, 1], (Kind::Bool, device)),
        );
    }
    let idx = lb.select(1, 0).to_kind(Kind::Int64);
    let cls = lb.select(1, 1).reshape([-1, 1]).to_kind(Kind::Float);
    let xywh = lb.narrow(1, 2, 4);
    let xy = xywh.narrow(1, 0, 2) * s;
    let wh = xywh.narrow(1, 2, 2) * s;
    let xyxy = Tensor::cat(&[&xy - &wh / 2.0, &xy + &wh / 2.0], 1);

    let mut max_boxes = 0i64;
    let mut per: Vec<(Tensor, Tensor)> = Vec::new();
    for b in 0..batch.batch_size {
        let mask = idx.eq(b);
        let sel = mask.nonzero().squeeze_dim(1);
        let n = sel.size()[0];
        let (l, bx) = if n > 0 {
            (cls.index_select(0, &sel), xyxy.index_select(0, &sel))
        } else {
            (Tensor::zeros([0, 1], (Kind::Float, device)), Tensor::zeros([0, 4], (Kind::Float, device)))
        };
        max_boxes = max_boxes.max(n);
        per.push((l, bx));
    }
    let mut lbl_v = Vec::new();
    let mut bb_v = Vec::new();
    let mut mgt_v = Vec::new();
    for (l, bx) in &per {
        let n = l.size()[0];
        let pad = max_boxes - n;
        let (fl, fb) = if pad > 0 {
            (
                Tensor::cat(&[l.shallow_clone(), Tensor::zeros([pad, 1], (Kind::Float, device))], 0),
                Tensor::cat(&[bx.shallow_clone(), Tensor::zeros([pad, 4], (Kind::Float, device))], 0),
            )
        } else {
            (l.shallow_clone(), bx.shallow_clone())
        };
        let m = fb.sum_dim_intlist(Some(&[1i64][..]), true, None).gt(0.0);
        lbl_v.push(fl.reshape([1, max_boxes, 1]));
        bb_v.push(fb.reshape([1, max_boxes, 4]));
        mgt_v.push(m.reshape([1, max_boxes, 1]));
    }
    (
        Tensor::cat(&lbl_v, 0),
        Tensor::cat(&bb_v, 0),
        Tensor::cat(&mgt_v, 0),
    )
}

/// 分类损失：CE + label smoothing（v8ClassificationLoss）。
pub fn classification_loss(logits: &Tensor, targets: &Tensor, smoothing: f64) -> Tensor {
    let n = logits.size()[0];
    let nc = logits.size()[1];
    let lse = logits.log_softmax(-1, Kind::Float);
    let picked = lse.gather(1, &targets.unsqueeze(1), false).squeeze_dim(1);
    let onehot = Tensor::zeros([n, nc], (Kind::Float, logits.device()));
    let onehot = onehot.scatter(1, &targets.unsqueeze(1), &Tensor::ones([n, 1], (Kind::Float, logits.device())));
    let smooth = onehot * (1.0 - smoothing) + (nc as f64 - 1.0).recip() * smoothing;
    let ce = -(smooth * lse).sum_dim_intlist(Some(&[1i64][..]), false, Kind::Float);
    ce.mean_dim(Some(&[0i64][..]), false, Kind::Float).squeeze()
}

/// bbox_iou 成对版（ops::bbox_iou 的 (N,1) 包装）：取对角。
pub fn bbox_iou_pairwise(box1: &Tensor, box2: &Tensor) -> Tensor {
    let iou = bbox_iou(box1, box2, false, false, true, 1e-7);
    iou.diag(0).unsqueeze(1)
}
