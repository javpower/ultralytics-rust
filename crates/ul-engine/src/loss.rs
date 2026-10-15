//! 损失：v8DetectionLoss（TAL + BCE + 成对 CIoU + DFL 量化 CE）、E2E 双头（o2m topk10 +
//! o2o topk1 相加）、reg_max=1 的归一化 L1 路径、分类 CE。
//! 对齐 ultralytics/utils/loss.py（8.4.157 v8DetectionLoss/BboxLoss/DFLoss/E2EDetectLoss）。

use tch::{Kind, Tensor};


use ul_nn::head::HeadOutput;
use ul_nn::model::YoloModel;
use ul_nn::ops::{bbox_iou, dist2bbox, make_anchors};
use ul_nn::tal::TaskAlignedAssigner;

/// 训练批（检测/分割/姿态/旋转框共用）：标签 (m, 6+) = [img_idx, cls, cx, cy, w, h, ...] 归一化。
/// obb=true 时标签 5 列为 xywhr（角度弧度）。
pub struct DetBatch {
    pub img: Tensor,    // (B, 3, S, S) 0..1 RGB
    pub labels: Tensor, // (m, >=5)
    pub batch_size: i64,
    pub imgsz: i64,
    /// 分割：合并实例掩码（B, S, S）f32，像素值 = 实例序号+1（0=背景）。
    pub masks: Option<Tensor>,
    /// 姿态：关键点 (m, K*dim) 归一化（与 labels 行序一致）。
    pub kpts: Option<Tensor>,
    /// OBB 标志：labels 为 xywhr。
    pub obb: bool,
}

impl DetBatch {
    pub fn detect(img: Tensor, labels: Tensor, batch_size: i64, imgsz: i64) -> Self {
        DetBatch { img, labels, batch_size, imgsz, masks: None, kpts: None, obb: false }
    }
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

/// 分量日志（含增益、不含 batch_size——官方 items 语义）。
#[derive(Clone, Copy, Default)]
pub struct LossParts {
    pub box_: f64,
    pub cls: f64,
    pub dfl: f64,
    pub seg: f64,
    pub pose: f64,
    pub kobj: f64,
    pub angle: f64,
}

impl LossParts {
    pub fn add(&mut self, o: &LossParts) {
        self.box_ += o.box_;
        self.cls += o.cls;
        self.dfl += o.dfl;
        self.seg += o.seg;
        self.pose += o.pose;
        self.kobj += o.kobj;
        self.angle += o.angle;
    }
}

/// 分配中间量（seg/pose 附加损失复用）。
pub struct AssignInfo {
    pub fg_mask: Tensor,       // (B, A) bool
    pub target_gt_idx: Tensor, // (B, A) i64
    pub target_bboxes: Tensor, // (B, A, 4) xyxy 像素（分配器输出）
    pub anchor_points: Tensor, // (2, A) 格单位
    pub stride_tensor: Tensor, // (1, A)
}

/// COCO17 OKS sigma（官方 OKS_SIGMA / 10）。
pub fn oks_sigmas(k: usize) -> Vec<f64> {
    const C17: [f64; 17] = [
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
    ];
    if k == 17 {
        C17.to_vec()
    } else {
        vec![1.0; k]
    }
}

/// 检测损失：total 供反传，parts 供日志，assign 供 seg/pose 附加损失。
pub struct DetLoss {
    pub total: Tensor,
    pub parts: LossParts,
    pub assign: Option<AssignInfo>,
}

/// 成对 CIoU（tal.py bbox_iou 语义：(N,4)×(N,4) → (N,1)）。
pub fn iou_pairwise(pred: &Tensor, target: &Tensor) -> Tensor {
    bbox_iou_pairwise(pred, target)
}

/// 检测损失入口。`assigner`（topk=10）用于 one2many，`assigner_o2o`（topk=1）用于 one2one
/// （官方 E2EDetectLoss：两分支损失直接相加）。
#[allow(clippy::too_many_arguments)]
pub fn v8_detection_loss(
    model: &YoloModel,
    w: LossW,
    assigner: &TaskAlignedAssigner,
    assigner_o2o: &TaskAlignedAssigner,
    out: &HeadOutput,
    batch: &DetBatch,
    e2e: bool,
) -> DetLoss {
    let bs = batch.batch_size as f64;
    let (o2m, o2o) = match out {
        HeadOutput::TrainDual(a, b) => (a.as_ref(), Some(b.as_ref())),
        HeadOutput::Train(a) => (a.as_ref(), None),
        _ => panic!("loss requires training head output"),
    };
    let feats = &o2m.feats;
    let (points, strides) = make_anchors(feats, &model.stride, 0.5);

    let (mut loss, mut parts, assign) = detect_branch_loss(model, assigner, o2m, batch, &points, &strides, w);
    if e2e {
        if let Some(o2o) = o2o {
            let (l, p, _a2) = detect_branch_loss(model, assigner_o2o, o2o, batch, &points, &strides, w);
            loss = loss + l;
            parts.add(&p);
        }
    }
    DetLoss {
        total: loss * bs,
        // 官方 items 语义：分量不乘 batch_size
        parts,
        assign: Some(assign),
    }
}

/// 单分支检测损失（TAL + cls BCE + CIoU + DFL/L1）。返回（标量损失, 分量, 分配信息）。
#[allow(clippy::type_complexity)]
pub fn detect_branch_loss(
    model: &YoloModel,
    assigner: &TaskAlignedAssigner,
    preds: &ul_nn::head::RawPreds,
    batch: &DetBatch,
    points: &Tensor,  // (2, A) 格单位
    strides: &Tensor, // (1, A)
    w: LossW,
) -> (Tensor, LossParts, AssignInfo) {
    let feats = &preds.feats;

    // 解码预测：官方 bbox_decode——DFL 积分 + dist2bbox(xywh=False) → xyxy 格单位
    let raw_boxes = dfl_apply(model, &preds.boxes); // (B, 4, A) 格单位 ltrb（积分后）
    let (points2, strides2) = make_anchors(feats, &model.stride, 0.5);
    let pd_grid = dist2bbox(&raw_boxes, &points2, false, 1); // (B, 4, A) xyxy 格单位
    let pd_pixels = &pd_grid * &strides2; // (B, 4, A) xyxy 像素（分配器输入）

    // 目标
    let (gt_labels, gt_bboxes, mask_gt) = prepare_targets(batch);

    // 官方语义：锚点与预测框均以像素坐标参与分配
    let anc_pixels = &points2 * &strides2; // (2, A)
    let scores_t = preds.scores.sigmoid().transpose(1, 2); // (B, A, nc)
    let out = assigner.assign(&scores_t, &pd_pixels.transpose(1, 2), &anc_pixels, &gt_labels, &gt_bboxes, &mask_gt);
    let assign_info = AssignInfo {
        fg_mask: out.fg_mask.shallow_clone(),
        target_gt_idx: out.target_gt_idx.shallow_clone(),
        target_bboxes: out.target_bboxes.shallow_clone(),
        anchor_points: points2.shallow_clone(),
        stride_tensor: strides2.shallow_clone(),
    };
    let fg_mask = out.fg_mask;
    let target_scores_sum =
        out.target_scores.sum_dim_intlist(None::<&[i64]>, false, Kind::Float).clamp_min(1.0);

    // cls BCE（BCE-with-logits 需要原始 logits）：sum / target_scores_sum × cls
    let logits_t = preds.scores.transpose(1, 2).contiguous();
    let cls_loss = bce_sum(&logits_t, &out.target_scores) / target_scores_sum.shallow_clone() * w.cls;

    let fg_sum = fg_mask.sum_dim_intlist(None::<&[i64]>, false, Kind::Float);
    // 注意：fg_sum 是 Float 标量——Vec::<i64>::try_from 对 kind 不匹配会失败（曾致 box/dfl 永远为零）
    let has_fg = f64::try_from(&fg_sum).unwrap_or(0.0) > 0.0;
    if !has_fg {
        // 无前景：仅 cls；box/dfl 记零（保持梯度路径有内容：乘 0）
        let zero = preds.scores.sum_dim_intlist(None::<&[i64]>, false, Kind::Float) * 0.0;
        let total = cls_loss.shallow_clone() + zero * (w.box_gain + w.dfl);
        let parts = LossParts {
            box_: 0.0,
            cls: f64::try_from(&cls_loss).unwrap_or(f64::NAN),
            dfl: 0.0,
            ..Default::default()
        };
        return (total, parts, assign_info);
    }

    // 框：格单位 xyxy（官方 pred_bboxes 保持格单位；仅 target_bboxes/stride）
    let stride_pa = strides2.transpose(0, 1); // (A, 1)
    let stride_b = stride_pa.unsqueeze(0); // (1, A, 1)
    let t_grid = &out.target_bboxes / &stride_b; // (B, A, 4)
    let p_grid = pd_grid.transpose(1, 2); // (B, A, 4) 已是格单位

    let weight = out
        .target_scores
        .sum_dim_intlist(Some(&[2i64][..]), false, None)
        .reshape([-1])
        .index_select(0, &fg_mask.reshape([-1]).nonzero().squeeze_dim(1))
        .unsqueeze(1); // (N, 1)

    let pred_fg = select_fg(&p_grid, &fg_mask);
    let target_fg = select_fg(&t_grid, &fg_mask);
    let iou = iou_pairwise(&pred_fg, &target_fg); // (N, 1) xyxy 成对 CIoU
    let iou_term: Tensor = (1.0 - &iou) * &weight;
    let iou_loss = iou_term.sum_dim_intlist(None::<&[i64]>, false, Kind::Float)
        / target_scores_sum.shallow_clone()
        * w.box_gain;

    // DFL（reg_max>1）或归一化 L1（reg_max=1，官方 use_dfl=False 分支）
    let dfl_term = match dfl_apply_model(model, &preds.boxes, &fg_mask, &t_grid, points, &weight) {
        Some(d) => d / target_scores_sum.shallow_clone() * w.dfl,
        None => {
            l1_reg1(&raw_boxes, &fg_mask, &t_grid, points, &stride_pa, &weight, batch.imgsz as f64)
                / target_scores_sum.shallow_clone()
                * w.dfl
        }
    };

    // 数值安全：各分量截到 ≥ 0（退化框时 CIoU 罚项可能越界）
    let cls_final = cls_loss.clamp_min(0.0);
    let iou_final = iou_loss.clamp_min(0.0);
    let dfl_final = dfl_term.clamp_min(0.0);

    let parts = LossParts {
        box_: f64::try_from(&iou_final).unwrap_or(f64::NAN),
        cls: f64::try_from(&cls_final).unwrap_or(f64::NAN),
        dfl: f64::try_from(&dfl_final).unwrap_or(f64::NAN),
        ..Default::default()
    };
    let total = cls_final + iou_final + dfl_final;
    (total, parts, assign_info)
}

/// reg_max=1 的 L1 损失：目标 ltrb ×stride 归一化到 [0,1]（x 除宽、y 除高），预测同样处理后 L1。
/// 官方 BboxLoss else 分支（bbox2dist 无 clamp）。
#[allow(clippy::too_many_arguments)]
fn l1_reg1(
    raw_boxes: &Tensor,     // (B, 4, A) 格单位 ltrb
    fg_mask: &Tensor,       // (B, A) bool
    target_bboxes: &Tensor, // (B, A, 4) 格单位 xyxy
    points: &Tensor,        // (2, A) 格单位
    stride_pa: &Tensor,     // (A, 1)
    weight: &Tensor,        // (N, 1)
    imgsz: f64,
) -> Tensor {
    let a = raw_boxes.size()[2];
    // 目标 ltrb（格单位，无 clamp——官方 bbox2dist(reg_max=None)）
    let t = select_fg(target_bboxes, fg_mask); // (N, 4)
    let anc = select_fg_points(points, fg_mask); // (N, 2)
    let ax = anc.select(1, 0).unsqueeze(1);
    let ay = anc.select(1, 1).unsqueeze(1);
    let l = &ax - t.select(1, 0).unsqueeze(1);
    let t_ = &ay - t.select(1, 1).unsqueeze(1);
    let r = t.select(1, 2).unsqueeze(1) - ax;
    let b = t.select(1, 3).unsqueeze(1) - ay;
    let target_ltrb = Tensor::cat(&[l, t_, r, b], 1); // (N, 4)

    // 每 fg 锚的 stride（格 → 像素）
    let st = select_fg(&stride_pa.reshape([1, a, 1]), fg_mask); // (N, 1)
    let target_px = &target_ltrb * &st;
    let pred_px = select_fg(&raw_boxes.transpose(1, 2), fg_mask) * &st; // (N, 4)
    // 归一化：x 分量 (l, r) 除宽，y 分量 (t, b) 除高
    let norm = |x: &Tensor| {
        let p = x.split_with_sizes(&[1, 1, 1, 1], 1);
        Tensor::cat(&[&p[0] / imgsz, &p[1] / imgsz, &p[2] / imgsz, &p[3] / imgsz], 1)
    };
    let (tn, pn) = (norm(&target_px), norm(&pred_px));
    let l1 = (pn - tn)
        .abs()
        .mean_dim(Some(&[1i64][..]), true, Kind::Float); // (N, 1)
    (l1 * weight).sum_dim_intlist(None::<&[i64]>, false, Kind::Float)
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

/// DFL 损失：目标 ltrb 量化 + CE（官方 DFLoss）。reg_max<=1 返回 None（调用方走 L1）。
fn dfl_apply_model(
    model: &YoloModel,
    pred_dist: &Tensor,     // (B, 4*rm, A) 未积分分布
    fg_mask: &Tensor,       // (B, A) bool
    target_bboxes: &Tensor, // (B, A, 4) 格单位 xyxy
    points: &Tensor,        // (2, A) anchor 中心（格单位）
    weight: &Tensor,        // (N, 1) 每 fg 锚的分数权重
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
    // 选 fg 分布：(N, 4*rm) → (N, 4, rm)。
    // 注意 (B, 4*rm, A) 必须先转成 (B, A, 4*rm) 再按锚展平，否则行内通道被打散。
    let b = pred_dist.size()[0];
    let a = pred_dist.size()[2];
    let flat = pred_dist.transpose(1, 2).contiguous().reshape([b * a, -1]);
    let sel = flat.index_select(0, &fg_mask.reshape([-1]).nonzero().squeeze_dim(1));
    let n = sel.size()[0];
    let pd = sel.reshape([n, 4, reg_max]);

    // 目标 ltrb：anchor 中心到目标框的格距离（bbox2dist clamp 15.99）
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
    // points (2, A) 每图共享；(B, A) 的 fg 逐图选取
    let (b, a) = (fg.size()[0], fg.size()[1]);
    let pts = points.transpose(0, 1).reshape([1, a, 2]).expand([b, a, 2], true);
    select_fg(&pts, fg)
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

/// 数值稳定 BCE（sum 归约）。
/// BCE with logits：l = softplus(z) − z·t（softplus(z) = max(z,0) + log(1+e^{−|z|})，
/// 与 nn.BCEWithLogitsLoss 同一稳定形式；此前误写为 softplus(−z)，梯度方向反向）。
fn bce_sum(pred: &Tensor, target: &Tensor) -> Tensor {
    let p = pred.clamp(-100.0, 100.0);
    let softplus = p.f_softplus().unwrap();
    let l = &softplus - target * &p;
    l.sum_dim_intlist(None::<&[i64]>, false, Kind::Float)
}

/// CE（类别索引目标，reduction="none"——官方 DFLoss 语义，不做均值）。
fn cross_entropy(logits: &Tensor, target: &Tensor) -> Tensor {
    let lse = logits.log_softmax(-1, Kind::Float);
    let picked = lse.gather(1, &target.unsqueeze(1), false).squeeze_dim(1);
    -picked
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
    (Tensor::cat(&lbl_v, 0), Tensor::cat(&bb_v, 0), Tensor::cat(&mgt_v, 0))
}

/// 分类损失：CE + label smoothing（v8ClassificationLoss）。
pub fn classification_loss(logits: &Tensor, targets: &Tensor, smoothing: f64) -> Tensor {
    let n = logits.size()[0];
    let nc = logits.size()[1];
    let lse = logits.log_softmax(-1, Kind::Float);
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

// ---------------- 分割损失（v8SegmentationLoss） ----------------

/// 实例掩码损失（官方 calculate_segmentation_loss + single_mask_loss）：
/// proto 双线性上采样到掩码分辨率，einsum 出预测掩码，sigmoid-BCE 经 bbox 裁剪后 / 面积，
/// 全批求和 ÷ fg 锚数。返回标量（未乘 gain、未乘 bs）。
pub fn segmentation_mask_loss(
    info: &AssignInfo,
    proto: &Tensor,      // (B, nm, mh, mw)
    pred_masks: &Tensor, // (B, nm, A) 系数（训练原始输出）
    gt_masks: &Tensor,   // (B, S, S) f32，值 = 实例序号+1
    imgsz: i64,
) -> Tensor {
    let (bs, nm, mh, mw) = (proto.size()[0], proto.size()[1], proto.size()[2], proto.size()[3]);
    let (gh, gw) = (gt_masks.size()[1], gt_masks.size()[2]);
    let imgsz_f = Tensor::from_slice(&[imgsz as f32]);
    let mut loss = proto.zeros_like().sum_dim_intlist(None::<&[i64]>, false, Kind::Float);
    let mut total_fg = 0f64;
    for i in 0..bs {
        let fg = info.fg_mask.get(i); // (A) bool
        let nfg = f64::try_from(&fg.to_kind(Kind::Float).sum_dim_intlist(None::<&[i64]>, false, Kind::Float)).unwrap_or(0.0);
        if nfg < 1.0 {
            loss = &loss + (proto.get(i) * 0.0).sum_dim_intlist(None::<&[i64]>, false, Kind::Float)
                + (pred_masks.get(i) * 0.0).sum_dim_intlist(None::<&[i64]>, false, Kind::Float);
            continue;
        }
        total_fg += nfg;
        let nz = fg.nonzero().squeeze_dim(1);
        let mask_idx = info.target_gt_idx.get(i).index_select(0, &nz); // (nf,)
        let coef = pred_masks.get(i).transpose(0, 1).index_select(0, &nz); // (nf, nm)
        // proto 上采样到掩码分辨率（官方 bilinear）
        let proto_i: Tensor = if (proto.size()[2], proto.size()[3]) != (gh, gw) {
            proto
                .get(i)
                .unsqueeze(0)
                .f_upsample_bilinear2d(&[gh, gw], false, None, None)
                .unwrap()
                .squeeze_dim(0)
        } else {
            proto.get(i)
        }; // (nm, gh, gw)
        let proto_flat = proto_i.reshape([nm, gh * gw]);
        let pred_mask = coef.matmul(&proto_flat).sigmoid(); // (nf, gh*gw)
        // gt = (masks[i] == mask_idx+1)
        let mrow = gt_masks.get(i).reshape([gh * gw]); // (gh*gw,)
        let one = mask_idx + 1;
        let gt = (mrow.unsqueeze(0) - &one.unsqueeze(1)).abs().lt(1e-3).to_kind(Kind::Float); // (nf, gh*gw)
        // bbox 裁剪 + 面积（官方：归一化 xyxy → 掩码分辨率）
        let bb = info.target_bboxes.get(i).index_select(0, &nz); // (nf, 4) xyxy 像素
        let sc = Tensor::from_slice(&[gw as f32, gh as f32, gw as f32, gh as f32]);
        let mxyxy = &bb / &sc; // 归一化
        let mxyxy = &mxyxy * &sc;
        let (x1, y1, x2, y2) = (
            mxyxy.narrow(1, 0, 1),
            mxyxy.narrow(1, 1, 1),
            mxyxy.narrow(1, 2, 1),
            mxyxy.narrow(1, 3, 1),
        );
        // 官方 marea：归一化 xywh 的 wh 积（0.3×0.4≈0.12 量级）
        let area = (bb.narrow(1, 2, 1) / &imgsz_f) * (bb.narrow(1, 3, 1) / &imgsz_f) + 1e-9; // (nf,1)
        // crop：bbox 外置 0
        let pr = Tensor::arange(gh, (Kind::Float, pred_mask.device())).unsqueeze(0); // (1,gh)
        let pc = Tensor::arange(gw, (Kind::Float, pred_mask.device())).unsqueeze(0);
        let crop = pc
            .ge_tensor(&x1)
            .logical_and(&pc.le_tensor(&x2))
            .unsqueeze(1)
            .logical_and(&pr.ge_tensor(&y1).logical_and(&pr.le_tensor(&y2)).unsqueeze(2))
            .to_kind(Kind::Float); // (nf, gh, gw)
        let nf = nz.size()[0];
        let pm_img = pred_mask.reshape([nf, gh, gw]);
        let gt_img = gt.reshape([nf, gh, gw]);
        // sigmoid 已在 pred_mask 处做；BCE with prob：l = -t·log(p+ε) - (1-t)·log(1-p+ε)
        let eps = 1e-7f64;
        let one_m: Tensor = 1.0 - &gt_img;
        let pm1: Tensor = (1.0 - &pm_img) + eps;
        let bce = -(gt_img * (&pm_img + eps).log() + one_m * pm1.log());
        let bce = &bce * &crop;
        let per = bce.mean_dim(Some(&[1i64][..]), false, Kind::Float).mean_dim(Some(&[1i64][..]), false, Kind::Float); // (nf,)
        loss = &loss + (per / &area).sum_dim_intlist(None::<&[i64]>, false, Kind::Float);
    }
    let fg_t = Tensor::from_slice(&[total_fg as f32]);
    loss / &fg_t
}

// ---------------- 姿态损失（v8PoseLoss） ----------------

/// 关键点损失：返回 (pose_loss, kobj_loss)（未乘 gain、未乘 bs）。
/// 官方语义：预测 kpt 在格单位解码（×2 + anchor−0.5），gt 像素 ÷ 每 fg 锚 stride；
/// e = d/((2σ)²·(area+ε)·2)，按 (1−e^−e)·mask 均值；kobj = BCE(原始 v 通道, mask)。
pub fn pose_keypoints_loss(
    info: &AssignInfo,
    pred_kpts_raw: &Tensor, // (B, nk, A) 原始输出，nk = K*dim，每 kpt [x,y,(v)] 连续
    kpt_dim: i64,
    gt_kpts: &Tensor,   // (N, K*dim) 归一化
    batch_idx: &Tensor, // (N,) i64（与 labels 行序一致）
    imgsz: i64,
) -> (Tensor, Tensor) {
    let b = pred_kpts_raw.size()[0];
    let nk = pred_kpts_raw.size()[1];
    let a = pred_kpts_raw.size()[2];
    let k = nk / kpt_dim;
    let ki = k;
    let ku = k as usize;
    // 解码到格单位 (B, A, K, dim)
    let pa = pred_kpts_raw.transpose(1, 2).contiguous().reshape([b, a, k, kpt_dim]);
    let anc = info.anchor_points.transpose(0, 1).reshape([1, a, 1, 2]); // (1,A,1,2)
    let xy = pa.narrow(3, 0, 2) * 2.0 + (anc - 0.5); // (B, A, K, 2) 格单位

    let zero = xy.zeros_like().sum_dim_intlist(None::<&[i64]>, false, Kind::Float);
    let (mut kpt_sum, mut kobj_sum) = (zero.shallow_clone(), zero);
    let mut total_fg = 0f64;
    let mut any_fg = false;

    for i in 0..b {
        let fg = info.fg_mask.get(i);
        let nfg = f64::try_from(&fg.to_kind(Kind::Float).sum_dim_intlist(None::<&[i64]>, false, Kind::Float)).unwrap_or(0.0);
        if nfg < 1.0 {
            continue;
        }
        any_fg = true;
        total_fg += nfg;
        let nz = fg.nonzero().squeeze_dim(1);
        let mask_idx = info.target_gt_idx.get(i).index_select(0, &nz); // (nf_i,)
        let rows = batch_idx.eq(i).to_kind(Kind::Int64).nonzero().squeeze_dim(1); // (n_i,)
        let sel_rows = rows.index_select(0, &mask_idx); // (nf_i,)
        let krow = gt_kpts.index_select(0, &sel_rows).to_kind(Kind::Float).reshape([sel_rows.size()[0], k, kpt_dim]);
        let st = info
            .stride_tensor
            .transpose(0, 1)
            .reshape([1, a, 1])
            .index_select(1, &nz); // (1, nf, 1)
        let sz_t = Tensor::from_slice(&[imgsz as f32]);
        let mut gxy = krow.narrow(2, 0, 2) * &sz_t; // (nf,K,2) 像素
        gxy = &gxy / st.reshape([st.size()[1], 1, 1]); // 格单位
        let gv = if kpt_dim == 3 { Some(krow.narrow(2, 2, 1).shallow_clone()) } else { None };
        let tb = info.target_bboxes.get(i).index_select(0, &nz) / st.reshape([st.size()[1], 1]); // (nf,4) 格 xyxy
        let bw = tb.narrow(1, 2, 1) - tb.narrow(1, 0, 1);
        let bh = tb.narrow(1, 3, 1) - tb.narrow(1, 1, 1);
        let area = bw * bh; // (nf,1)
        let pk = xy.get(i).index_select(0, &nz); // (nf,K,2)
        let d = (pk.narrow(2, 0, 1) - gxy.narrow(2, 0, 1)).pow_tensor_scalar(2.0)
            + (pk.narrow(2, 1, 1) - gxy.narrow(2, 1, 1)).pow_tensor_scalar(2.0);
        let sig = oks_sigmas(ku);
        let sig_t = Tensor::from_slice(&sig.iter().map(|x| *x as f32).collect::<Vec<_>>()).reshape([1, k]);
        let ar = (area.shallow_clone() + 1e-9).reshape([area.size()[0], 1]);
        let d2: Tensor = d.reshape([d.size()[0], k]); // (nf, K)
        let sig2: Tensor = (sig_t * 2.0).pow_tensor_scalar(2.0);
        let e = &d2 / &sig2 / &ar / 2.0; // d/((2σ)²·area·2)
        let kmask: Tensor = match &gv {
            Some(v) => v.reshape([v.size()[0], k]).gt(0.0).to_kind(Kind::Float),
            None => Tensor::ones_like(&e),
        };
        let kft: Tensor = k as f32 / (kmask.gt(0.0).to_kind(Kind::Float).sum_dim_intlist(Some(&[1i64][..]), false, Kind::Float) + 1e-9); // (nf,)
        let exp_term: Tensor = (-&e).exp() * -1.0 + 1.0;
        let term = &kft.unsqueeze(1) * &(exp_term * &kmask); // (nf,K)
        kpt_sum = &kpt_sum + term.sum_dim_intlist(None::<&[i64]>, false, Kind::Float);
        if let Some(vt0) = gv.as_ref() {
            let mut parts = Vec::new();
            for kj in 0..ki {
                let ch = pred_kpts_raw.get(i).narrow(0, kj * kpt_dim + 2, 1); // (1, A)
                parts.push(ch);
            }
            let pv_all = Tensor::cat(&parts, 0); // (K, A)
            let pv_fg = pv_all.transpose(0, 1).index_select(0, &nz); // (nf, K)
            let vt: Tensor = vt0.reshape([vt0.size()[0], k]).shallow_clone();
            let pv_c = pv_fg.clamp(1e-7, 1.0 - 1e-7);
            let ones_c: Tensor = &pv_c * 0.0 + 1.0;
            let one_v: Tensor = &ones_c - &vt;
            let pv1: Tensor = &ones_c - &pv_c;
            let ones_c: Tensor = &pv_c * 0.0 + 1.0;
            let one_v: Tensor = &ones_c - &vt;
            let pv1: Tensor = &ones_c - &pv_c;
            let bce = -(&vt * pv_c.log() + one_v * pv1.log());
            let cnt = Tensor::from_slice(&[nfg as f32 * k as f32]);
            kobj_sum = &kobj_sum + bce.sum_dim_intlist(None::<&[i64]>, false, Kind::Float) / &cnt;
        }
    }
    if !any_fg {
        kpt_sum = &kpt_sum + pred_kpts_raw.narrow(1, 0, 0).sum_dim_intlist(None::<&[i64]>, false, Kind::Float);
        return (kpt_sum, kobj_sum);
    }
    let cnt = Tensor::from_slice(&[total_fg as f32 * k as f32]);
    (kpt_sum / &cnt, kobj_sum / &cnt)
}

// ---------------- OBB 损失（v8OBBLoss） ----------------

/// OBB 单分支损失：旋转分配 + probiou + rbox2dist DFL + 角度损失。
/// preds.extra = 角度 (B,1,A)，已过 (sigmoid−0.25)π。
pub fn obb_branch_loss(
    model: &YoloModel,
    assigner: &TaskAlignedAssigner,
    preds: &ul_nn::head::RawPreds,
    batch: &DetBatch,
    w: LossW,
) -> (Tensor, LossParts) {
    use ul_nn::ops::{dist2rbox, make_anchors, probiou_pairwise, rbox2dist};
    let feats = &preds.feats;
    let raw_boxes = dfl_apply(model, &preds.boxes); // (B,4,A) 格 ltrb
    let (points2, strides2) = make_anchors(feats, &model.stride, 0.5);
    let angle = preds.extra.as_ref().expect("OBB requires angle preds"); // (B,1,A)
    if std::env::var("UL_LOSS_DEBUG").is_ok() {
        eprintln!("[obb0] raw={:?} ang={:?} labels={:?}", raw_boxes.size(), angle.size(), batch.labels.size());
    }
    let pred_xywhr = Tensor::cat(&[dist2rbox(&raw_boxes, angle, &points2, 1), angle.shallow_clone()], 1); // (B,5,A) 格

    let bs = batch.batch_size;
    let imgsz = batch.imgsz as f64;
    let (gt_labels, gt_xywhr_px, mask_gt) = prepare_targets_obb(&batch.labels, bs, imgsz);
    if std::env::var("UL_LOSS_DEBUG").is_ok() {
        eprintln!("[obb1] targets ok gt_l={:?} gt_b={:?}", gt_labels.size(), gt_xywhr_px.size());
    }

    let anc_pixels = &points2 * &strides2; // (2, A)
    let scores_t = preds.scores.sigmoid().transpose(1, 2);
    // 分配器输入：[:4]×stride（像素 xywhr），角度通道 ×1
    let ones = strides2.zeros_like() + 1.0;
    let st5 = Tensor::stack(&[&strides2, &strides2, &strides2, &strides2, &ones], 1); // (1,5,A)
    let assign_in = &pred_xywhr * &st5;
    if std::env::var("UL_LOSS_DEBUG").is_ok() {
        eprintln!("[obb] scores={:?} assign_in={:?} anc={:?} gt_l={:?} gt_b={:?} mg={:?}",
            scores_t.size(), assign_in.transpose(1, 2).size(), anc_pixels.size(),
            gt_labels.size(), gt_xywhr_px.size(), mask_gt.size());
    }
    let out = assigner.assign_rotated(
        &scores_t,
        &assign_in.transpose(1, 2),
        &anc_pixels,
        &gt_labels,
        &gt_xywhr_px,
        &mask_gt,
    );
    if std::env::var("UL_LOSS_DEBUG").is_ok() {
        eprintln!("[obb] assign ok fg={} tb={:?}", 
            f64::try_from(&out.fg_mask.to_kind(Kind::Float).sum_dim_intlist(None::<&[i64]>, false, Kind::Float)).unwrap_or(0.0),
            out.target_bboxes.size());
    }
    let fg_mask = out.fg_mask;
    let tss = out.target_scores.sum_dim_intlist(None::<&[i64]>, false, Kind::Float).clamp_min(1.0);
    if std::env::var("UL_OBB_DEBUG").is_ok() {
        let f = |t: &Tensor| f64::try_from(&t.to_kind(Kind::Float).sum_dim_intlist(None::<&[i64]>, false, Kind::Float)).unwrap_or(0.0);
        eprintln!("[obb] fg={} ts_sum={:.6}", f(&fg_mask), f(&out.target_scores));
    }

    let logits_t = preds.scores.transpose(1, 2).contiguous();
    let cls_loss = bce_sum(&logits_t, &out.target_scores) / tss.shallow_clone() * w.cls;

    let nfg = f64::try_from(&fg_mask.to_kind(Kind::Float).sum_dim_intlist(None::<&[i64]>, false, Kind::Float)).unwrap_or(0.0);
    if nfg < 1.0 {
        let z = preds.scores.sum_dim_intlist(None::<&[i64]>, false, Kind::Float) * 0.0;
        let parts = LossParts { cls: f64::try_from(&cls_loss).unwrap_or(f64::NAN), ..Default::default() };
        let total = cls_loss + z * (w.box_gain + w.dfl);
        return (total, parts);
    }

    let ones_a = strides2.zeros_like() + 1.0;
    let st5g = Tensor::stack(&[&strides2, &strides2, &strides2, &strides2, &ones_a], 1); // (1,5,A)
    let t_grid = out.target_bboxes.transpose(1, 2) / &st5g; // (B,5,A) 格 xywhr
    let weight = out
        .target_scores
        .sum_dim_intlist(Some(&[2i64][..]), false, None)
        .reshape([-1])
        .index_select(0, &fg_mask.reshape([-1]).nonzero().squeeze_dim(1))
        .unsqueeze(1); // (N,1)

    let nz = fg_mask.reshape([-1]).nonzero().squeeze_dim(1);
    let p_flat = pred_xywhr.transpose(1, 2).contiguous().reshape([bs * pred_xywhr.size()[2], 5]);
    let t_flat = t_grid.transpose(1, 2).contiguous().reshape([bs * t_grid.size()[2], 5]);
    let p_fg = p_flat.index_select(0, &nz);
    let t_fg = t_flat.index_select(0, &nz);
    let iou = probiou_pairwise(&p_fg, &t_fg, false, 0.01); // (N,1)
    let iou_term: Tensor = (1.0 - &iou) * &weight;
    let iou_loss = iou_term.sum_dim_intlist(None::<&[i64]>, false, Kind::Float) / tss.shallow_clone() * w.box_gain;

    // DFL：rbox2dist 目标
    let reg_max = match model.layers.last().map(|l| &l.module) {
        Some(ul_nn::parse::Module::Head(ul_nn::parse::HeadMod::Detect(dh))) => dh.reg_max,
        _ => 16,
    };
    let n = p_fg.size()[0];
    let pd = pred_dist_rows(&preds.boxes, &fg_mask); // (N,4,rm)
    let anc_fg = select_fg_points(&points2, &fg_mask); // (N,2)
    let t_xywh = t_fg.narrow(1, 0, 4); // (N,4) 格 xywh
    let t_ang = t_fg.narrow(1, 4, 1); // (N,1)
    let ltrb = rbox2dist(&t_xywh, &anc_fg.transpose(0, 1), &t_ang, reg_max);
    let pos1 = ltrb.to_kind(Kind::Int64);
    let pos2 = &pos1 + 1;
    let w1 = &pos2.to_kind(Kind::Float) - &ltrb;
    let w2 = 1.0 - &w1;
    let rm = pd.size()[2];
    let ce1 = cross_entropy(&pd.reshape([n * 4, rm]), &pos1.reshape([-1])).reshape([n, 4]);
    let ce2 = cross_entropy(&pd.reshape([n * 4, rm]), &pos2.reshape([-1])).reshape([n, 4]);
    let dfl = ((&ce1 * w1 + &ce2 * w2) as Tensor).mean_dim(Some(&[1i64][..]), true, Kind::Float);
    let dfl_term = (dfl * &weight).sum_dim_intlist(None::<&[i64]>, false, Kind::Float) / tss.shallow_clone() * w.dfl;

    // 角度损失（官方 calculate_angle_loss，lambda=3）
    let w_gt = t_fg.narrow(1, 2, 1);
    let h_gt = t_fg.narrow(1, 3, 1);
    let p_th = p_fg.narrow(1, 4, 1);
    let t_th = t_fg.narrow(1, 4, 1);
    let log_ar = ((w_gt + 1e-9) / (h_gt + 1e-9)).log();
    let scale_w = (-log_ar.pow_tensor_scalar(2.0) / 9.0).exp();
    let dth = p_th - t_th;
    let pi_t = Tensor::from_slice(&[std::f32::consts::PI]);
    let dth_w = &dth - ((&dth / &pi_t).round() * &pi_t);
    let ang = dth_w.sin().pow_tensor_scalar(2.0) * &scale_w * &weight;
    let angle_loss = ang.sum_dim_intlist(None::<&[i64]>, false, Kind::Float) / tss;

    let parts = LossParts {
        box_: f64::try_from(&iou_loss.clamp_min(0.0)).unwrap_or(f64::NAN),
        cls: f64::try_from(&cls_loss.clamp_min(0.0)).unwrap_or(f64::NAN),
        dfl: f64::try_from(&dfl_term.clamp_min(0.0)).unwrap_or(f64::NAN),
        angle: f64::try_from(&angle_loss.clamp_min(0.0)).unwrap_or(f64::NAN),
        ..Default::default()
    };
    let total = cls_loss.clamp_min(0.0) + iou_loss.clamp_min(0.0) + dfl_term.clamp_min(0.0) + angle_loss.clamp_min(0.0);
    (total, parts)
}

/// (B,4*rm,A) → fg 选择 (N,4,rm)。
fn pred_dist_rows(pred_dist: &Tensor, fg_mask: &Tensor) -> Tensor {
    let b = pred_dist.size()[0];
    let a = pred_dist.size()[2];
    let c = pred_dist.size()[1];
    let rm = c / 4;
    let flat = pred_dist.transpose(1, 2).contiguous().reshape([b * a, c]);
    let sel = flat.index_select(0, &fg_mask.reshape([-1]).nonzero().squeeze_dim(1));
    let n = sel.size()[0];
    sel.reshape([n, 4, rm])
}

/// OBB 目标准备：labels (N,6+) [idx, cls, cx, cy, w, h, r] 归一化 → 像素，过滤 wh≤2。
/// 返回 (labels (bs,m,1), xywhr (bs,m,5) 像素, mask_gt)。
fn prepare_targets_obb(lb: &Tensor, bs: i64, imgsz: f64) -> (Tensor, Tensor, Tensor) {
    let device = lb.device();
    if lb.size()[0] == 0 {
        return (
            Tensor::zeros([bs, 1, 1], (Kind::Float, device)),
            Tensor::zeros([bs, 1, 5], (Kind::Float, device)),
            Tensor::zeros([bs, 1, 1], (Kind::Bool, device)),
        );
    }
    let idx = lb.select(1, 0).to_kind(Kind::Int64);
    let cls = lb.select(1, 1).reshape([-1, 1]).to_kind(Kind::Float);
    // 官方 preprocess：仅 cx,cy,w,h ×imgsz（列 0..4），角度保持
    let cxywh = lb.narrow(1, 2, 4) * imgsz;
    let ang = lb.narrow(1, 6, 1);
    let whr = Tensor::cat(&[cxywh, ang], 1);
    // 官方过滤 rw,rh≥2（wh 列，labels 里为第 4/5 列）
    let keep = whr.narrow(1, 4 - 2, 1).gt(2.0).logical_and(&whr.narrow(1, 5 - 2, 1).gt(2.0)).to_kind(Kind::Int64);
    let keep_rows = keep.reshape([-1]).nonzero().squeeze_dim(1);
    if std::env::var("UL_LOSS_DEBUG").is_ok() {
        eprintln!("[obbt] keep_rows={:?} idx={:?}", keep_rows.size(), idx.size());
    }
    let cls = cls.index_select(0, &keep_rows);
    let whr = whr.index_select(0, &keep_rows);
    let idx = idx.index_select(0, &keep_rows);
    if std::env::var("UL_LOSS_DEBUG").is_ok() {
        eprintln!("[obbt] filtered ok");
    }

    let mut max_boxes = 0i64;
    let mut per: Vec<(Tensor, Tensor)> = Vec::new();
    for b in 0..bs {
        let sel = idx.eq(b).nonzero().squeeze_dim(1);
        let ni = sel.size()[0];
        let pair = if ni > 0 {
            (cls.index_select(0, &sel), whr.index_select(0, &sel))
        } else {
            (Tensor::zeros([0, 1], (Kind::Float, device)), Tensor::zeros([0, 5], (Kind::Float, device)))
        };
        max_boxes = max_boxes.max(ni);
        per.push(pair);
    }
    let mut lbl_v = Vec::new();
    let mut bb_v = Vec::new();
    let mut mgt_v = Vec::new();
    for (l, bx) in &per {
        let pad = max_boxes - l.size()[0];
        let (fl, fb) = if pad > 0 {
            (
                Tensor::cat(&[l.shallow_clone(), Tensor::zeros([pad, 1], (Kind::Float, device))], 0),
                Tensor::cat(&[bx.shallow_clone(), Tensor::zeros([pad, 5], (Kind::Float, device))], 0),
            )
        } else {
            (l.shallow_clone(), bx.shallow_clone())
        };
        let m = fb.sum_dim_intlist(Some(&[1i64][..]), true, None).gt(0.0);
        lbl_v.push(fl.reshape([1, max_boxes, 1]));
        bb_v.push(fb.reshape([1, max_boxes, 5]));
        mgt_v.push(m.reshape([1, max_boxes, 1]));
    }
    (Tensor::cat(&lbl_v, 0), Tensor::cat(&bb_v, 0), Tensor::cat(&mgt_v, 0))
}

impl LossW {
    /// 官方 default.yaml：pose=12.0, kobj=1.0, angle=1.0。
    pub fn pose_gain(&self) -> f64 {
        12.0
    }
    pub fn kobj_gain(&self) -> f64 {
        1.0
    }
    pub fn angle_gain(&self) -> f64 {
        1.0
    }
}
